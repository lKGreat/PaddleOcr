using System.Text.Json;
using System.Text;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace PaddleOcr.Export;

public sealed class NativeExporter
{
    private readonly ILogger _logger;

    public NativeExporter(ILogger logger)
    {
        _logger = logger;
    }

    public string ExportNative(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);
        var ckpt = ResolveCheckpoint(cfg);
        if (!File.Exists(ckpt))
        {
            throw new FileNotFoundException($"checkpoint not found: {ckpt}");
        }

        var paramsPath = Path.Combine(cfg.SaveInferenceDir, "inference.pdiparams");
        var jsonPath = Path.Combine(cfg.SaveInferenceDir, "inference.json");
        var ymlPath = Path.Combine(cfg.SaveInferenceDir, "inference.yml");

        File.Copy(ckpt, paramsPath, overwrite: true);
        File.WriteAllText(jsonPath, BuildInferenceJson(cfg, ckpt));
        File.WriteAllText(ymlPath, BuildInferenceYaml(cfg));
        CopyDictIfNeeded(cfg);

        var manifest = BuildManifest(cfg, "paddle-infer-shim", DateTime.UtcNow, "inference.pdiparams", ckpt, ckpt, cfg.SaveModelDir, null, null);
        WriteManifest(
            cfg.SaveInferenceDir,
            manifest);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);
        _logger.LogInformation("Exported paddle-infer shim: {Dir}", cfg.SaveInferenceDir);
        return paramsPath;
    }

    public string ExportOnnx(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);
        var onnx = ResolveOnnxSource(cfg);
        if (onnx is null || !File.Exists(onnx))
        {
            throw new FileNotFoundException("No ONNX source found. Set Global.checkpoints or Global.pretrained_model to an .onnx file.");
        }

        var target = Path.Combine(cfg.SaveInferenceDir, "inference.onnx");
        File.Copy(onnx, target, overwrite: true);
        var io = GetOnnxIoMetadata(target);
        var manifest = BuildManifest(cfg, "onnx", DateTime.UtcNow, "inference.onnx", null, onnx, null, io.Inputs, io.Outputs);
        WriteManifest(
            cfg.SaveInferenceDir,
            manifest);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);
        _logger.LogInformation("Exported ONNX model: {Path}", target);
        return target;
    }

    public string ExportPaddleStatic(ExportConfigView cfg)
    {
        return ExportPaddleStatic(cfg, cfg.StaticEquivalence, cfg.PaddleExportSource);
    }

    public string ExportPaddleStatic(ExportConfigView cfg, string staticEquivalence, string paddleExportSource)
    {
        staticEquivalence = NormalizeMode(staticEquivalence, "strict", "compatible");
        paddleExportSource = NormalizeMode(paddleExportSource, "paddle", "onnx");

        if (paddleExportSource.Equals("onnx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "Pure native C# mode does not support onnx->paddle static conversion yet. " +
                "Use --paddle_export_source paddle with an existing Paddle static source.");
        }

        if (staticEquivalence.Equals("strict", StringComparison.OrdinalIgnoreCase))
        {
            return ExportPaddleStaticStrict(cfg);
        }

        return ExportPaddleStaticCompatibleFromPaddle(cfg);
    }

    private string ExportPaddleStaticCompatibleFromPaddle(ExportConfigView cfg)
    {
        var outPath = ExportPaddleStaticStrict(cfg);
        var manifestPath = Path.Combine(cfg.SaveInferenceDir, "manifest.json");
        var old = JsonSerializer.Deserialize<ExportManifest>(File.ReadAllText(manifestPath));
        if (old is null)
        {
            throw new InvalidOperationException("failed to load manifest for compatible rewrite");
        }

        var rewritten = old with
        {
            StaticEquivalence = "compatible",
            ConversionChain = "paddle_export_native"
        };
        WriteManifest(cfg.SaveInferenceDir, rewritten);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);
        return outPath;
    }

    private string ExportPaddleStaticStrict(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);
        var source = ResolvePaddleSource(cfg);
        if (source is null)
        {
            throw new FileNotFoundException(
                "No Paddle static source found for strict export. " +
                "Set Global.pretrained_model to a Paddle inference dir (inference.json + inference.pdiparams), " +
                "or set Global.checkpoints to inference.json.");
        }
        var (graphSource, paramsSource) = PickPaddleStaticArtifacts(source);
        var graphExt = Path.GetExtension(graphSource).ToLowerInvariant();
        var graphTarget = graphExt == ".json"
            ? Path.Combine(cfg.SaveInferenceDir, "inference.json")
            : Path.Combine(cfg.SaveInferenceDir, "inference.pdmodel");
        var paramsTarget = Path.Combine(cfg.SaveInferenceDir, "inference.pdiparams");
        File.Copy(graphSource, graphTarget, overwrite: true);
        File.Copy(paramsSource, paramsTarget, overwrite: true);

        var ymlPath = Path.Combine(cfg.SaveInferenceDir, "inference.yml");
        var sourceYml = ResolvePaddleYamlPath(source);
        if (!string.IsNullOrWhiteSpace(sourceYml) && File.Exists(sourceYml))
        {
            File.Copy(sourceYml, ymlPath, overwrite: true);
        }
        else if (!File.Exists(ymlPath))
        {
            File.WriteAllText(ymlPath, BuildInferenceYaml(cfg));
        }
        CopyDictIfNeeded(cfg);

        var manifest = BuildManifest(
            cfg,
            "paddle-static",
            DateTime.UtcNow,
            Path.GetFileName(graphTarget),
            checkpoint: cfg.Checkpoints,
            source: source,
            sourceDirectory: Directory.Exists(source) ? source : Path.GetDirectoryName(source),
            onnxInputs: null,
            onnxOutputs: null,
            staticEquivalence: "strict",
            conversionChain: "paddle_export_native");
        WriteManifest(cfg.SaveInferenceDir, manifest);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);
        _logger.LogInformation("Exported strict paddle static model: {Dir}", cfg.SaveInferenceDir);
        return graphTarget;
    }

    public string ConvertJsonToPdmodel(string jsonModelDir, string outputDir)
    {
        if (!Directory.Exists(jsonModelDir))
        {
            throw new DirectoryNotFoundException($"json model dir not found: {jsonModelDir}");
        }

        Directory.CreateDirectory(outputDir);
        var srcJson = Path.Combine(jsonModelDir, "inference.json");
        var srcParams = Path.Combine(jsonModelDir, "inference.pdiparams");
        if (!File.Exists(srcJson) || !File.Exists(srcParams))
        {
            throw new FileNotFoundException("Expect inference.json and inference.pdiparams in json model dir.");
        }

        var dstModel = Path.Combine(outputDir, "inference.pdmodel");
        var dstParams = Path.Combine(outputDir, "inference.pdiparams");
        File.Copy(srcJson, dstModel, overwrite: true);
        File.Copy(srcParams, dstParams, overwrite: true);

        WriteManifest(
            outputDir,
            new ExportManifest(
                SchemaVersion: "1.0",
                Format: "pdmodel-shim",
                ModelType: "unknown",
                CreatedAtUtc: DateTime.UtcNow,
                ArtifactFile: "inference.pdmodel",
                Checkpoint: null,
                Source: jsonModelDir,
                SourceDirectory: jsonModelDir,
                LabelList: [],
                RecCharDictPath: null,
                ClsImageShape: [],
                DetInputSize: null,
                Compatibility: new ExportCompatibility("1.x", "shim", true),
                OnnxInputs: [],
                OnnxOutputs: [],
                StaticEquivalence: "compatible",
                ConversionChain: "json_pdmodel_shim"));
        ValidateManifestOrThrow(outputDir);
        _logger.LogInformation("Converted json model dir to pdmodel shim: {Dir}", outputDir);
        return dstModel;
    }

    public bool ValidateJsonModelDir(string jsonModelDir, out string message)
    {
        if (!Directory.Exists(jsonModelDir))
        {
            message = $"json model dir not found: {jsonModelDir}";
            return false;
        }

        var srcJson = Path.Combine(jsonModelDir, "inference.json");
        var srcParams = Path.Combine(jsonModelDir, "inference.pdiparams");
        if (!File.Exists(srcJson) || !File.Exists(srcParams))
        {
            message = "missing inference.json or inference.pdiparams";
            return false;
        }

        message = jsonModelDir;
        return true;
    }

    public bool ValidateManifestFile(string dir, out string message)
    {
        var manifestPath = Path.Combine(dir, "manifest.json");
        if (!File.Exists(manifestPath))
        {
            message = "manifest.json not found";
            return false;
        }

        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize<ExportManifest>(json);
            if (manifest is null)
            {
                message = "manifest parse failed";
                return false;
            }

            if (string.IsNullOrWhiteSpace(manifest.SchemaVersion) ||
                string.IsNullOrWhiteSpace(manifest.Format) ||
                string.IsNullOrWhiteSpace(manifest.ArtifactFile))
            {
                message = "manifest missing required fields";
                return false;
            }

            if (manifest.Compatibility is null ||
                string.IsNullOrWhiteSpace(manifest.Compatibility.ManifestSemVer) ||
                string.IsNullOrWhiteSpace(manifest.Compatibility.Runtime))
            {
                message = "manifest missing compatibility fields";
                return false;
            }

            if (!manifest.Compatibility.ManifestSemVer.StartsWith("1.", StringComparison.Ordinal))
            {
                message = $"unsupported manifest compatibility: {manifest.Compatibility.ManifestSemVer}";
                return false;
            }

            message = manifestPath;
            return true;
        }
        catch (Exception ex)
        {
            message = ex.Message;
            return false;
        }
    }

    private static void WriteManifest(string dir, object manifest)
    {
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(dir, "manifest.json"), json);
    }

    private static ExportManifest BuildManifest(
        ExportConfigView cfg,
        string format,
        DateTime createdAtUtc,
        string artifactFile,
        string? checkpoint,
        string? source,
        string? sourceDirectory,
        IReadOnlyList<ExportTensorInfo>? onnxInputs,
        IReadOnlyList<ExportTensorInfo>? onnxOutputs,
        string staticEquivalence = "compatible",
        string conversionChain = "native")
    {
        return new ExportManifest(
            SchemaVersion: "1.0",
            Format: format,
            ModelType: cfg.ModelType,
            CreatedAtUtc: createdAtUtc,
            ArtifactFile: artifactFile,
            Checkpoint: checkpoint,
            Source: source,
            SourceDirectory: sourceDirectory,
            LabelList: cfg.LabelList,
            RecCharDictPath: cfg.RecCharDictPath,
            ClsImageShape: cfg.ClsImageShape,
            DetInputSize: cfg.DetInputSize,
            Compatibility: new ExportCompatibility("1.x", "native", true),
            OnnxInputs: onnxInputs ?? [],
            OnnxOutputs: onnxOutputs ?? [],
            StaticEquivalence: staticEquivalence,
            ConversionChain: conversionChain);
    }

    private static (IReadOnlyList<ExportTensorInfo> Inputs, IReadOnlyList<ExportTensorInfo> Outputs) GetOnnxIoMetadata(string onnxPath)
    {
        using var session = new InferenceSession(onnxPath);
        var inputs = session.InputMetadata
            .Select(kv => new ExportTensorInfo(kv.Key, kv.Value.Dimensions.ToArray()))
            .ToList();
        var outputs = session.OutputMetadata
            .Select(kv => new ExportTensorInfo(kv.Key, kv.Value.Dimensions.ToArray()))
            .ToList();
        return (inputs, outputs);
    }

    private void ValidateManifestOrThrow(string dir)
    {
        if (!ValidateManifestFile(dir, out var message))
        {
            throw new InvalidOperationException($"manifest validation failed: {message}");
        }
    }

    private static string ResolveScriptPath(string scriptName)
    {
        var cwdCandidate = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "scripts", scriptName));
        if (File.Exists(cwdCandidate))
        {
            return cwdCandidate;
        }

        var baseCandidate = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "scripts", scriptName));
        if (File.Exists(baseCandidate))
        {
            return baseCandidate;
        }

        return cwdCandidate;
    }

    private static ProcessRunResult TryRunPython(string args)
    {
        var attempts = new[]
        {
            ("python", args),
            ("py", $"-3 {args}")
        };

        foreach (var (fileName, commandArgs) in attempts)
        {
            var run = RunProcess(fileName, commandArgs);
            if (run.Success)
            {
                return run;
            }
        }

        return RunProcess("python", args);
    }

    private static ProcessRunResult RunProcess(string fileName, string args)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process is null)
            {
                return new ProcessRunResult(false, string.Empty, $"failed to start process: {fileName}");
            }

            var stdOutTask = process.StandardOutput.ReadToEndAsync();
            var stdErrTask = process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            Task.WaitAll(stdOutTask, stdErrTask);
            var stdOut = stdOutTask.Result;
            var stdErr = stdErrTask.Result;
            return new ProcessRunResult(process.ExitCode == 0, stdOut, stdErr);
        }
        catch (Exception ex)
        {
            return new ProcessRunResult(false, string.Empty, ex.Message);
        }
    }

    private static string BuildInferenceJson(ExportConfigView cfg, string checkpointPath)
    {
        var payload = new Dictionary<string, object?>
        {
            ["format"] = "paddle-infer-shim",
            ["source_checkpoint"] = checkpointPath,
            ["model_type"] = cfg.ModelType,
            ["created_at_utc"] = DateTime.UtcNow,
            ["architecture"] = new Dictionary<string, object?>
            {
                ["algorithm"] = GetConfigString(cfg, "Architecture.algorithm", "SVTR_LCNet"),
                ["backbone"] = GetConfigString(cfg, "Architecture.Backbone.name", "unknown"),
                ["neck"] = GetConfigString(cfg, "Architecture.Neck.name", "SequenceEncoder"),
                ["head"] = GetConfigString(cfg, "Architecture.Head.name", "CTCHead")
            }
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
    }

    private static string BuildInferenceYaml(ExportConfigView cfg)
    {
        var (c, h, w) = ParseRecImageShape(cfg);
        var modelName = GetConfigString(cfg, "Global.model_name", "paddleocr_rec");
        var gtcEncode = ResolveGtcEncode(cfg) ?? "NRTRLabelEncode";
        var postName = GetConfigString(cfg, "PostProcess.name", "CTCLabelDecode");
        var dictTokens = LoadDictTokens(cfg.RecCharDictPath);

        var sb = new StringBuilder();
        sb.AppendLine("Global:");
        sb.AppendLine($"  model_name: {QuoteYaml(modelName)}");
        sb.AppendLine("PreProcess:");
        sb.AppendLine("  transform_ops:");
        sb.AppendLine("  - DecodeImage:");
        sb.AppendLine("      img_mode: BGR");
        sb.AppendLine("      channel_first: false");
        sb.AppendLine("  - MultiLabelEncode:");
        sb.AppendLine($"      gtc_encode: {QuoteYaml(gtcEncode)}");
        sb.AppendLine("  - RecResizeImg:");
        sb.AppendLine("      image_shape:");
        sb.AppendLine($"      - {c}");
        sb.AppendLine($"      - {h}");
        sb.AppendLine($"      - {w}");
        sb.AppendLine("  - KeepKeys:");
        sb.AppendLine("      keep_keys:");
        sb.AppendLine("      - image");
        sb.AppendLine("      - label_ctc");
        sb.AppendLine("      - label_gtc");
        sb.AppendLine("      - length");
        sb.AppendLine("      - valid_ratio");
        sb.AppendLine("PostProcess:");
        sb.AppendLine($"  name: {QuoteYaml(postName)}");
        sb.AppendLine("  character_dict:");
        if (dictTokens.Count == 0)
        {
            sb.AppendLine("  - ''");
        }
        else
        {
            foreach (var token in dictTokens)
            {
                sb.AppendLine($"  - {QuoteYaml(token)}");
            }
        }

        return sb.ToString();
    }

    private static (int C, int H, int W) ParseRecImageShape(ExportConfigView cfg)
    {
        var fromD2S = TryGetIntList(cfg.GetByPathPublic("Global.d2s_train_image_shape"));
        if (fromD2S.Count >= 3)
        {
            return (fromD2S[0], fromD2S[1], fromD2S[2]);
        }

        var transforms = cfg.GetByPathPublic("Train.dataset.transforms");
        if (transforms is List<object?> list)
        {
            foreach (var item in list)
            {
                if (item is not Dictionary<string, object?> op)
                {
                    continue;
                }

                if (!op.TryGetValue("RecResizeImg", out var resizeCfgRaw) ||
                    resizeCfgRaw is not Dictionary<string, object?> resizeCfg ||
                    !resizeCfg.TryGetValue("image_shape", out var shapeRaw))
                {
                    continue;
                }

                var shape = TryGetIntList(shapeRaw);
                if (shape.Count >= 3)
                {
                    return (shape[0], shape[1], shape[2]);
                }
            }
        }

        return (3, 48, 320);
    }

    private static IReadOnlyList<int> TryGetIntList(object? raw)
    {
        if (raw is not IList<object?> list)
        {
            return [];
        }

        var parsed = new List<int>(list.Count);
        foreach (var item in list)
        {
            if (int.TryParse(item?.ToString(), out var value))
            {
                parsed.Add(value);
            }
        }

        return parsed;
    }

    private static string? ResolveGtcEncode(ExportConfigView cfg)
    {
        var transforms = cfg.GetByPathPublic("Train.dataset.transforms");
        if (transforms is not List<object?> list)
        {
            return null;
        }

        foreach (var item in list)
        {
            if (item is not Dictionary<string, object?> op ||
                !op.TryGetValue("MultiLabelEncode", out var cfgRaw) ||
                cfgRaw is not Dictionary<string, object?> encodeCfg ||
                !encodeCfg.TryGetValue("gtc_encode", out var gtcRaw))
            {
                continue;
            }

            var gtc = gtcRaw?.ToString();
            if (!string.IsNullOrWhiteSpace(gtc))
            {
                return gtc;
            }
        }

        return null;
    }

    private static string? ResolvePaddleYamlPath(string source)
    {
        if (Directory.Exists(source))
        {
            var yml = Path.Combine(source, "inference.yml");
            return File.Exists(yml) ? yml : null;
        }

        var dir = Path.GetDirectoryName(source);
        if (string.IsNullOrWhiteSpace(dir))
        {
            return null;
        }

        var fromDir = Path.Combine(dir, "inference.yml");
        return File.Exists(fromDir) ? fromDir : null;
    }

    private static (string GraphPath, string ParamsPath) PickPaddleStaticArtifacts(string source)
    {
        source = Path.GetFullPath(source);
        if (Directory.Exists(source))
        {
            var graph = FirstExisting([
                Path.Combine(source, "inference.json"),
                Path.Combine(source, "inference.pdmodel"),
                Path.Combine(source, "model.json"),
                Path.Combine(source, "model.pdmodel")
            ]);
            var parms = FirstExisting([
                Path.Combine(source, "inference.pdiparams"),
                Path.Combine(source, "model.pdiparams"),
                Path.Combine(source, "model.pdparams")
            ]);
            if (!string.IsNullOrWhiteSpace(graph) && !string.IsNullOrWhiteSpace(parms))
            {
                return (graph, parms);
            }

            throw new FileNotFoundException($"missing Paddle graph/params in directory: {source}");
        }

        if (!File.Exists(source))
        {
            throw new FileNotFoundException($"Paddle source does not exist: {source}");
        }

        var ext = Path.GetExtension(source).ToLowerInvariant();
        if (ext is not (".json" or ".pdmodel"))
        {
            throw new InvalidOperationException($"unsupported Paddle static source file: {source}");
        }

        var dir = Path.GetDirectoryName(source) ?? Directory.GetCurrentDirectory();
        var baseName = Path.Combine(dir, Path.GetFileNameWithoutExtension(source));
        var parmsPath = FirstExisting([
            baseName + ".pdiparams",
            Path.Combine(dir, "inference.pdiparams"),
            Path.Combine(dir, "model.pdiparams"),
            Path.Combine(dir, "model.pdparams")
        ]);
        if (string.IsNullOrWhiteSpace(parmsPath))
        {
            throw new FileNotFoundException($"cannot find pdiparams for source: {source}");
        }

        return (source, parmsPath);
    }

    private static string? FirstExisting(IEnumerable<string> candidates)
    {
        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private static List<string> LoadDictTokens(string? dictPath)
    {
        var tokens = new List<string>();
        if (string.IsNullOrWhiteSpace(dictPath) || !File.Exists(dictPath))
        {
            return tokens;
        }

        foreach (var line in File.ReadLines(dictPath))
        {
            var token = line.TrimEnd('\r', '\n');
            if (token.Length > 0)
            {
                tokens.Add(token);
            }
        }

        return tokens;
    }

    private static string GetConfigString(ExportConfigView cfg, string path, string fallback)
    {
        var raw = cfg.GetByPathPublic(path)?.ToString();
        return string.IsNullOrWhiteSpace(raw) ? fallback : raw;
    }

    private static string QuoteYaml(string value)
    {
        return $"'{value.Replace("'", "''")}'";
    }

    private void CopyDictIfNeeded(ExportConfigView cfg)
    {
        if (string.IsNullOrWhiteSpace(cfg.RecCharDictPath) || !File.Exists(cfg.RecCharDictPath))
        {
            return;
        }

        var dest = Path.Combine(cfg.SaveInferenceDir, Path.GetFileName(cfg.RecCharDictPath));
        if (!File.Exists(dest))
        {
            File.Copy(cfg.RecCharDictPath, dest);
        }
    }

    private static string ResolveCheckpoint(ExportConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints))
        {
            return cfg.Checkpoints;
        }

        var best = Path.Combine(cfg.SaveModelDir, "best.pt");
        if (File.Exists(best))
        {
            return best;
        }

        var latest = Path.Combine(cfg.SaveModelDir, "latest.pt");
        if (File.Exists(latest))
        {
            return latest;
        }

        return best;
    }

    private static string? ResolveOnnxSource(ExportConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints) &&
            cfg.Checkpoints.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            return cfg.Checkpoints;
        }

        if (!string.IsNullOrWhiteSpace(cfg.PretrainedModel))
        {
            if (cfg.PretrainedModel.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                return cfg.PretrainedModel;
            }

            var candidate = Path.Combine(cfg.PretrainedModel, "inference.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private static string? ResolvePaddleSource(ExportConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.PretrainedModel))
        {
            if (Directory.Exists(cfg.PretrainedModel))
            {
                return cfg.PretrainedModel;
            }

            if (File.Exists(cfg.PretrainedModel) &&
                (cfg.PretrainedModel.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ||
                 cfg.PretrainedModel.EndsWith(".pdmodel", StringComparison.OrdinalIgnoreCase)))
            {
                return cfg.PretrainedModel;
            }
        }

        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints) &&
            File.Exists(cfg.Checkpoints) &&
            (cfg.Checkpoints.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ||
             cfg.Checkpoints.EndsWith(".pdmodel", StringComparison.OrdinalIgnoreCase)))
        {
            return cfg.Checkpoints;
        }

        return null;
    }

    private static string NormalizeMode(string? value, params string[] allowed)
    {
        var text = value?.Trim().ToLowerInvariant() ?? string.Empty;
        foreach (var candidate in allowed)
        {
            if (text.Equals(candidate, StringComparison.OrdinalIgnoreCase))
            {
                return candidate;
            }
        }

        return allowed[0];
    }
}

internal sealed record ProcessRunResult(bool Success, string StdOut, string StdErr);

public sealed record ExportManifest(
    string SchemaVersion,
    string Format,
    string ModelType,
    DateTime CreatedAtUtc,
    string ArtifactFile,
    string? Checkpoint,
    string? Source,
    string? SourceDirectory,
    IReadOnlyList<string> LabelList,
    string? RecCharDictPath,
    IReadOnlyList<int> ClsImageShape,
    int? DetInputSize,
    ExportCompatibility Compatibility,
    IReadOnlyList<ExportTensorInfo> OnnxInputs,
    IReadOnlyList<ExportTensorInfo> OnnxOutputs,
    string? StaticEquivalence,
    string? ConversionChain);

public sealed record ExportCompatibility(string ManifestSemVer, string Runtime, bool BackwardCompatible);
public sealed record ExportTensorInfo(string Name, IReadOnlyList<int> Dims);
