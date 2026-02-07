using System.Text.Json;
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

        var target = Path.Combine(cfg.SaveInferenceDir, "model.pt");
        File.Copy(ckpt, target, overwrite: true);
        var manifest = BuildManifest(cfg, "torchsharp-native", DateTime.UtcNow, "model.pt", ckpt, null, null, null, null);
        WriteManifest(
            cfg.SaveInferenceDir,
            manifest);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);
        _logger.LogInformation("Exported native model: {Path}", target);
        return target;
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
                OnnxOutputs: []));
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
        IReadOnlyList<ExportTensorInfo>? onnxOutputs)
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
            OnnxOutputs: onnxOutputs ?? []);
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
}

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
    IReadOnlyList<ExportTensorInfo> OnnxOutputs);

public sealed record ExportCompatibility(string ManifestSemVer, string Runtime, bool BackwardCompatible);
public sealed record ExportTensorInfo(string Name, IReadOnlyList<int> Dims);
