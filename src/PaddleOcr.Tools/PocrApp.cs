using Microsoft.Extensions.Logging;
using PaddleOcr.Config;
using PaddleOcr.Core.Cli;
using PaddleOcr.Core.Errors;
using PaddleOcr.Training.Runtime;
using System.Diagnostics;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Tools;

public sealed class PocrApp
{
    private readonly ILogger _logger;
    private readonly ConfigLoader _configLoader;
    private readonly ICommandExecutor _training;
    private readonly ICommandExecutor _inference;
    private readonly ICommandExecutor _export;
    private readonly ICommandExecutor _service;
    private readonly ICommandExecutor _e2e;
    private readonly ICommandExecutor _benchmark;
    private readonly ICommandExecutor _plugin;

    public PocrApp(
        ILogger logger,
        ConfigLoader configLoader,
        ICommandExecutor training,
        ICommandExecutor inference,
        ICommandExecutor export,
        ICommandExecutor service,
        ICommandExecutor e2e,
        ICommandExecutor benchmark,
        ICommandExecutor plugin)
    {
        _logger = logger;
        _configLoader = configLoader;
        _training = training;
        _inference = inference;
        _export = export;
        _service = service;
        _e2e = e2e;
        _benchmark = benchmark;
        _plugin = plugin;
    }

    public async Task<int> RunAsync(string[] args, CancellationToken cancellationToken = default)
    {
        var parsed = CommandLine.Parse(args);
        var context = BuildContext(parsed);

        CommandResult result = parsed.Root.ToLowerInvariant() switch
        {
            "train" => await _training.ExecuteAsync("train", context, cancellationToken),
            "eval" => await _training.ExecuteAsync("eval", context, cancellationToken),
            "export" => await _export.ExecuteAsync("export", context, cancellationToken),
            "export-pdmodel" => await _export.ExecuteAsync("export-pdmodel", context, cancellationToken),
            "export-onnx" => await _export.ExecuteAsync("export-onnx", context, cancellationToken),
            "export-center" => await _export.ExecuteAsync("export-center", context, cancellationToken),
            "infer" => await _inference.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            "convert" => await RunConvertAsync(parsed, context, cancellationToken),
            "config" => await RunConfigAsync(parsed, context),
            "doctor" => await RunDoctorAsync(parsed, context),
            "service" => await _service.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            "e2e" => await _e2e.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            "benchmark" => await _benchmark.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            "plugin" => await _plugin.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            _ => CommandResult.Fail($"Unknown command: {parsed.Root}\n{CommandLine.GetHelp()}")
        };

        if (result.Success)
        {
            _logger.LogInformation(result.Message);
            return 0;
        }

        Console.Error.WriteLine(result.Message);
        return 2;
    }

    private Task<CommandResult> RunConvertAsync(ParsedCommand parsed, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken)
    {
        if (string.Equals(parsed.Sub, "json2pdmodel", StringComparison.OrdinalIgnoreCase))
        {
            return _export.ExecuteAsync("convert:json2pdmodel", context, cancellationToken);
        }

        if (string.Equals(parsed.Sub, "check-json-model", StringComparison.OrdinalIgnoreCase))
        {
            return _export.ExecuteAsync("convert:check-json-model", context, cancellationToken);
        }

        return Task.FromResult(CommandResult.Fail("convert supports: json2pdmodel | check-json-model"));
    }

    private Task<CommandResult> RunConfigAsync(ParsedCommand parsed, PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (string.Equals(parsed.Sub, "check", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(context.ConfigPath))
            {
                return Task.FromResult(CommandResult.Fail("config check requires -c/--config"));
            }

            var ok = ConfigValidator.ValidateBasic(context.Config, out var message);
            return Task.FromResult(ok
                ? CommandResult.Ok($"config check passed: {message}")
                : CommandResult.Fail($"config check failed: {message}"));
        }

        if (string.Equals(parsed.Sub, "diff", StringComparison.OrdinalIgnoreCase))
        {
            if (!context.Options.TryGetValue("--base", out var basePath) ||
                !context.Options.TryGetValue("--target", out var targetPath))
            {
                return Task.FromResult(CommandResult.Fail("config diff requires --base and --target"));
            }

            var left = _configLoader.Load(basePath);
            var right = _configLoader.Load(targetPath);
            var diffs = ConfigValidator.Diff(left, right);
            if (diffs.Count == 0)
            {
                return Task.FromResult(CommandResult.Ok("config diff: no differences"));
            }

            var preview = string.Join('\n', diffs.Take(20));
            return Task.FromResult(CommandResult.Ok($"config diff: {diffs.Count} differences\n{preview}"));
        }

        return Task.FromResult(CommandResult.Fail("config supports: check | diff"));
    }

    private static Task<CommandResult> RunDoctorAsync(ParsedCommand parsed, PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (string.Equals(parsed.Sub, "check-models", StringComparison.OrdinalIgnoreCase))
        {
            var missing = new List<string>();
            ValidateModelPath(context, "--det_model_dir", "Global.det_model_dir", missing);
            ValidateModelPath(context, "--rec_model_dir", "Global.rec_model_dir", missing);
            ValidateModelPath(context, "--cls_model_dir", "Global.cls_model_dir", missing);
            ValidateModelPath(context, "--table_model_dir", "Global.table_model_dir", missing);
            ValidateModelPath(context, "--sr_model_dir", "Global.sr_model_dir", missing);
            ValidateModelPath(context, "--kie_model_dir", "Global.kie_model_dir", missing);
            ValidateModelPath(context, "--ser_model_dir", "Global.ser_model_dir", missing);
            ValidateModelPath(context, "--re_model_dir", "Global.re_model_dir", missing);

            if (missing.Count > 0)
            {
                return Task.FromResult(CommandResult.Fail("doctor check-models failed:\n" + string.Join('\n', missing)));
            }

            return Task.FromResult(CommandResult.Ok("doctor check-models passed"));
        }

        if (string.Equals(parsed.Sub, "parity-table-kie", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(context.ConfigPath))
            {
                return Task.FromResult(CommandResult.Fail("doctor parity-table-kie requires -c/--config"));
            }

            var mode = context.Options.TryGetValue("--mode", out var modeText) && !string.IsNullOrWhiteSpace(modeText)
                ? modeText.Trim().ToLowerInvariant()
                : "all";
            if (mode is not ("all" or "table" or "kie"))
            {
                return Task.FromResult(CommandResult.Fail("doctor parity-table-kie --mode must be one of all|table|kie"));
            }

            var missing = new List<string>();
            if (mode is "all" or "table")
            {
                ValidateRequiredPath(context, "--table_model_dir", "Global.table_model_dir", missing);
                ValidateRequiredPath(context, "--det_model_dir", "Global.det_model_dir", missing);
                ValidateRequiredPath(context, "--rec_model_dir", "Global.rec_model_dir", missing);
                ValidateRequiredPath(context, "--rec_char_dict_path", "Global.rec_char_dict_path", missing);
            }

            if (mode is "all" or "kie")
            {
                ValidateRequiredPath(context, "--kie_model_dir", "Global.kie_model_dir", missing);
                ValidateRequiredPath(context, "--ser_model_dir", "Global.ser_model_dir", missing);
                ValidateRequiredPath(context, "--re_model_dir", "Global.re_model_dir", missing);
                ValidateRequiredPath(context, "--det_model_dir", "Global.det_model_dir", missing);
                ValidateRequiredPath(context, "--rec_model_dir", "Global.rec_model_dir", missing);
                ValidateRequiredPath(context, "--rec_char_dict_path", "Global.rec_char_dict_path", missing);
            }

            if (missing.Count > 0)
            {
                return Task.FromResult(CommandResult.Fail("doctor parity-table-kie failed:\n" + string.Join('\n', missing)));
            }

            return Task.FromResult(CommandResult.Ok($"doctor parity-table-kie passed (mode={mode})"));
        }

        if (string.Equals(parsed.Sub, "train-det-ready", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunDoctorTrainDetReady(context));
        }

        if (string.Equals(parsed.Sub, "train-device", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunDoctorTrainDevice(context));
        }

        if (string.Equals(parsed.Sub, "det-parity", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunDoctorDetParity(context));
        }

        if (string.Equals(parsed.Sub, "verify-rec-paddle", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunDoctorVerifyRecPaddle(context));
        }

        return Task.FromResult(CommandResult.Fail("doctor supports: check-models | parity-table-kie | train-det-ready | train-device | det-parity | verify-rec-paddle"));
    }

    private static void ValidateModelPath(PaddleOcr.Core.Cli.ExecutionContext context, string optionKey, string configPath, ICollection<string> errors)
    {
        string? path = null;
        if (context.Options.TryGetValue(optionKey, out var opt) && !string.IsNullOrWhiteSpace(opt))
        {
            path = opt;
        }
        else
        {
            path = GetConfigValue(context.Config, configPath);
        }

        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        if (!File.Exists(path))
        {
            errors.Add($"{optionKey}: not found -> {path}");
        }
    }

    private static void ValidateRequiredPath(PaddleOcr.Core.Cli.ExecutionContext context, string optionKey, string configPath, ICollection<string> errors)
    {
        string? path = null;
        if (context.Options.TryGetValue(optionKey, out var opt) && !string.IsNullOrWhiteSpace(opt))
        {
            path = opt;
        }
        else
        {
            path = GetConfigValue(context.Config, configPath);
        }

        if (string.IsNullOrWhiteSpace(path))
        {
            errors.Add($"{optionKey}: missing (option or {configPath})");
            return;
        }

        if (!File.Exists(path))
        {
            errors.Add($"{optionKey}: not found -> {path}");
        }
    }

    private static CommandResult RunDoctorTrainDetReady(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (string.IsNullOrWhiteSpace(context.ConfigPath))
        {
            return CommandResult.Fail("doctor train-det-ready requires -c/--config");
        }

        var errors = new List<string>();
        var warnings = new List<string>();
        var modelType = GetConfigValue(context.Config, "Architecture.model_type") ?? string.Empty;
        if (!modelType.Equals("det", StringComparison.OrdinalIgnoreCase))
        {
            warnings.Add($"Architecture.model_type is '{modelType}', expected 'det'");
        }

        var configDir = Path.GetDirectoryName(Path.GetFullPath(context.ConfigPath)) ?? Directory.GetCurrentDirectory();
        var trainLabel = ResolvePath(configDir, GetFirstListItem(context.Config, "Train.dataset.label_file_list"));
        var evalLabel = ResolvePath(configDir, GetFirstListItem(context.Config, "Eval.dataset.label_file_list"));
        var dataDir = ResolvePath(configDir, GetConfigValue(context.Config, "Train.dataset.data_dir"));
        var evalDataDir = ResolvePath(configDir, GetConfigValue(context.Config, "Eval.dataset.data_dir") ?? dataDir);
        var minValidSamples = ParseInt(GetConfigValue(context.Config, "Train.dataset.min_valid_samples"), 1, 1);
        var invalidPolicy = (GetConfigValue(context.Config, "Train.dataset.invalid_sample_policy") ?? "skip").ToLowerInvariant();
        var detSize = ResolveDetInputSize(context.Config);

        if (string.IsNullOrWhiteSpace(trainLabel) || !File.Exists(trainLabel))
        {
            errors.Add($"Train.dataset.label_file_list[0] not found: {trainLabel}");
        }

        if (string.IsNullOrWhiteSpace(evalLabel) || !File.Exists(evalLabel))
        {
            errors.Add($"Eval.dataset.label_file_list[0] not found: {evalLabel}");
        }

        if (string.IsNullOrWhiteSpace(dataDir) || !Directory.Exists(dataDir))
        {
            errors.Add($"Train.dataset.data_dir not found: {dataDir}");
        }

        if (string.IsNullOrWhiteSpace(evalDataDir) || !Directory.Exists(evalDataDir))
        {
            errors.Add($"Eval.dataset.data_dir not found: {evalDataDir}");
        }

        if (detSize <= 0)
        {
            errors.Add("Train.dataset.transforms.ResizeTextImg.size must be > 0");
        }

        if (errors.Count == 0)
        {
            var trainAudit = AuditDetLabel(trainLabel!, dataDir!);
            var evalAudit = AuditDetLabel(evalLabel!, evalDataDir!);
            if (trainAudit.ValidSamples < minValidSamples)
            {
                errors.Add($"train valid det samples {trainAudit.ValidSamples} < min_valid_samples {minValidSamples}");
            }

            if (evalAudit.ValidSamples < minValidSamples)
            {
                errors.Add($"eval valid det samples {evalAudit.ValidSamples} < min_valid_samples {minValidSamples}");
            }

            if (invalidPolicy == "fail" && (trainAudit.InvalidSamples > 0 || evalAudit.InvalidSamples > 0))
            {
                errors.Add($"invalid_sample_policy=fail but invalid lines found: train={trainAudit.InvalidSamples}, eval={evalAudit.InvalidSamples}");
            }

            if (trainAudit.InvalidSamples > 0 && invalidPolicy != "fail")
            {
                warnings.Add($"train invalid lines skipped: {trainAudit.InvalidSamples}");
            }

            if (evalAudit.InvalidSamples > 0 && invalidPolicy != "fail")
            {
                warnings.Add($"eval invalid lines skipped: {evalAudit.InvalidSamples}");
            }

            var summary = $"doctor train-det-ready passed: train_valid={trainAudit.ValidSamples}, eval_valid={evalAudit.ValidSamples}, det_size={detSize}, policy={invalidPolicy}";
            if (errors.Count == 0)
            {
                return warnings.Count == 0 ? CommandResult.Ok(summary) : CommandResult.Ok(summary + "\n" + string.Join('\n', warnings.Select(x => "warn: " + x)));
            }
        }

        return CommandResult.Fail("doctor train-det-ready failed:\n" + string.Join('\n', errors));
    }

    private static CommandResult RunDoctorDetParity(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (string.IsNullOrWhiteSpace(context.ConfigPath))
        {
            return CommandResult.Fail("doctor det-parity requires -c/--config");
        }

        var errors = new List<string>();
        var warnings = new List<string>();
        var modelType = GetConfigValue(context.Config, "Architecture.model_type") ?? string.Empty;
        if (!modelType.Equals("det", StringComparison.OrdinalIgnoreCase))
        {
            warnings.Add($"Architecture.model_type is '{modelType}', expected 'det'");
        }

        var algorithm = (GetConfigValue(context.Config, "Architecture.algorithm")
            ?? GetConfigValue(context.Config, "Global.det_algorithm")
            ?? "DB").Trim();
        var normalizedAlg = algorithm.ToUpperInvariant();
        var supportedAlgorithms = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "DB",
            "DB++",
            "EAST",
            "SAST",
            "PSE",
            "FCE",
            "CT"
        };
        if (!supportedAlgorithms.Contains(algorithm))
        {
            errors.Add($"Architecture.algorithm unsupported for det parity: {algorithm}");
        }

        var expectedPost = normalizedAlg switch
        {
            "DB" => "DBPostProcess",
            "DB++" => "DBPostProcess",
            "EAST" => "EASTPostProcess",
            "SAST" => "SASTPostProcess",
            "PSE" => "PSEPostProcess",
            "FCE" => "FCEPostProcess",
            "CT" => "CTPostProcess",
            _ => string.Empty
        };
        var postName = GetConfigValue(context.Config, "PostProcess.name");
        if (string.IsNullOrWhiteSpace(postName))
        {
            warnings.Add("PostProcess.name missing");
        }
        else if (!string.IsNullOrWhiteSpace(expectedPost) && !postName.Equals(expectedPost, StringComparison.OrdinalIgnoreCase))
        {
            warnings.Add($"PostProcess.name is '{postName}', expected '{expectedPost}' for algorithm '{algorithm}'");
        }

        var boxType = (GetConfigValue(context.Config, "Global.det_box_type") ?? "quad").Trim().ToLowerInvariant();
        if (boxType is not ("quad" or "poly"))
        {
            errors.Add($"Global.det_box_type must be quad|poly, got={boxType}");
        }

        ValidateUnitRange(context, "Global.det_db_thresh", errors);
        ValidateUnitRange(context, "Global.det_db_box_thresh", errors);
        ValidatePositiveNumber(context, "Global.det_db_unclip_ratio", errors);
        ValidatePositiveInt(context, "Global.det_limit_side_len", errors);

        var detModel = GetConfigValue(context.Config, "Global.det_model_dir");
        if (!string.IsNullOrWhiteSpace(detModel) && !File.Exists(detModel))
        {
            errors.Add($"Global.det_model_dir not found: {detModel}");
        }

        var hasTrain = GetConfigNode(context.Config, "Train.dataset") is not null;
        var hasEval = GetConfigNode(context.Config, "Eval.dataset") is not null;
        if (hasTrain && hasEval)
        {
            var gate = RunDoctorTrainDetReady(context);
            if (!gate.Success)
            {
                errors.Add(gate.Message.Replace("doctor train-det-ready failed:\n", string.Empty));
            }
        }
        else
        {
            warnings.Add("Train/Eval datasets missing, skipped train-det-ready gate");
        }

        if (errors.Count > 0)
        {
            return CommandResult.Fail("doctor det-parity failed:\n" + string.Join('\n', errors));
        }

        var summary = $"doctor det-parity passed: algorithm={algorithm}, postprocess={postName ?? "n/a"}";
        if (warnings.Count == 0)
        {
            return CommandResult.Ok(summary);
        }

        return CommandResult.Ok(summary + "\n" + string.Join('\n', warnings.Select(x => "warn: " + x)));
    }

    private static CommandResult RunDoctorTrainDevice(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (string.IsNullOrWhiteSpace(context.ConfigPath))
        {
            return CommandResult.Fail("doctor train-device requires -c/--config");
        }

        try
        {
            var device = (GetConfigValue(context.Config, "Global.device") ?? string.Empty).Trim().ToLowerInvariant();
            var useGpu = bool.TryParse(GetConfigValue(context.Config, "Global.use_gpu"), out var ug) && ug;
            var useAmpRaw = GetConfigValue(context.Config, "Global.use_amp");
            var useAmp = !string.IsNullOrWhiteSpace(useAmpRaw) && bool.TryParse(useAmpRaw, out var ua)
                ? ua
                : (useGpu || device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase));
            var resolved = TrainingDeviceResolver.Resolve(
                device,
                useGpu,
                useAmp,
                () => new CudaRuntimeInfo(cuda.is_available(), cuda.is_available() ? cuda.device_count() : 0));
            var cudaAvailable = cuda.is_available();
            var cudaCount = cudaAvailable ? cuda.device_count() : 0;
            return CommandResult.Ok(
                $"doctor train-device passed: requested={resolved.RequestedDevice}, resolved={resolved.Device.type}, " +
                $"cuda_available={cudaAvailable}, cuda_count={cudaCount}, amp={resolved.UseAmp}, reason={resolved.Reason}");
        }
        catch (Exception ex)
        {
            return CommandResult.Fail($"doctor train-device failed: {ex.Message}");
        }
    }

    private static CommandResult RunDoctorVerifyRecPaddle(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (!context.Options.TryGetValue("--model_dir", out var modelDir) || string.IsNullOrWhiteSpace(modelDir))
        {
            return CommandResult.Fail("doctor verify-rec-paddle requires --model_dir");
        }

        var modelFile = Path.Combine(modelDir, "inference.json");
        var paramsFile = Path.Combine(modelDir, "inference.pdiparams");
        if (!File.Exists(modelFile) || !File.Exists(paramsFile))
        {
            return CommandResult.Fail($"doctor verify-rec-paddle expects inference.json + inference.pdiparams in: {modelDir}");
        }

        var repoRoot = FindRepoRoot();
        var scriptPath = Path.Combine(repoRoot, "scripts", "verify_rec_paddle.py");
        if (!File.Exists(scriptPath))
        {
            return CommandResult.Fail($"doctor verify-rec-paddle script not found: {scriptPath}");
        }

        var pythonExe = context.Options.TryGetValue("--python_exe", out var py) && !string.IsNullOrWhiteSpace(py) ? py : "python";
        var imagePath = context.Options.TryGetValue("--image_path", out var img) ? img : null;
        var saveJson = context.Options.TryGetValue("--save_json", out var json) ? json : null;

        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        psi.ArgumentList.Add(scriptPath);
        psi.ArgumentList.Add("--model_dir");
        psi.ArgumentList.Add(modelDir);

        if (!string.IsNullOrWhiteSpace(imagePath))
        {
            psi.ArgumentList.Add("--image_path");
            psi.ArgumentList.Add(imagePath);
        }

        if (!string.IsNullOrWhiteSpace(saveJson))
        {
            psi.ArgumentList.Add("--save_json");
            psi.ArgumentList.Add(saveJson);
        }

        try
        {
            using var proc = Process.Start(psi);
            if (proc is null)
            {
                return CommandResult.Fail($"doctor verify-rec-paddle failed to start process: {pythonExe}");
            }

            var stdout = proc.StandardOutput.ReadToEnd();
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            if (proc.ExitCode != 0)
            {
                var err = string.IsNullOrWhiteSpace(stderr) ? stdout : stderr;
                return CommandResult.Fail($"doctor verify-rec-paddle failed: {err.Trim()}");
            }

            var preview = stdout.Trim();
            return string.IsNullOrWhiteSpace(preview)
                ? CommandResult.Ok("doctor verify-rec-paddle passed")
                : CommandResult.Ok($"doctor verify-rec-paddle passed:\n{preview}");
        }
        catch (Exception ex)
        {
            return CommandResult.Fail($"doctor verify-rec-paddle failed: {ex.Message}");
        }
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "PaddleOcr.slnx")))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private static string? GetFirstListItem(IReadOnlyDictionary<string, object?> cfg, string path)
    {
        var value = GetConfigNode(cfg, path);
        if (value is List<object?> list && list.Count > 0 && list[0] is not null)
        {
            return list[0]!.ToString();
        }

        return null;
    }

    private static int ResolveDetInputSize(IReadOnlyDictionary<string, object?> cfg)
    {
        var transforms = GetConfigNode(cfg, "Train.dataset.transforms");
        if (transforms is not List<object?> ops)
        {
            return 0;
        }

        foreach (var op in ops)
        {
            if (op is not Dictionary<string, object?> d || !d.TryGetValue("ResizeTextImg", out var resizeObj))
            {
                continue;
            }

            if (resizeObj is Dictionary<string, object?> resizeCfg &&
                resizeCfg.TryGetValue("size", out var sizeObj) &&
                int.TryParse(sizeObj?.ToString(), out var size))
            {
                return size;
            }
        }

        return 0;
    }

    private static string? ResolvePath(string baseDir, string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return path;
        }

        return Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(baseDir, path));
    }

    private static DetLabelAudit AuditDetLabel(string labelFile, string dataDir)
    {
        var audit = new DetLabelAudit();
        foreach (var line in File.ReadLines(labelFile))
        {
            audit.TotalLines++;
            if (string.IsNullOrWhiteSpace(line))
            {
                audit.InvalidSamples++;
                continue;
            }

            var tab = line.IndexOf('\t');
            if (tab <= 0 || tab >= line.Length - 1)
            {
                audit.InvalidSamples++;
                continue;
            }

            var imageRel = line[..tab];
            var json = line[(tab + 1)..];
            var fullPath = Path.IsPathRooted(imageRel) ? imageRel : Path.GetFullPath(Path.Combine(dataDir, imageRel));
            if (!File.Exists(fullPath) || !HasAnyValidPolygon(json))
            {
                audit.InvalidSamples++;
                continue;
            }

            audit.ValidSamples++;
        }

        return audit;
    }

    private static bool HasAnyValidPolygon(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
                return false;
            }

            foreach (var item in doc.RootElement.EnumerateArray())
            {
                if (!item.TryGetProperty("points", out var points) || points.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                var count = 0;
                foreach (var p in points.EnumerateArray())
                {
                    if (p.ValueKind == JsonValueKind.Array && p.GetArrayLength() >= 2 && p[0].TryGetInt32(out _) && p[1].TryGetInt32(out _))
                    {
                        count++;
                    }
                }

                if (count >= 4)
                {
                    return true;
                }
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    private static int ParseInt(string? text, int fallback, int min)
    {
        if (!int.TryParse(text, out var value))
        {
            return fallback;
        }

        return Math.Max(min, value);
    }

    private static void ValidateUnitRange(PaddleOcr.Core.Cli.ExecutionContext context, string configPath, ICollection<string> errors)
    {
        var text = GetConfigValue(context.Config, configPath);
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        if (!float.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var value) ||
            value < 0f || value > 1f)
        {
            errors.Add($"{configPath} must be in [0,1], got={text}");
        }
    }

    private static void ValidatePositiveNumber(PaddleOcr.Core.Cli.ExecutionContext context, string configPath, ICollection<string> errors)
    {
        var text = GetConfigValue(context.Config, configPath);
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        if (!float.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var value) ||
            value <= 0f)
        {
            errors.Add($"{configPath} must be > 0, got={text}");
        }
    }

    private static void ValidatePositiveInt(PaddleOcr.Core.Cli.ExecutionContext context, string configPath, ICollection<string> errors)
    {
        var text = GetConfigValue(context.Config, configPath);
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        if (!int.TryParse(text, out var value) || value <= 0)
        {
            errors.Add($"{configPath} must be > 0, got={text}");
        }
    }

    private static string? GetConfigValue(IReadOnlyDictionary<string, object?> cfg, string path)
    {
        return GetConfigNode(cfg, path)?.ToString();
    }

    private static object? GetConfigNode(IReadOnlyDictionary<string, object?> cfg, string path)
    {
        object? cur = cfg;
        foreach (var part in path.Split('.', StringSplitOptions.RemoveEmptyEntries))
        {
            if (cur is IReadOnlyDictionary<string, object?> rd && rd.TryGetValue(part, out var rv))
            {
                cur = rv;
                continue;
            }

            if (cur is Dictionary<string, object?> d && d.TryGetValue(part, out var dv))
            {
                cur = dv;
                continue;
            }

            return null;
        }

        return cur;
    }

    private PaddleOcr.Core.Cli.ExecutionContext BuildContext(ParsedCommand parsed)
    {
        var cfg = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(parsed.ConfigPath))
        {
            foreach (var pair in _configLoader.Load(parsed.ConfigPath))
            {
                cfg[pair.Key] = pair.Value;
            }
        }

        if (parsed.Overrides.Count > 0)
        {
            var parsedOverrides = OverrideParser.Parse(parsed.Overrides);
            ConfigMerger.MergeInPlace(cfg, parsedOverrides);
        }

        ApplyTrainingRuntimeOptions(parsed.Root, cfg, parsed.Options);

        return new PaddleOcr.Core.Cli.ExecutionContext(
            _logger,
            parsed.RawArgs.ToArray(),
            parsed.ConfigPath,
            cfg,
            parsed.Options,
            parsed.Overrides);
    }

    private static void ApplyTrainingRuntimeOptions(string root, Dictionary<string, object?> cfg, IReadOnlyDictionary<string, string> options)
    {
        if (!root.Equals("train", StringComparison.OrdinalIgnoreCase) &&
            !root.Equals("eval", StringComparison.OrdinalIgnoreCase) &&
            !root.Equals("doctor", StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        var overrides = new List<string>();
        if (options.TryGetValue("--device", out var device) && !string.IsNullOrWhiteSpace(device))
        {
            overrides.Add($"Global.device={device}");
        }

        if (options.TryGetValue("--use_gpu", out var useGpu) && !string.IsNullOrWhiteSpace(useGpu))
        {
            overrides.Add($"Global.use_gpu={useGpu}");
        }

        if (options.TryGetValue("--use_amp", out var useAmp) && !string.IsNullOrWhiteSpace(useAmp))
        {
            overrides.Add($"Global.use_amp={useAmp}");
        }

        if (overrides.Count == 0)
        {
            return;
        }

        var parsedOverrides = OverrideParser.Parse(overrides);
        ConfigMerger.MergeInPlace(cfg, parsedOverrides);
    }
}

internal sealed record DetLabelAudit
{
    public int TotalLines { get; set; }
    public int ValidSamples { get; set; }
    public int InvalidSamples { get; set; }
}
