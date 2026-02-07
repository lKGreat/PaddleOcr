using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;
using PaddleOcr.Inference.Onnx;

namespace PaddleOcr.Inference;

public sealed class InferenceExecutor : ICommandExecutor
{
    private static readonly HashSet<string> Supported = new(StringComparer.OrdinalIgnoreCase)
    {
        "det",
        "rec",
        "cls",
        "e2e",
        "kie",
        "kie-ser",
        "kie-re",
        "table",
        "sr",
        "system"
    };

    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!Supported.Contains(subCommand))
        {
            return Task.FromResult(CommandResult.Fail($"Unsupported infer target: {subCommand}"));
        }

        if (string.IsNullOrWhiteSpace(context.ConfigPath) && !subCommand.Equals("system", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(CommandResult.Fail($"infer {subCommand} requires -c/--config"));
        }

        if (subCommand.Equals("system", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunSystem(context));
        }

        if (subCommand.Equals("det", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunDet(context));
        }

        if (subCommand.Equals("rec", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunRec(context));
        }

        if (subCommand.Equals("cls", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunCls(context));
        }

        if (subCommand.Equals("e2e", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunE2e(context));
        }

        if (subCommand.Equals("sr", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunSr(context));
        }

        if (subCommand.Equals("table", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunTable(context));
        }

        if (subCommand.Equals("kie", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunKie(context, "kie", "kie", "--kie_model_dir"));
        }

        if (subCommand.Equals("kie-ser", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunKie(context, "kie-ser", "kie_ser", "--ser_model_dir"));
        }

        if (subCommand.Equals("kie-re", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(RunKie(context, "kie-re", "kie_re", "--re_model_dir"));
        }

        context.Logger.LogInformation("Running infer {Mode}", subCommand);
        return Task.FromResult(CommandResult.Ok($"infer {subCommand} scaffold executed."));
    }

    private static CommandResult RunSystem(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var recModel = GetOrNull(context, "--rec_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(recModel))
        {
            return CommandResult.Fail("infer system requires --image_dir and --rec_model_dir");
        }

        var output = ResolveOutputDir(context, "system");
        var useOnnx = bool.TryParse(GetOrDefault(context, "--use_onnx", "false"), out var flag) && flag;

        if (!useOnnx)
        {
            return CommandResult.Fail("infer system currently supports --use_onnx=true only.");
        }

        var options = new SystemOnnxOptions(
            imageDir,
            recModel,
            GetOrNull(context, "--det_model_dir"),
            GetOrNull(context, "--cls_model_dir"),
            output,
            GetOrNull(context, "--rec_char_dict_path"),
            ParseBool(GetOrDefault(context, "--use_space_char", "true")),
            ParseCsv(GetOrDefault(context, "--label_list", "0,180")),
            ParseFloat(GetOrDefault(context, "--drop_score", "0.5")),
            ParseFloat(GetOrDefault(context, "--cls_thresh", "0.9")),
            ParseFloat(GetOrDefault(context, "--det_db_thresh", "0.3")));

        if (!File.Exists(options.RecModelPath))
        {
            return CommandResult.Fail($"rec model not found: {options.RecModelPath}");
        }

        if (!string.IsNullOrWhiteSpace(options.DetModelPath) && !File.Exists(options.DetModelPath))
        {
            return CommandResult.Fail($"det model not found: {options.DetModelPath}");
        }

        if (!string.IsNullOrWhiteSpace(options.ClsModelPath) && !File.Exists(options.ClsModelPath))
        {
            return CommandResult.Fail($"cls model not found: {options.ClsModelPath}");
        }

        var runner = new SystemOnnxRunner();
        runner.Run(options);
        context.Logger.LogInformation("Running infer system (onnx)");
        return CommandResult.Ok($"infer system completed. output={options.OutputDir}");
    }

    private static CommandResult RunDet(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var detModel = GetOrNull(context, "--det_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(detModel))
        {
            return CommandResult.Fail("infer det requires --image_dir and --det_model_dir");
        }

        var detAlgorithm = (ResolveString(context, "--det_algorithm", "Global.det_algorithm", "Architecture.algorithm") ?? "DB").Trim();
        var supportedDetAlgorithms = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "DB",
            "DB++",
            "EAST",
            "SAST",
            "PSE",
            "FCE",
            "CT"
        };
        if (!supportedDetAlgorithms.Contains(detAlgorithm))
        {
            return CommandResult.Fail($"infer det --det_algorithm unsupported: {detAlgorithm}. expected one of DB|DB++|EAST|SAST|PSE|FCE|CT");
        }

        var detBoxType = (ResolveString(context, "--det_box_type", "Global.det_box_type") ?? "quad").Trim().ToLowerInvariant();
        if (detBoxType is not ("quad" or "poly"))
        {
            return CommandResult.Fail($"infer det --det_box_type must be quad|poly, got={detBoxType}");
        }

        var detLimitType = (ResolveString(context, "--det_limit_type", "Global.det_limit_type") ?? "max").Trim().ToLowerInvariant();
        if (detLimitType is not ("max" or "min"))
        {
            return CommandResult.Fail($"infer det --det_limit_type must be max|min, got={detLimitType}");
        }

        var detThresh = ParseFloat(ResolveString(context, "--det_db_thresh", "Global.det_db_thresh") ?? "0.3", 0.3f);
        var detBoxThresh = ParseFloat(ResolveString(context, "--det_db_box_thresh", "Global.det_db_box_thresh") ?? "0.6", 0.6f);
        var detUnclipRatio = ParseFloat(ResolveString(context, "--det_db_unclip_ratio", "Global.det_db_unclip_ratio") ?? "1.5", 1.5f);
        var detLimitSideLen = ParseInt(ResolveString(context, "--det_limit_side_len", "Global.det_limit_side_len") ?? "640", 640, 32);
        var useDilation = ParseBool(ResolveString(context, "--use_dilation", "Global.use_dilation") ?? "false");
        var saveResPath = ResolveString(context, "--save_res_path", "Global.save_res_path");
        var detEastScoreThresh = ParseFloat(ResolveString(context, "--det_east_score_thresh", "Global.det_east_score_thresh") ?? "0.8", 0.8f);
        var detEastCoverThresh = ParseFloat(ResolveString(context, "--det_east_cover_thresh", "Global.det_east_cover_thresh") ?? "0.1", 0.1f);
        var detEastNmsThresh = ParseFloat(ResolveString(context, "--det_east_nms_thresh", "Global.det_east_nms_thresh") ?? "0.2", 0.2f);
        var detSastScoreThresh = ParseFloat(ResolveString(context, "--det_sast_score_thresh", "Global.det_sast_score_thresh") ?? "0.5", 0.5f);
        var detSastNmsThresh = ParseFloat(ResolveString(context, "--det_sast_nms_thresh", "Global.det_sast_nms_thresh") ?? "0.2", 0.2f);
        var detPseThresh = ParseFloat(ResolveString(context, "--det_pse_thresh", "Global.det_pse_thresh") ?? "0.0", 0.0f);
        var detPseBoxThresh = ParseFloat(ResolveString(context, "--det_pse_box_thresh", "Global.det_pse_box_thresh") ?? "0.85", 0.85f);
        var detPseMinArea = ParseFloat(ResolveString(context, "--det_pse_min_area", "Global.det_pse_min_area") ?? "16", 16f);
        var detPseScale = ParseFloat(ResolveString(context, "--det_pse_scale", "Global.det_pse_scale") ?? "1", 1f);
        var fceScales = ParseCsvInt(ResolveString(context, "--scales", "Global.scales") ?? "8,16,32", [8, 16, 32]);
        var fceAlpha = ParseFloat(ResolveString(context, "--alpha", "Global.alpha") ?? "1.0", 1.0f);
        var fceBeta = ParseFloat(ResolveString(context, "--beta", "Global.beta") ?? "1.0", 1.0f);
        var fceFourierDegree = ParseInt(ResolveString(context, "--fourier_degree", "Global.fourier_degree") ?? "5", 5, 1);
        var detGtLabelPath = ResolveString(context, "--det_gt_label", "Global.det_gt_label");
        var detEvalIouThresh = ParseFloat(ResolveString(context, "--det_eval_iou_thresh", "Global.det_eval_iou_thresh") ?? "0.5", 0.5f);
        var detMetricsPath = ResolveString(context, "--det_metrics_path", "Global.det_metrics_path");

        if (!IsUnitRange(detThresh))
        {
            return CommandResult.Fail($"infer det --det_db_thresh must be in [0,1], got={detThresh}");
        }

        if (!IsUnitRange(detBoxThresh))
        {
            return CommandResult.Fail($"infer det --det_db_box_thresh must be in [0,1], got={detBoxThresh}");
        }

        if (detUnclipRatio <= 0f)
        {
            return CommandResult.Fail($"infer det --det_db_unclip_ratio must be > 0, got={detUnclipRatio}");
        }

        if (!IsUnitRange(detEastScoreThresh) || !IsUnitRange(detEastCoverThresh) || !IsUnitRange(detEastNmsThresh))
        {
            return CommandResult.Fail("infer det EAST thresholds must be in [0,1]");
        }

        if (!IsUnitRange(detSastScoreThresh) || !IsUnitRange(detSastNmsThresh))
        {
            return CommandResult.Fail("infer det SAST thresholds must be in [0,1]");
        }

        if (!IsUnitRange(detPseThresh) || !IsUnitRange(detPseBoxThresh))
        {
            return CommandResult.Fail("infer det PSE thresholds must be in [0,1]");
        }

        if (detPseMinArea <= 0f || detPseScale <= 0f)
        {
            return CommandResult.Fail("infer det PSE params --det_pse_min_area and --det_pse_scale must be > 0");
        }

        if (fceAlpha <= 0f || fceBeta <= 0f)
        {
            return CommandResult.Fail("infer det FCE params --alpha and --beta must be > 0");
        }

        if (!IsUnitRange(detEvalIouThresh))
        {
            return CommandResult.Fail($"infer det --det_eval_iou_thresh must be in [0,1], got={detEvalIouThresh}");
        }

        if (!string.IsNullOrWhiteSpace(detGtLabelPath) && !File.Exists(detGtLabelPath))
        {
            return CommandResult.Fail($"infer det --det_gt_label not found: {detGtLabelPath}");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer det currently supports --use_onnx=true only.");
        }

        if (!File.Exists(detModel))
        {
            return CommandResult.Fail($"det model not found: {detModel}");
        }

        var output = ResolveOutputDir(context, "det");
        var options = new DetOnnxOptions(
            imageDir,
            detModel,
            output,
            detAlgorithm,
            detThresh,
            detBoxThresh,
            detUnclipRatio,
            useDilation,
            detBoxType,
            detLimitSideLen,
            detLimitType,
            saveResPath,
            detEastScoreThresh,
            detEastCoverThresh,
            detEastNmsThresh,
            detSastScoreThresh,
            detSastNmsThresh,
            detPseThresh,
            detPseBoxThresh,
            detPseMinArea,
            detPseScale,
            fceScales,
            fceAlpha,
            fceBeta,
            fceFourierDegree,
            detGtLabelPath,
            detEvalIouThresh,
            detMetricsPath);
        new DetOnnxRunner().Run(options);
        var resultFile = string.IsNullOrWhiteSpace(saveResPath) ? Path.Combine(output, "det_results.txt") : saveResPath;
        var metricsFile = string.IsNullOrWhiteSpace(detMetricsPath) ? Path.Combine(output, "det_metrics.json") : detMetricsPath;
        return CommandResult.Ok($"infer det completed. output={output}, result={resultFile}, metrics={metricsFile}");
    }

    private static CommandResult RunRec(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var recModel = GetOrNull(context, "--rec_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(recModel))
        {
            return CommandResult.Fail("infer rec requires --image_dir and --rec_model_dir");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer rec currently supports --use_onnx=true only.");
        }

        if (!File.Exists(recModel))
        {
            return CommandResult.Fail($"rec model not found: {recModel}");
        }

        var output = ResolveOutputDir(context, "rec");
        var options = new RecOnnxOptions(
            imageDir,
            recModel,
            output,
            GetOrNull(context, "--rec_char_dict_path"),
            ParseBool(GetOrDefault(context, "--use_space_char", "true")),
            ParseFloat(GetOrDefault(context, "--drop_score", "0.5")));
        new RecOnnxRunner().Run(options);
        return CommandResult.Ok($"infer rec completed. output={output}");
    }

    private static CommandResult RunCls(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var clsModel = GetOrNull(context, "--cls_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(clsModel))
        {
            return CommandResult.Fail("infer cls requires --image_dir and --cls_model_dir");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer cls currently supports --use_onnx=true only.");
        }

        if (!File.Exists(clsModel))
        {
            return CommandResult.Fail($"cls model not found: {clsModel}");
        }

        var output = ResolveOutputDir(context, "cls");
        var options = new ClsOnnxOptions(
            imageDir,
            clsModel,
            output,
            ParseCsv(GetOrDefault(context, "--label_list", "0,180")),
            ParseFloat(GetOrDefault(context, "--cls_thresh", "0.9")));
        new ClsOnnxRunner().Run(options);
        return CommandResult.Ok($"infer cls completed. output={output}");
    }

    private static CommandResult RunE2e(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var detModel = GetOrNull(context, "--det_model_dir");
        var recModel = GetOrNull(context, "--rec_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(detModel) || string.IsNullOrWhiteSpace(recModel))
        {
            return CommandResult.Fail("infer e2e requires --image_dir, --det_model_dir and --rec_model_dir");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer e2e currently supports --use_onnx=true only.");
        }

        if (!File.Exists(detModel))
        {
            return CommandResult.Fail($"det model not found: {detModel}");
        }

        if (!File.Exists(recModel))
        {
            return CommandResult.Fail($"rec model not found: {recModel}");
        }

        var output = ResolveOutputDir(context, "e2e");
        var options = new SystemOnnxOptions(
            imageDir,
            recModel,
            detModel,
            GetOrNull(context, "--cls_model_dir"),
            output,
            GetOrNull(context, "--rec_char_dict_path"),
            ParseBool(GetOrDefault(context, "--use_space_char", "true")),
            ParseCsv(GetOrDefault(context, "--label_list", "0,180")),
            ParseFloat(GetOrDefault(context, "--drop_score", "0.5")),
            ParseFloat(GetOrDefault(context, "--cls_thresh", "0.9")),
            ParseFloat(GetOrDefault(context, "--det_db_thresh", "0.3")));
        new SystemOnnxRunner().Run(options);
        return CommandResult.Ok($"infer e2e completed. output={output}");
    }

    private static CommandResult RunSr(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = ResolveString(context, "--image_dir", "Global.image_dir", "Global.infer_img");
        var srModel = ResolveString(context, "--sr_model_dir", "Global.sr_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(srModel))
        {
            return CommandResult.Fail("infer sr requires --image_dir and --sr_model_dir");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer sr currently supports --use_onnx=true only.");
        }

        if (!File.Exists(srModel))
        {
            return CommandResult.Fail($"sr model not found: {srModel}");
        }

        var output = ResolveOutputDir(context, "sr");
        var options = new SrOnnxOptions(imageDir, srModel, output);
        new SrOnnxRunner().Run(options);
        return CommandResult.Ok($"infer sr completed. output={output}");
    }

    private static CommandResult RunTable(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = ResolveString(context, "--image_dir", "Global.image_dir", "Global.infer_img");
        var tableModel = ResolveString(context, "--table_model_dir", "Global.table_model_dir");
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(tableModel))
        {
            return CommandResult.Fail("infer table requires --image_dir and --table_model_dir");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer table currently supports --use_onnx=true only.");
        }

        if (!File.Exists(tableModel))
        {
            return CommandResult.Fail($"table model not found: {tableModel}");
        }

        var output = ResolveOutputDir(context, "table");
        var options = new TableOnnxOptions(
            imageDir,
            tableModel,
            output,
            ResolveString(context, "--det_model_dir", "Global.det_model_dir"),
            ResolveString(context, "--rec_model_dir", "Global.rec_model_dir"),
            ResolveString(context, "--rec_char_dict_path", "Global.rec_char_dict_path", "Global.character_dict_path"),
            ParseBool(GetOrDefault(context, "--use_space_char", "true")),
            ParseFloat(GetOrDefault(context, "--drop_score", "0.5")),
            ParseFloat(GetOrDefault(context, "--det_db_thresh", "0.3")));
        new TableOnnxRunner().Run(options);
        return CommandResult.Ok($"infer table completed. output={output}");
    }

    private static CommandResult RunKie(PaddleOcr.Core.Cli.ExecutionContext context, string commandName, string taskName, string modelArg)
    {
        var imageDir = ResolveString(context, "--image_dir", "Global.image_dir", "Global.infer_img");
        var modelPath = commandName switch
        {
            "kie" => ResolveString(context, modelArg, "Global.kie_model_dir"),
            "kie-ser" => ResolveString(context, modelArg, "Global.ser_model_dir"),
            "kie-re" => ResolveString(context, modelArg, "Global.re_model_dir"),
            _ => ResolveString(context, modelArg)
        };
        if (string.IsNullOrWhiteSpace(imageDir) || string.IsNullOrWhiteSpace(modelPath))
        {
            return CommandResult.Fail($"infer {commandName} requires --image_dir and {modelArg}");
        }

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail($"infer {commandName} currently supports --use_onnx=true only.");
        }

        if (!File.Exists(modelPath))
        {
            return CommandResult.Fail($"{commandName} model not found: {modelPath}");
        }

        var output = ResolveOutputDir(context, commandName);
        var options = new KieOnnxOptions(
            taskName,
            imageDir,
            modelPath,
            output,
            ResolveString(context, "--det_model_dir", "Global.det_model_dir"),
            ResolveString(context, "--rec_model_dir", "Global.rec_model_dir"),
            ResolveString(context, "--rec_char_dict_path", "Global.rec_char_dict_path", "Global.character_dict_path"),
            ParseBool(GetOrDefault(context, "--use_space_char", "true")),
            ParseFloat(GetOrDefault(context, "--drop_score", "0.5")),
            ParseFloat(GetOrDefault(context, "--det_db_thresh", "0.3")));
        new KieOnnxRunner().Run(options);
        return CommandResult.Ok($"infer {commandName} completed. output={output}");
    }

    private static string? GetOrNull(PaddleOcr.Core.Cli.ExecutionContext context, string key)
    {
        return context.Options.TryGetValue(key, out var value) ? value : null;
    }

    private static string GetOrDefault(PaddleOcr.Core.Cli.ExecutionContext context, string key, string defaultValue)
    {
        return context.Options.TryGetValue(key, out var value) ? value : defaultValue;
    }

    private static string? ResolveString(PaddleOcr.Core.Cli.ExecutionContext context, string optionKey, params string[] configPaths)
    {
        if (context.Options.TryGetValue(optionKey, out var opt) && !string.IsNullOrWhiteSpace(opt))
        {
            return opt;
        }

        foreach (var path in configPaths)
        {
            var fromCfg = GetConfigString(context.Config, path);
            if (!string.IsNullOrWhiteSpace(fromCfg))
            {
                return fromCfg;
            }
        }

        return null;
    }

    private static string? GetConfigString(IReadOnlyDictionary<string, object?> root, string path)
    {
        object? cur = root;
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

        return cur?.ToString();
    }

    private static string ResolveOutputDir(PaddleOcr.Core.Cli.ExecutionContext context, string command)
    {
        if (context.Options.TryGetValue("--draw_img_save_dir", out var value) && !string.IsNullOrWhiteSpace(value))
        {
            return value;
        }

        return Path.Combine("./inference_results", command.Replace('-', '_'));
    }

    private static bool ParseBool(string text)
    {
        return bool.TryParse(text, out var value) && value;
    }

    private static float ParseFloat(string text, float defaultValue = 0f)
    {
        return float.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var value)
            ? value
            : defaultValue;
    }

    private static int ParseInt(string text, int defaultValue, int minValue)
    {
        if (!int.TryParse(text, out var value))
        {
            value = defaultValue;
        }

        return Math.Max(minValue, value);
    }

    private static bool IsUnitRange(float value)
    {
        return value >= 0f && value <= 1f;
    }

    private static IReadOnlyList<int> ParseCsvInt(string text, IReadOnlyList<int> fallback)
    {
        var values = text
            .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(x => int.TryParse(x, out var n) ? n : 0)
            .Where(x => x > 0)
            .Distinct()
            .OrderBy(x => x)
            .ToList();
        return values.Count == 0 ? fallback : values;
    }

    private static IReadOnlyList<string> ParseCsv(string text)
    {
        return text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    }
}
