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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
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

        if (!ParseBool(GetOrDefault(context, "--use_onnx", "false")))
        {
            return CommandResult.Fail("infer det currently supports --use_onnx=true only.");
        }

        if (!File.Exists(detModel))
        {
            return CommandResult.Fail($"det model not found: {detModel}");
        }

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
        var options = new DetOnnxOptions(imageDir, detModel, output, ParseFloat(GetOrDefault(context, "--det_db_thresh", "0.3")));
        new DetOnnxRunner().Run(options);
        return CommandResult.Ok($"infer det completed. output={output}");
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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
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
        var imageDir = GetOrNull(context, "--image_dir");
        var srModel = GetOrNull(context, "--sr_model_dir");
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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
        var options = new SrOnnxOptions(imageDir, srModel, output);
        new SrOnnxRunner().Run(options);
        return CommandResult.Ok($"infer sr completed. output={output}");
    }

    private static CommandResult RunTable(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var tableModel = GetOrNull(context, "--table_model_dir");
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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
        var options = new TableOnnxOptions(
            imageDir,
            tableModel,
            output,
            GetOrNull(context, "--det_model_dir"),
            GetOrNull(context, "--rec_model_dir"),
            GetOrNull(context, "--rec_char_dict_path"),
            ParseBool(GetOrDefault(context, "--use_space_char", "true")),
            ParseFloat(GetOrDefault(context, "--drop_score", "0.5")),
            ParseFloat(GetOrDefault(context, "--det_db_thresh", "0.3")));
        new TableOnnxRunner().Run(options);
        return CommandResult.Ok($"infer table completed. output={output}");
    }

    private static CommandResult RunKie(PaddleOcr.Core.Cli.ExecutionContext context, string commandName, string taskName, string modelArg)
    {
        var imageDir = GetOrNull(context, "--image_dir");
        var modelPath = GetOrNull(context, modelArg);
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

        var output = GetOrDefault(context, "--draw_img_save_dir", "./inference_results");
        var options = new KieOnnxOptions(
            taskName,
            imageDir,
            modelPath,
            output,
            GetOrNull(context, "--det_model_dir"),
            GetOrNull(context, "--rec_model_dir"),
            GetOrNull(context, "--rec_char_dict_path"),
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

    private static bool ParseBool(string text)
    {
        return bool.TryParse(text, out var value) && value;
    }

    private static float ParseFloat(string text)
    {
        return float.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var value)
            ? value
            : 0f;
    }

    private static IReadOnlyList<string> ParseCsv(string text)
    {
        return text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    }
}
