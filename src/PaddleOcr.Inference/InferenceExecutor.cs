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
            output);

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

    private static string? GetOrNull(PaddleOcr.Core.Cli.ExecutionContext context, string key)
    {
        return context.Options.TryGetValue(key, out var value) ? value : null;
    }

    private static string GetOrDefault(PaddleOcr.Core.Cli.ExecutionContext context, string key, string defaultValue)
    {
        return context.Options.TryGetValue(key, out var value) ? value : defaultValue;
    }
}
