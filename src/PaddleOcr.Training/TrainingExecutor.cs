using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;
using PaddleOcr.Core.Errors;

namespace PaddleOcr.Training;

public sealed class TrainingExecutor : ICommandExecutor
{
    private static readonly HashSet<string> Supported = new(StringComparer.OrdinalIgnoreCase)
    {
        "train",
        "eval"
    };

    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!Supported.Contains(subCommand))
        {
            return Task.FromResult(CommandResult.Fail($"Unsupported training command: {subCommand}"));
        }

        if (string.IsNullOrWhiteSpace(context.ConfigPath))
        {
            return Task.FromResult(CommandResult.Fail($"{subCommand} requires -c/--config"));
        }

        var cfg = new TrainingConfigView(context.Config, context.ConfigPath);
        context.Logger.LogInformation("Running {Command} with config: {ConfigPath}", subCommand, context.ConfigPath);
        context.Logger.LogInformation("Override count: {Count}", context.OverrideOptions.Count);
        try
        {
            if (string.Equals(cfg.ModelType, "cls", StringComparison.OrdinalIgnoreCase))
            {
                var trainer = new SimpleClsTrainer(context.Logger);
                if (subCommand.Equals("train", StringComparison.OrdinalIgnoreCase))
                {
                    var summary = trainer.Train(cfg);
                    return Task.FromResult(CommandResult.Ok($"train completed: best_acc={summary.BestAccuracy:F4}, save_dir={summary.SaveDir}"));
                }

                var eval = trainer.Eval(cfg);
                return Task.FromResult(CommandResult.Ok($"eval completed: acc={eval.Accuracy:F4}, samples={eval.Samples}"));
            }

            if (string.Equals(cfg.ModelType, "det", StringComparison.OrdinalIgnoreCase))
            {
                var trainer = new SimpleDetTrainer(context.Logger);
                if (subCommand.Equals("train", StringComparison.OrdinalIgnoreCase))
                {
                    var summary = trainer.Train(cfg);
                    return Task.FromResult(CommandResult.Ok($"train completed: best_iou={summary.BestAccuracy:F4}, save_dir={summary.SaveDir}"));
                }

                var eval = trainer.Eval(cfg);
                return Task.FromResult(CommandResult.Ok($"eval completed: iou={eval.Accuracy:F4}, samples={eval.Samples}"));
            }

            if (string.Equals(cfg.ModelType, "rec", StringComparison.OrdinalIgnoreCase))
            {
                var trainer = new SimpleRecTrainer(context.Logger);
                if (subCommand.Equals("train", StringComparison.OrdinalIgnoreCase))
                {
                    var summary = trainer.Train(cfg);
                    return Task.FromResult(CommandResult.Ok($"train completed: best_acc={summary.BestAccuracy:F4}, save_dir={summary.SaveDir}"));
                }

                var eval = trainer.Eval(cfg);
                return Task.FromResult(CommandResult.Ok($"eval completed: acc={eval.Accuracy:F4}, samples={eval.Samples}"));
            }

            return Task.FromResult(CommandResult.Fail($"model_type '{cfg.ModelType}' not supported yet. Current implementation supports cls/det/rec."));
        }
        catch (Exception ex)
        {
            throw new PocrException($"training failed: {ex.Message}");
        }
    }
}
