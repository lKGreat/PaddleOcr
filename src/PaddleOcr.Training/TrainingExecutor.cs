using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;

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

        context.Logger.LogInformation("Running {Command} with config: {ConfigPath}", subCommand, context.ConfigPath);
        context.Logger.LogInformation("Override count: {Count}", context.OverrideOptions.Count);
        return Task.FromResult(CommandResult.Ok($"{subCommand} pipeline scaffold executed."));
    }
}
