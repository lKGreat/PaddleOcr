using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;

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

        context.Logger.LogInformation("Running infer {Mode}", subCommand);
        return Task.FromResult(CommandResult.Ok($"infer {subCommand} scaffold executed."));
    }
}
