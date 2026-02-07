using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.Export;

public sealed class ExportExecutor : ICommandExecutor
{
    private static readonly HashSet<string> Supported = new(StringComparer.OrdinalIgnoreCase)
    {
        "export",
        "export-onnx",
        "export-center",
        "convert:json2pdmodel"
    };

    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!Supported.Contains(subCommand))
        {
            return Task.FromResult(CommandResult.Fail($"Unsupported export command: {subCommand}"));
        }

        if (!subCommand.Equals("convert:json2pdmodel", StringComparison.OrdinalIgnoreCase) &&
            string.IsNullOrWhiteSpace(context.ConfigPath))
        {
            return Task.FromResult(CommandResult.Fail($"{subCommand} requires -c/--config"));
        }

        context.Logger.LogInformation("Running {Command}", subCommand);
        return Task.FromResult(CommandResult.Ok($"{subCommand} pipeline scaffold executed."));
    }
}
