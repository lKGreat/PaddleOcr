using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.Data;

public sealed class E2eToolsExecutor : ICommandExecutor
{
    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (subCommand.Equals("convert-label", StringComparison.OrdinalIgnoreCase))
        {
            var ok = context.Options.ContainsKey("--label_path") && context.Options.ContainsKey("--save_folder");
            if (!ok)
            {
                return Task.FromResult(CommandResult.Fail("e2e convert-label requires --label_path and --save_folder"));
            }

            context.Logger.LogInformation("Running e2e convert-label");
            return Task.FromResult(CommandResult.Ok("e2e convert-label scaffold executed."));
        }

        if (subCommand.Equals("eval", StringComparison.OrdinalIgnoreCase))
        {
            var hasPositional = context.RawArgs.Length >= 4;
            if (!hasPositional)
            {
                return Task.FromResult(CommandResult.Fail("e2e eval requires <gt_dir> <pred_dir>"));
            }

            context.Logger.LogInformation("Running e2e eval");
            return Task.FromResult(CommandResult.Ok("e2e eval scaffold executed."));
        }

        return Task.FromResult(CommandResult.Fail($"Unsupported e2e subcommand: {subCommand}"));
    }
}
