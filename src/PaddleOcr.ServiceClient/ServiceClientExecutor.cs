using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.ServiceClient;

public sealed class ServiceClientExecutor : ICommandExecutor
{
    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!subCommand.Equals("test", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(CommandResult.Fail($"Unsupported service subcommand: {subCommand}"));
        }

        if (!context.Options.ContainsKey("--server_url"))
        {
            return Task.FromResult(CommandResult.Fail("service test requires --server_url"));
        }

        if (!context.Options.ContainsKey("--image_dir"))
        {
            return Task.FromResult(CommandResult.Fail("service test requires --image_dir"));
        }

        context.Logger.LogInformation("Running service test for endpoint: {ServerUrl}", context.Options["--server_url"]);
        return Task.FromResult(CommandResult.Ok("service test scaffold executed."));
    }
}
