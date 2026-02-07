namespace PaddleOcr.Core.Cli;

public interface ICommandExecutor
{
    Task<CommandResult> ExecuteAsync(string subCommand, ExecutionContext context, CancellationToken cancellationToken = default);
}

