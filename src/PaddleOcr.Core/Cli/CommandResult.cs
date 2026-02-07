namespace PaddleOcr.Core.Cli;

public sealed record CommandResult(bool Success, string Message)
{
    public static CommandResult Ok(string message) => new(true, message);
    public static CommandResult Fail(string message) => new(false, message);
}

