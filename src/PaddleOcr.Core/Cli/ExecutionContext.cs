using Microsoft.Extensions.Logging;

namespace PaddleOcr.Core.Cli;

public sealed record ExecutionContext(
    ILogger Logger,
    string[] RawArgs,
    string? ConfigPath,
    IReadOnlyDictionary<string, object?> Config,
    IReadOnlyDictionary<string, string> Options,
    IReadOnlyList<string> OverrideOptions);

