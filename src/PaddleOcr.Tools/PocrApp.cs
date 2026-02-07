using Microsoft.Extensions.Logging;
using PaddleOcr.Config;
using PaddleOcr.Core.Cli;
using PaddleOcr.Core.Errors;

namespace PaddleOcr.Tools;

public sealed class PocrApp
{
    private readonly ILogger _logger;
    private readonly ConfigLoader _configLoader;
    private readonly ICommandExecutor _training;
    private readonly ICommandExecutor _inference;
    private readonly ICommandExecutor _export;
    private readonly ICommandExecutor _service;
    private readonly ICommandExecutor _e2e;

    public PocrApp(
        ILogger logger,
        ConfigLoader configLoader,
        ICommandExecutor training,
        ICommandExecutor inference,
        ICommandExecutor export,
        ICommandExecutor service,
        ICommandExecutor e2e)
    {
        _logger = logger;
        _configLoader = configLoader;
        _training = training;
        _inference = inference;
        _export = export;
        _service = service;
        _e2e = e2e;
    }

    public async Task<int> RunAsync(string[] args, CancellationToken cancellationToken = default)
    {
        var parsed = CommandLine.Parse(args);
        var context = BuildContext(parsed);

        CommandResult result = parsed.Root.ToLowerInvariant() switch
        {
            "train" => await _training.ExecuteAsync("train", context, cancellationToken),
            "eval" => await _training.ExecuteAsync("eval", context, cancellationToken),
            "export" => await _export.ExecuteAsync("export", context, cancellationToken),
            "export-onnx" => await _export.ExecuteAsync("export-onnx", context, cancellationToken),
            "export-center" => await _export.ExecuteAsync("export-center", context, cancellationToken),
            "infer" => await _inference.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            "convert" => await RunConvertAsync(parsed, context, cancellationToken),
            "service" => await _service.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            "e2e" => await _e2e.ExecuteAsync(parsed.Sub ?? string.Empty, context, cancellationToken),
            _ => CommandResult.Fail($"Unknown command: {parsed.Root}\n{CommandLine.GetHelp()}")
        };

        if (result.Success)
        {
            _logger.LogInformation(result.Message);
            return 0;
        }

        _logger.LogError(result.Message);
        return 2;
    }

    private Task<CommandResult> RunConvertAsync(ParsedCommand parsed, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken)
    {
        if (!string.Equals(parsed.Sub, "json2pdmodel", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(CommandResult.Fail("convert supports only json2pdmodel"));
        }
        
        return _export.ExecuteAsync("convert:json2pdmodel", context, cancellationToken);
    }

    private PaddleOcr.Core.Cli.ExecutionContext BuildContext(ParsedCommand parsed)
    {
        var cfg = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (!string.IsNullOrWhiteSpace(parsed.ConfigPath))
        {
            foreach (var pair in _configLoader.Load(parsed.ConfigPath))
            {
                cfg[pair.Key] = pair.Value;
            }
        }

        if (parsed.Overrides.Count > 0)
        {
            var parsedOverrides = OverrideParser.Parse(parsed.Overrides);
            ConfigMerger.MergeInPlace(cfg, parsedOverrides);
        }

        return new PaddleOcr.Core.Cli.ExecutionContext(
            _logger,
            parsed.RawArgs.ToArray(),
            parsed.ConfigPath,
            cfg,
            parsed.Options,
            parsed.Overrides);
    }
}
