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
            "config" => await RunConfigAsync(parsed, context),
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
        if (string.Equals(parsed.Sub, "json2pdmodel", StringComparison.OrdinalIgnoreCase))
        {
            return _export.ExecuteAsync("convert:json2pdmodel", context, cancellationToken);
        }

        if (string.Equals(parsed.Sub, "check-json-model", StringComparison.OrdinalIgnoreCase))
        {
            return _export.ExecuteAsync("convert:check-json-model", context, cancellationToken);
        }

        return Task.FromResult(CommandResult.Fail("convert supports: json2pdmodel | check-json-model"));
    }

    private Task<CommandResult> RunConfigAsync(ParsedCommand parsed, PaddleOcr.Core.Cli.ExecutionContext context)
    {
        if (string.Equals(parsed.Sub, "check", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrWhiteSpace(context.ConfigPath))
            {
                return Task.FromResult(CommandResult.Fail("config check requires -c/--config"));
            }

            var ok = ConfigValidator.ValidateBasic(context.Config, out var message);
            return Task.FromResult(ok
                ? CommandResult.Ok($"config check passed: {message}")
                : CommandResult.Fail($"config check failed: {message}"));
        }

        if (string.Equals(parsed.Sub, "diff", StringComparison.OrdinalIgnoreCase))
        {
            if (!context.Options.TryGetValue("--base", out var basePath) ||
                !context.Options.TryGetValue("--target", out var targetPath))
            {
                return Task.FromResult(CommandResult.Fail("config diff requires --base and --target"));
            }

            var left = _configLoader.Load(basePath);
            var right = _configLoader.Load(targetPath);
            var diffs = ConfigValidator.Diff(left, right);
            if (diffs.Count == 0)
            {
                return Task.FromResult(CommandResult.Ok("config diff: no differences"));
            }

            var preview = string.Join('\n', diffs.Take(20));
            return Task.FromResult(CommandResult.Ok($"config diff: {diffs.Count} differences\n{preview}"));
        }

        return Task.FromResult(CommandResult.Fail("config supports: check | diff"));
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
