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
        "convert:json2pdmodel",
        "convert:check-json-model"
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

        try
        {
            var exporter = new NativeExporter(context.Logger);
            if (subCommand.Equals("convert:json2pdmodel", StringComparison.OrdinalIgnoreCase))
            {
                if (!context.Options.TryGetValue("--json_model_dir", out var jsonDir) ||
                    !context.Options.TryGetValue("--output_dir", out var outputDir))
                {
                    return Task.FromResult(CommandResult.Fail("convert json2pdmodel requires --json_model_dir and --output_dir"));
                }

                exporter.ConvertJsonToPdmodel(jsonDir, outputDir);
                return Task.FromResult(CommandResult.Ok($"convert:json2pdmodel completed. output={outputDir}"));
            }

            if (subCommand.Equals("convert:check-json-model", StringComparison.OrdinalIgnoreCase))
            {
                if (!context.Options.TryGetValue("--json_model_dir", out var jsonDir))
                {
                    return Task.FromResult(CommandResult.Fail("convert check-json-model requires --json_model_dir"));
                }

                var ok = exporter.ValidateJsonModelDir(jsonDir, out var message);
                return Task.FromResult(ok
                    ? CommandResult.Ok($"convert:check-json-model passed: {message}")
                    : CommandResult.Fail($"convert:check-json-model failed: {message}"));
            }

            var cfg = new ExportConfigView(context.Config, context.ConfigPath!);
            if (subCommand.Equals("export-onnx", StringComparison.OrdinalIgnoreCase))
            {
                var outFile = exporter.ExportOnnx(cfg);
                return Task.FromResult(CommandResult.Ok($"export-onnx completed. output={outFile}"));
            }

            var native = exporter.ExportNative(cfg);
            return Task.FromResult(CommandResult.Ok($"{subCommand} completed. output={native}"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(CommandResult.Fail($"{subCommand} failed: {ex.Message}"));
        }
    }
}
