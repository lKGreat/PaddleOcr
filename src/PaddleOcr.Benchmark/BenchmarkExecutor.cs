using System.Diagnostics;
using System.Text.Json;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.Benchmark;

public sealed class BenchmarkExecutor : ICommandExecutor
{
    private static readonly HashSet<string> InternalFlags = new(StringComparer.OrdinalIgnoreCase)
    {
        "--scenario",
        "--warmup",
        "--iterations",
        "--continue_on_error",
        "--report_json"
    };

    private readonly ICommandExecutor _training;
    private readonly ICommandExecutor _inference;
    private readonly ICommandExecutor _export;
    private readonly ICommandExecutor _service;
    private readonly ICommandExecutor _e2e;

    public BenchmarkExecutor(
        ICommandExecutor training,
        ICommandExecutor inference,
        ICommandExecutor export,
        ICommandExecutor service,
        ICommandExecutor e2e)
    {
        _training = training;
        _inference = inference;
        _export = export;
        _service = service;
        _e2e = e2e;
    }

    public async Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!subCommand.Equals("run", StringComparison.OrdinalIgnoreCase))
        {
            return CommandResult.Fail("benchmark supports: run");
        }

        if (!context.Options.TryGetValue("--scenario", out var scenario) || string.IsNullOrWhiteSpace(scenario))
        {
            return CommandResult.Fail("benchmark run requires --scenario");
        }

        var warmup = ParseInt(context, "--warmup", 2, 0, 50);
        var iterations = ParseInt(context, "--iterations", 10, 1, 500);
        var continueOnError = ParseBool(context, "--continue_on_error");
        var reportJson = context.Options.TryGetValue("--report_json", out var reportPath) && !string.IsNullOrWhiteSpace(reportPath)
            ? reportPath
            : Path.Combine("benchmark_results", $"{scenario.Replace(':', '_')}_{DateTime.UtcNow:yyyyMMddHHmmss}.json");

        var routed = ResolveScenario(scenario);
        if (routed is null)
        {
            return CommandResult.Fail("unsupported scenario, expected one of infer:system|service:test|e2e:eval|export:export-onnx|train:train");
        }
        var route = routed.Value;

        var scenarioContext = BuildScenarioContext(context);
        var warmupFailures = 0;
        for (var i = 0; i < warmup; i++)
        {
            var result = await route.Executor.ExecuteAsync(route.SubCommand, scenarioContext, cancellationToken);
            if (!result.Success)
            {
                warmupFailures++;
                if (!continueOnError)
                {
                    return CommandResult.Fail($"benchmark warmup failed at {i + 1}/{warmup}: {result.Message}");
                }
            }
        }

        var samples = new List<BenchmarkIteration>(iterations);
        var success = 0;
        var failed = 0;
        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            var result = await route.Executor.ExecuteAsync(route.SubCommand, scenarioContext, cancellationToken);
            sw.Stop();

            samples.Add(new BenchmarkIteration(i + 1, sw.Elapsed.TotalMilliseconds, result.Success, result.Message));
            if (result.Success)
            {
                success++;
            }
            else
            {
                failed++;
                if (!continueOnError)
                {
                    break;
                }
            }
        }

        if (success == 0)
        {
            return CommandResult.Fail("benchmark completed with 0 successful iterations");
        }

        var successLatencies = samples.Where(x => x.Success).Select(x => x.ElapsedMs).OrderBy(x => x).ToArray();
        var totalSec = samples.Sum(x => x.ElapsedMs) / 1000.0;
        var summary = new BenchmarkSummary(
            scenario,
            warmup,
            iterations,
            samples.Count,
            warmupFailures,
            success,
            failed,
            successLatencies.Average(),
            Percentile(successLatencies, 0.5),
            Percentile(successLatencies, 0.95),
            Percentile(successLatencies, 0.99),
            totalSec <= 0 ? 0 : success / totalSec,
            DateTime.UtcNow,
            samples);

        var reportDir = Path.GetDirectoryName(reportJson);
        if (!string.IsNullOrWhiteSpace(reportDir))
        {
            Directory.CreateDirectory(reportDir);
        }

        File.WriteAllText(reportJson, JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
        return CommandResult.Ok(
            $"benchmark run completed: scenario={scenario}, success={success}, failed={failed}, p50={summary.P50Ms:F2}ms, p95={summary.P95Ms:F2}ms, throughput={summary.ThroughputPerSec:F2}/s, report={reportJson}");
    }

    private (ICommandExecutor Executor, string SubCommand)? ResolveScenario(string scenario)
    {
        return scenario.ToLowerInvariant() switch
        {
            "infer:system" => (_inference, "system"),
            "service:test" => (_service, "test"),
            "e2e:eval" => (_e2e, "eval"),
            "export:export-onnx" => (_export, "export-onnx"),
            "train:train" => (_training, "train"),
            _ => null
        };
    }

    private static PaddleOcr.Core.Cli.ExecutionContext BuildScenarioContext(PaddleOcr.Core.Cli.ExecutionContext source)
    {
        var options = source.Options
            .Where(x => !InternalFlags.Contains(x.Key))
            .ToDictionary(x => x.Key, x => x.Value, StringComparer.OrdinalIgnoreCase);
        return new PaddleOcr.Core.Cli.ExecutionContext(
            source.Logger,
            source.RawArgs,
            source.ConfigPath,
            source.Config,
            options,
            source.OverrideOptions);
    }

    private static int ParseInt(PaddleOcr.Core.Cli.ExecutionContext context, string key, int fallback, int min, int max)
    {
        if (!context.Options.TryGetValue(key, out var text) || !int.TryParse(text, out var value))
        {
            return fallback;
        }

        return Math.Clamp(value, min, max);
    }

    private static bool ParseBool(PaddleOcr.Core.Cli.ExecutionContext context, string key)
    {
        return context.Options.TryGetValue(key, out var value)
               && bool.TryParse(value, out var flag)
               && flag;
    }

    private static double Percentile(double[] sortedValues, double percentile)
    {
        if (sortedValues.Length == 0)
        {
            return 0;
        }

        if (sortedValues.Length == 1)
        {
            return sortedValues[0];
        }

        var p = Math.Clamp(percentile, 0, 1);
        var rank = p * (sortedValues.Length - 1);
        var low = (int)Math.Floor(rank);
        var high = (int)Math.Ceiling(rank);
        if (low == high)
        {
            return sortedValues[low];
        }

        var w = rank - low;
        return sortedValues[low] * (1 - w) + sortedValues[high] * w;
    }
}

public sealed record BenchmarkSummary(
    string Scenario,
    int Warmup,
    int PlannedIterations,
    int ExecutedIterations,
    int WarmupFailures,
    int Success,
    int Failed,
    double AvgMs,
    double P50Ms,
    double P95Ms,
    double P99Ms,
    double ThroughputPerSec,
    DateTime GeneratedAtUtc,
    IReadOnlyList<BenchmarkIteration> Iterations);

public sealed record BenchmarkIteration(int Index, double ElapsedMs, bool Success, string Message);
