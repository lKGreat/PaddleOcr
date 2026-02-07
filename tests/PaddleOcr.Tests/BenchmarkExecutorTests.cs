using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Benchmark;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.Tests;

public sealed class BenchmarkExecutorTests
{
    [Fact]
    public async Task ExecuteAsync_Should_Generate_Report_For_Valid_Scenario()
    {
        var probe = new ProbeExecutor();
        var executor = new BenchmarkExecutor(probe, probe, probe, probe, probe);
        var report = Path.Combine(Path.GetTempPath(), $"bench_{Guid.NewGuid():N}.json");
        var context = new PaddleOcr.Core.Cli.ExecutionContext(
            NullLogger.Instance,
            ["benchmark", "run"],
            null,
            new Dictionary<string, object?>(),
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["--scenario"] = "infer:system",
                ["--warmup"] = "1",
                ["--iterations"] = "3",
                ["--report_json"] = report
            },
            []);

        var result = await executor.ExecuteAsync("run", context);

        result.Success.Should().BeTrue();
        File.Exists(report).Should().BeTrue();
        probe.Calls.Should().Be(4);
    }

    [Fact]
    public async Task ExecuteAsync_Should_Fail_For_Unsupported_Scenario()
    {
        var probe = new ProbeExecutor();
        var executor = new BenchmarkExecutor(probe, probe, probe, probe, probe);
        var context = new PaddleOcr.Core.Cli.ExecutionContext(
            NullLogger.Instance,
            ["benchmark", "run"],
            null,
            new Dictionary<string, object?>(),
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["--scenario"] = "infer:unknown"
            },
            []);

        var result = await executor.ExecuteAsync("run", context);
        result.Success.Should().BeFalse();
    }

    [Fact]
    public async Task ExecuteAsync_Should_Apply_Service_Profile_When_Provided()
    {
        var probe = new ProbeExecutor();
        var executor = new BenchmarkExecutor(probe, probe, probe, probe, probe);
        var context = new PaddleOcr.Core.Cli.ExecutionContext(
            NullLogger.Instance,
            ["benchmark", "run"],
            null,
            new Dictionary<string, object?>(),
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["--scenario"] = "service:test",
                ["--profile"] = "stress",
                ["--warmup"] = "0",
                ["--iterations"] = "1"
            },
            []);

        var result = await executor.ExecuteAsync("run", context);

        result.Success.Should().BeTrue();
        probe.LastOptions.Should().ContainKey("--parallel").WhoseValue.Should().Be("8");
        probe.LastOptions.Should().ContainKey("--stress_rounds").WhoseValue.Should().Be("5");
    }

    private sealed class ProbeExecutor : ICommandExecutor
    {
        public int Calls { get; private set; }
        public IReadOnlyDictionary<string, string> LastOptions { get; private set; } =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
        {
            Calls++;
            LastOptions = context.Options;
            return Task.FromResult(CommandResult.Ok(subCommand));
        }
    }
}
