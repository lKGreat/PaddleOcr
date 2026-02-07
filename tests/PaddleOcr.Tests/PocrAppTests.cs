using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Config;
using PaddleOcr.Core.Cli;
using PaddleOcr.Inference;
using PaddleOcr.Tools;

namespace PaddleOcr.Tests;

public sealed class PocrAppTests
{
    [Fact]
    public async Task RunAsync_Should_Route_To_Infer_Executor()
    {
        var training = new ProbeExecutor("training");
        var infer = new ProbeExecutor("inference");
        var export = new ProbeExecutor("export");
        var service = new ProbeExecutor("service");
        var e2e = new ProbeExecutor("e2e");

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            training,
            infer,
            export,
            service,
            e2e);

        var code = await app.RunAsync(["infer", "system", "--image_dir", "./imgs"]);

        code.Should().Be(0);
        infer.Calls.Should().ContainSingle(c => c == "system");
    }

    private sealed class ProbeExecutor(string name) : ICommandExecutor
    {
        public List<string> Calls { get; } = [];

        public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
        {
            Calls.Add(subCommand);
            return Task.FromResult(CommandResult.Ok($"{name}:{subCommand}"));
        }
    }

    [Fact]
    public async Task RunAsync_Should_Fail_For_System_Without_Rec_Model()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new InferenceExecutor(),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"));

        var code = await app.RunAsync(["infer", "system", "--image_dir", "./imgs", "--use_onnx", "true"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_Should_Route_Convert_CheckJsonModel_To_Export()
    {
        var export = new ProbeExecutor("export");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            export,
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"));

        var code = await app.RunAsync(["convert", "check-json-model", "--json_model_dir", "./m"]);

        code.Should().Be(0);
        export.Calls.Should().ContainSingle(c => c == "convert:check-json-model");
    }

    [Fact]
    public async Task RunAsync_ConfigCheck_Should_Validate_Config()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_cfg_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var cfg = Path.Combine(dir, "ok.yml");
        await File.WriteAllTextAsync(cfg, "Global:\n  save_model_dir: ./out\nArchitecture:\n  model_type: cls\n");

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"));

        var code = await app.RunAsync(["config", "check", "-c", cfg]);
        code.Should().Be(0);
    }

    [Fact]
    public async Task RunAsync_ConfigDiff_Should_Report_Diff()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_cfg_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var a = Path.Combine(dir, "a.yml");
        var b = Path.Combine(dir, "b.yml");
        await File.WriteAllTextAsync(a, "Global:\n  epoch_num: 1\nArchitecture:\n  model_type: cls\n");
        await File.WriteAllTextAsync(b, "Global:\n  epoch_num: 2\nArchitecture:\n  model_type: cls\n");

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"));

        var code = await app.RunAsync(["config", "diff", "--base", a, "--target", b]);
        code.Should().Be(0);
    }
}
