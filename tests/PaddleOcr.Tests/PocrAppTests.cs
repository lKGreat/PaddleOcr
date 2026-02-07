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
}
