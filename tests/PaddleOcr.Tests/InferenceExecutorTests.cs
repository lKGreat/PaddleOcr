using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Inference;

namespace PaddleOcr.Tests;

public sealed class InferenceExecutorTests
{
    [Fact]
    public async Task InferTable_Should_Require_TableModelDir()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--use_onnx"] = "true"
        });

        var result = await executor.ExecuteAsync("table", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--table_model_dir");
    }

    [Fact]
    public async Task InferKie_Should_Require_Onnx_Mode()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--kie_model_dir"] = "./kie.onnx",
            ["--use_onnx"] = "false"
        });

        var result = await executor.ExecuteAsync("kie", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--use_onnx=true");
    }

    [Fact]
    public async Task InferKieSer_Should_Require_Model_Path()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--use_onnx"] = "true"
        });

        var result = await executor.ExecuteAsync("kie-ser", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--ser_model_dir");
    }

    [Fact]
    public async Task InferKieRe_Should_Require_Model_Path()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--use_onnx"] = "true"
        });

        var result = await executor.ExecuteAsync("kie-re", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--re_model_dir");
    }

    private static PaddleOcr.Core.Cli.ExecutionContext NewContext(IReadOnlyDictionary<string, string> options)
    {
        return new PaddleOcr.Core.Cli.ExecutionContext(
            NullLogger.Instance,
            ["infer", "x"],
            "dummy.yml",
            new Dictionary<string, object?>(),
            options,
            []);
    }
}
