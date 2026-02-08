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

    [Fact]
    public async Task InferDet_Should_Accept_East_And_Reach_ModelValidation()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--det_model_dir"] = "./det.onnx",
            ["--use_onnx"] = "true",
            ["--det_algorithm"] = "EAST"
        });

        var result = await executor.ExecuteAsync("det", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("det model not found");
    }

    [Fact]
    public async Task InferDet_Should_Reject_Unknown_DetAlgorithm()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--det_model_dir"] = "./det.onnx",
            ["--use_onnx"] = "true",
            ["--det_algorithm"] = "UNKNOWN"
        });

        var result = await executor.ExecuteAsync("det", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--det_algorithm unsupported");
    }

    [Fact]
    public async Task InferDet_Should_Reject_Invalid_BoxType()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--det_model_dir"] = "./det.onnx",
            ["--use_onnx"] = "true",
            ["--det_box_type"] = "rect"
        });

        var result = await executor.ExecuteAsync("det", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("quad|poly");
    }

    [Fact]
    public async Task InferDet_Should_Reject_Invalid_DbScoreMode()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--det_model_dir"] = "./det.onnx",
            ["--use_onnx"] = "true",
            ["--det_db_score_mode"] = "invalid"
        });

        var result = await executor.ExecuteAsync("det", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--det_db_score_mode must be fast|slow");
    }

    [Fact]
    public async Task InferDet_Should_Reject_Invalid_SliceMergeIou()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--det_model_dir"] = "./det.onnx",
            ["--use_onnx"] = "true",
            ["--det_slice_merge_iou"] = "2"
        });

        var result = await executor.ExecuteAsync("det", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--det_slice_merge_iou must be in [0,1]");
    }

    [Fact]
    public async Task InferRec_Should_Use_Paddle_When_UseOnnx_Is_False()
    {
        var executor = new InferenceExecutor();
        var imageDir = Path.Combine(Path.GetTempPath(), $"pocr-rec-img-{Guid.NewGuid():N}");
        var modelDir = Path.Combine(Path.GetTempPath(), $"pocr-rec-model-{Guid.NewGuid():N}");
        Directory.CreateDirectory(imageDir);
        Directory.CreateDirectory(modelDir);
        File.WriteAllText(Path.Combine(modelDir, "inference.json"), "{}");
        File.WriteAllText(Path.Combine(modelDir, "inference.pdiparams"), string.Empty);

        try
        {
            var context = NewContext(new Dictionary<string, string>
            {
                ["--image_dir"] = imageDir,
                ["--rec_model_dir"] = modelDir,
                ["--use_onnx"] = "false"
            });

            var result = await executor.ExecuteAsync("rec", context);

            result.Success.Should().BeFalse();
            result.Message.Should().Contain("No image found in");
        }
        finally
        {
            if (Directory.Exists(imageDir))
            {
                Directory.Delete(imageDir, true);
            }

            if (Directory.Exists(modelDir))
            {
                Directory.Delete(modelDir, true);
            }
        }
    }

    [Fact]
    public async Task InferRec_Should_Reject_Invalid_RuntimeBackend()
    {
        var executor = new InferenceExecutor();
        var context = NewContext(new Dictionary<string, string>
        {
            ["--image_dir"] = "./imgs",
            ["--rec_model_dir"] = "./rec.onnx",
            ["--runtime_backend"] = "cuda"
        });

        var result = await executor.ExecuteAsync("rec", context);

        result.Success.Should().BeFalse();
        result.Message.Should().Contain("--runtime_backend must be onnx|paddle");
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
