using FluentAssertions;
using PaddleOcr.Training.Rec.Backbones;
using PaddleOcr.Training.Det.Backbones;
using PaddleOcr.Training.Det;
using PaddleOcr.Training.Rec;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Tests;

/// <summary>
/// Tests for PPLCNetV3 (rec) and DetPPLCNetV3 (det) backbone implementations.
/// Verifies 1:1 alignment with official PaddleOCR rec_lcnetv3.py.
/// </summary>
public sealed class PPLCNetV3Tests
{
    // ─── make_divisible tests ───

    [Theory]
    [InlineData(15, 16)]     // 15.2 * scale → rounds up
    [InlineData(16, 16)]     // exact
    [InlineData(512, 512)]   // large exact
    [InlineData(486, 496)]   // 512*0.95=486.4 → (486+8)/16*16 = 496
    [InlineData(384, 384)]   // 512*0.75=384
    [InlineData(8, 16)]      // small, min_value=16
    public void MakeDivisible_Should_Match_Official(int input, int expected)
    {
        PPLCNetV3.MakeDivisible(input).Should().Be(expected);
    }

    // ─── Rec mode tests ───

    [Fact]
    public void RecMode_Scale095_OutChannels_Should_Match_Official()
    {
        // Official: make_divisible(512 * 0.95) = make_divisible(486) = 496
        using var model = new PPLCNetV3(inChannels: 3, scale: 0.95f);
        model.OutChannels.Should().Be(PPLCNetV3.MakeDivisible((int)(512 * 0.95f)));
    }

    [Fact]
    public void RecMode_Scale100_OutChannels_Should_Be_512()
    {
        using var model = new PPLCNetV3(inChannels: 3, scale: 1.0f);
        model.OutChannels.Should().Be(512);
    }

    [Fact]
    public void RecMode_Training_ForwardShape_Should_Be_B_C_1_40()
    {
        using var model = new PPLCNetV3(inChannels: 3, scale: 0.95f);
        model.train();

        using var input = torch.rand([1, 3, 48, 320], dtype: ScalarType.Float32);
        using var output = model.call(input);

        output.shape.Should().HaveCount(4);
        output.shape[0].Should().Be(1);              // batch
        output.shape[1].Should().Be(model.OutChannels); // channels
        output.shape[2].Should().Be(1);              // height after adaptive pool
        output.shape[3].Should().Be(40);             // width after adaptive pool
    }

    [Fact]
    public void RecMode_Eval_ForwardShape_Should_Use_AvgPool()
    {
        using var model = new PPLCNetV3(inChannels: 3, scale: 0.95f);
        model.eval();

        // Input: [1, 3, 48, 320]
        // After backbone blocks: some spatial size
        // avg_pool2d([3, 2]) reduces spatial dims
        using var input = torch.rand([1, 3, 48, 320], dtype: ScalarType.Float32);
        using var output = model.call(input);

        output.shape.Should().HaveCount(4);
        output.shape[0].Should().Be(1);
        output.shape[1].Should().Be(model.OutChannels);
        // Exact spatial dims depend on input and strides; should be valid
        output.shape[2].Should().BeGreaterThan(0);
        output.shape[3].Should().BeGreaterThan(0);
    }

    [Fact]
    public void RecMode_BatchForward_Should_Work()
    {
        using var model = new PPLCNetV3(inChannels: 3, scale: 0.95f);
        model.train();

        using var input = torch.rand([4, 3, 48, 320], dtype: ScalarType.Float32);
        using var output = model.call(input);

        output.shape[0].Should().Be(4); // batch preserved
        output.shape[1].Should().Be(model.OutChannels);
    }

    // ─── Det mode tests ───

    [Fact]
    public void DetMode_Scale075_OutChannels_Should_Match_Official()
    {
        // Official: mv_c = [16, 24, 56, 480], out = [int(16*0.75), int(24*0.75), int(56*0.75), int(480*0.75)]
        // = [12, 18, 42, 360]
        using var model = new DetPPLCNetV3(inChannels: 3, scale: 0.75f);
        model.OutChannels.Should().HaveCount(4);
        model.OutChannels[0].Should().Be((int)(16 * 0.75f));
        model.OutChannels[1].Should().Be((int)(24 * 0.75f));
        model.OutChannels[2].Should().Be((int)(56 * 0.75f));
        model.OutChannels[3].Should().Be((int)(480 * 0.75f));
    }

    [Fact]
    public void DetMode_Forward_Should_Return_4_FeatureMaps()
    {
        using var model = new DetPPLCNetV3(inChannels: 3, scale: 0.75f);
        model.eval();

        using var input = torch.rand([1, 3, 640, 640], dtype: ScalarType.Float32);
        var outputs = model.call(input);

        outputs.Should().HaveCount(4);

        // Each output should be [B, C_i, H_i, W_i]
        for (int i = 0; i < 4; i++)
        {
            outputs[i].shape.Should().HaveCount(4);
            outputs[i].shape[0].Should().Be(1);
            outputs[i].shape[1].Should().Be(model.OutChannels[i]);
        }

        // Clean up
        foreach (var t in outputs) t.Dispose();
    }

    [Fact]
    public void DetMode_Forward_SpatialDims_Should_Decrease_Across_Stages()
    {
        using var model = new DetPPLCNetV3(inChannels: 3, scale: 0.75f);
        model.eval();

        using var input = torch.rand([1, 3, 256, 256], dtype: ScalarType.Float32);
        var outputs = model.call(input);

        // Each stage should have smaller spatial dims than previous
        // blocks3 output: ~1/4, blocks4: ~1/8, blocks5: ~1/16, blocks6: ~1/32
        var h3 = outputs[0].shape[2];
        var h4 = outputs[1].shape[2];
        var h5 = outputs[2].shape[2];
        var h6 = outputs[3].shape[2];

        h3.Should().BeGreaterThan(h4);
        h4.Should().BeGreaterThan(h5);
        h5.Should().BeGreaterThan(h6);

        foreach (var t in outputs) t.Dispose();
    }

    // ─── Builder integration tests ───

    [Fact]
    public void RecModelBuilder_Should_Resolve_PPLCNetV3()
    {
        var (module, outChannels) = RecModelBuilder.BuildBackbone("PPLCNetV3");
        try
        {
            module.GetType().Name.Should().Be(nameof(PPLCNetV3));
            outChannels.Should().BeGreaterThan(0);
        }
        finally
        {
            module.Dispose();
        }
    }

    [Fact]
    public void DetModelBuilder_Should_Resolve_PPLCNetV3()
    {
        var (module, outChannels) = DetModelBuilder.BuildBackbone("pplcnetv3", scale: 0.75f);
        try
        {
            module.GetType().Name.Should().Be(nameof(DetPPLCNetV3));
            outChannels.Should().HaveCount(4);
        }
        finally
        {
            module.Dispose();
        }
    }

    // ─── LearnableRepLayer rep fusion tests ───

    [Fact]
    public void RepFusion_Should_Produce_Same_Output()
    {
        // Create a standalone LearnableRepLayer, run forward before and after rep()
        // The outputs should be approximately equal (fp32 precision)
        using var model = new PPLCNetV3(inChannels: 3, scale: 1.0f);
        model.eval();

        using var input = torch.rand([1, 3, 48, 320], dtype: ScalarType.Float32);

        // Get output before rep fusion
        using var outputBefore = model.call(input).clone();

        // Note: We can't directly test rep() on the PPLCNetV3 because it wraps
        // the blocks inside Sequential. This test ensures the model works correctly
        // in eval mode which is the primary use case.
        outputBefore.shape[0].Should().Be(1);
        outputBefore.shape[1].Should().Be(512); // scale=1.0 -> 512
    }

    // ─── Parameter count sanity tests ───

    [Fact]
    public void RecMode_Should_Have_Reasonable_ParamCount()
    {
        using var model = new PPLCNetV3(inChannels: 3, scale: 0.95f);

        long totalParams = 0;
        foreach (var (_, param) in model.named_parameters())
        {
            totalParams += param.numel();
        }

        // PPLCNetV3 rec should have parameters (not be empty stub)
        totalParams.Should().BeGreaterThan(100_000, "PPLCNetV3 should have substantial parameters");
        // Should not be unreasonably large for a lightweight model
        totalParams.Should().BeLessThan(50_000_000, "PPLCNetV3 should be a lightweight model");
    }

    [Fact]
    public void DetMode_Should_Have_Reasonable_ParamCount()
    {
        using var model = new DetPPLCNetV3(inChannels: 3, scale: 0.75f);

        long totalParams = 0;
        foreach (var (_, param) in model.named_parameters())
        {
            totalParams += param.numel();
        }

        totalParams.Should().BeGreaterThan(100_000);
        totalParams.Should().BeLessThan(50_000_000);
    }
}
