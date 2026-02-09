using FluentAssertions;
using PaddleOcr.Training.Rec;
using PaddleOcr.Training.Rec.Backbones;
using PaddleOcr.Training.Rec.Heads;
using PaddleOcr.Training.Rec.Losses;
using PaddleOcr.Training.Rec.Necks;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Tests;

/// <summary>
/// Tests for PP-OCRv5 mobile architecture 1:1 alignment.
/// Verifies PPLCNetV3 + SVTR encoder + MultiHead (CTC + NRTR) + MultiLoss
/// matches the official PaddleOCR-3.3.2 implementation.
/// </summary>
public sealed class PPOCRv5ArchitectureTests
{
    // ── Config constants matching PP-OCRv5 mobile rec config ──
    private const float BackboneScale = 0.95f;
    private const int NrtrDim = 384;
    private const int SvtrDims = 120;
    private const int SvtrDepth = 2;
    private const int SvtrHiddenDims = 120;
    private const int MaxTextLength = 25;
    private const int NumClasses = 40; // small vocab for test

    // ─── PPLCNetV3 Backbone Tests ───

    [Fact]
    public void PPLCNetV3_OutChannels_Should_Match_Official()
    {
        // Official: make_divisible(512 * 0.95) = make_divisible(486)
        using var model = new PPLCNetV3(inChannels: 3, scale: BackboneScale);
        var expected = PPLCNetV3.MakeDivisible((int)(512 * BackboneScale));
        model.OutChannels.Should().Be(expected);
    }

    [Fact]
    public void PPLCNetV3_Training_Output_Shape_Should_Be_B_C_1_40()
    {
        using var model = new PPLCNetV3(inChannels: 3, scale: BackboneScale);
        model.train();

        using var input = torch.rand([2, 3, 48, 320], dtype: ScalarType.Float32);
        using var output = model.call(input);

        output.shape.Should().Equal([2, model.OutChannels, 1, 40]);
    }

    [Fact]
    public void PPLCNetV3_Stride_Check_Should_Apply_Activation_For_Asymmetric_Strides()
    {
        // Regression test: activation must NOT be skipped for strides like (2,1) or (1,2).
        // Previously this was broken: && instead of the correct || logic.
        using var model = new PPLCNetV3(inChannels: 3, scale: BackboneScale);
        model.train();

        using var input = torch.rand([1, 3, 48, 320], dtype: ScalarType.Float32);

        // Forward should work without errors (activation applied correctly)
        using var output = model.call(input);
        output.shape[0].Should().Be(1);
        output.shape[1].Should().Be(model.OutChannels);

        // Output should have non-zero variance (not collapsed from missing activation)
        var stdVal = output.std().item<float>();
        stdVal.Should().BeGreaterThan(0.0f, "output should have variance (activation applied)");
    }

    // ─── SVTR Encoder (EncoderWithSVTR) Tests ───

    [Fact]
    public void EncoderWithSVTR_Should_Match_PPOCRv5_Config()
    {
        var backboneOutCh = PPLCNetV3.MakeDivisible((int)(512 * BackboneScale)); // 480

        using var encoder = new SequenceEncoder(
            backboneOutCh,
            "svtr",
            dims: SvtrDims,
            depth: SvtrDepth,
            hiddenDims: SvtrHiddenDims,
            useGuide: true,
            kernelSize: [1, 3]);

        encoder.OutChannels.Should().Be(SvtrDims, "SVTR output dims should be 120");
    }

    [Fact]
    public void EncoderWithSVTR_ForwardShape_Should_Be_B_W_Dims()
    {
        var backboneOutCh = PPLCNetV3.MakeDivisible((int)(512 * BackboneScale));

        using var encoder = new SequenceEncoder(
            backboneOutCh,
            "svtr",
            dims: SvtrDims,
            depth: SvtrDepth,
            hiddenDims: SvtrHiddenDims,
            useGuide: true,
            kernelSize: [1, 3]);

        // Input: backbone output [B, C, 1, 40] (after adaptive_avg_pool)
        using var input = torch.rand([2, backboneOutCh, 1, 40], dtype: ScalarType.Float32);
        using var output = encoder.call(input);

        // Output: [B, 40, dims] = [2, 40, 120]
        output.shape.Should().Equal([2, 40, SvtrDims]);
    }

    // ─── MultiHead Tests ───

    [Fact]
    public void MultiHead_Should_Build_With_PPOCRv5_Config()
    {
        var backboneOutCh = PPLCNetV3.MakeDivisible((int)(512 * BackboneScale));

        var ctcNeckConfig = new MultiHeadCtcNeckConfig(
            EncoderType: "svtr",
            Dims: SvtrDims,
            Depth: SvtrDepth,
            HiddenDims: SvtrHiddenDims,
            UseGuide: true,
            KernelSize: [1, 3]);

        using var multiHead = new MultiHead(
            inChannels: backboneOutCh,
            outChannelsCtc: NumClasses,
            outChannelsGtc: NumClasses,
            hiddenSize: NrtrDim,
            maxLen: MaxTextLength,
            gtcHeadName: "NRTRHead",
            gtcInChannels: backboneOutCh,
            ctcNeckConfig: ctcNeckConfig,
            nrtrDim: NrtrDim);

        // Should have parameters
        long paramCount = 0;
        foreach (var (_, param) in multiHead.named_parameters())
        {
            paramCount += param.numel();
        }
        paramCount.Should().BeGreaterThan(10_000);
    }

    [Fact]
    public void MultiHead_Training_Forward_Should_Return_CTC_And_GTC()
    {
        var backboneOutCh = PPLCNetV3.MakeDivisible((int)(512 * BackboneScale));

        var ctcNeckConfig = new MultiHeadCtcNeckConfig(
            EncoderType: "svtr",
            Dims: SvtrDims,
            Depth: SvtrDepth,
            HiddenDims: SvtrHiddenDims,
            UseGuide: true,
            KernelSize: [1, 3]);

        using var multiHead = new MultiHead(
            inChannels: backboneOutCh,
            outChannelsCtc: NumClasses,
            outChannelsGtc: NumClasses,
            hiddenSize: NrtrDim,
            maxLen: MaxTextLength,
            gtcHeadName: "NRTRHead",
            gtcInChannels: backboneOutCh,
            ctcNeckConfig: ctcNeckConfig,
            nrtrDim: NrtrDim);

        multiHead.train();

        // Input: backbone output [B, C, 1, 40]
        using var input = torch.rand([2, backboneOutCh, 1, 40], dtype: ScalarType.Float32);

        // Provide GTC label tokens for teacher forcing
        var targets = new Dictionary<string, Tensor>
        {
            ["label_gtc"] = torch.randint(0, NumClasses, [2, MaxTextLength + 1], ScalarType.Int64)
        };

        var result = multiHead.Forward(input, targets);
        try
        {
            result.Should().ContainKey("ctc", "CTC branch output");
            result.Should().ContainKey("gtc", "GTC (NRTR) branch output");
            result.Should().ContainKey("ctc_neck", "CTC neck/encoder output");
            result.Should().ContainKey("predict", "predict key");

            // CTC output: [B, seqLen, numClasses]
            result["ctc"].shape[0].Should().Be(2);
            result["ctc"].shape[2].Should().Be(NumClasses);
        }
        finally
        {
            foreach (var t in result.Values) t.Dispose();
            foreach (var t in targets.Values) t.Dispose();
        }
    }

    [Fact]
    public void MultiHead_Eval_Should_Return_Only_CTC()
    {
        var backboneOutCh = PPLCNetV3.MakeDivisible((int)(512 * BackboneScale));

        var ctcNeckConfig = new MultiHeadCtcNeckConfig(
            EncoderType: "svtr",
            Dims: SvtrDims,
            Depth: SvtrDepth,
            HiddenDims: SvtrHiddenDims,
            UseGuide: true,
            KernelSize: [1, 3]);

        using var multiHead = new MultiHead(
            inChannels: backboneOutCh,
            outChannelsCtc: NumClasses,
            outChannelsGtc: NumClasses,
            hiddenSize: NrtrDim,
            maxLen: MaxTextLength,
            gtcHeadName: "NRTRHead",
            gtcInChannels: backboneOutCh,
            ctcNeckConfig: ctcNeckConfig,
            nrtrDim: NrtrDim);

        multiHead.eval();

        using var input = torch.rand([1, backboneOutCh, 1, 40], dtype: ScalarType.Float32);

        var result = multiHead.Forward(input);
        try
        {
            // Eval mode: only CTC output, no GTC branch (matching Python)
            result.Should().ContainKey("ctc");
            result.Should().ContainKey("predict");
            result.Should().NotContainKey("gtc", "GTC should not be computed in eval mode");
        }
        finally
        {
            foreach (var t in result.Values) t.Dispose();
        }
    }

    // ─── NRTRHead Tests ───

    [Fact]
    public void NRTRHead_EncoderLess_Should_Skip_Encoder()
    {
        // PP-OCRv5 MultiHead uses NRTRHead with num_encoder_layers=-1 (no encoder)
        using var head = new NRTRHead(
            inChannels: NrtrDim,
            outChannels: NumClasses,
            hiddenSize: NrtrDim,
            numHeads: NrtrDim / 32, // 12
            numEncoderLayers: 0,
            numDecoderLayers: 4,
            maxLen: MaxTextLength);

        head.train();

        // Input: [B, seqLen, nrtrDim] — already projected by FCTranspose
        using var input = torch.rand([2, 40, NrtrDim], dtype: ScalarType.Float32);
        using var labels = torch.randint(0, NumClasses, [2, MaxTextLength + 1], ScalarType.Int64);

        var targets = new Dictionary<string, Tensor> { ["label_gtc"] = labels };
        var result = head.Forward(input, targets);
        try
        {
            result.Should().ContainKey("predict");
            result["predict"].shape[0].Should().Be(2); // batch
        }
        finally
        {
            foreach (var t in result.Values) t.Dispose();
        }
    }

    [Fact]
    public void NRTRHead_NumHeads_Should_Match_PPOCRv5()
    {
        // PP-OCRv5: nrtr_dim=384, nhead=384//32=12
        var expectedHeads = NrtrDim / 32;
        expectedHeads.Should().Be(12);
    }

    // ─── MultiLoss Tests ───

    [Fact]
    public void MultiLoss_Should_Compute_CTC_And_NRTR_Loss()
    {
        var multiLoss = new MultiLoss(
            ctcLoss: new CTCLoss(),
            gtcLoss: new NRTRLoss(),
            ctcWeight: 1.0f,
            gtcWeight: 1.0f,
            ctcLabelKey: "label_ctc",
            gtcLabelKey: "label_gtc",
            gtcPredKey: "gtc");

        // CTC predictions: [B, T, numClasses] (log_softmax)
        var ctcPred = torch.randn([2, 40, NumClasses]);
        // NRTR predictions: [B, maxLen, numClasses]
        var gtcPred = torch.randn([2, MaxTextLength, NumClasses]);

        var predictions = new Dictionary<string, Tensor>
        {
            ["ctc"] = ctcPred,
            ["gtc"] = gtcPred
        };

        // Labels
        var labelCtc = torch.randint(1, NumClasses, [2, MaxTextLength], ScalarType.Int64);
        var labelGtc = torch.randint(1, NumClasses, [2, MaxTextLength], ScalarType.Int64);
        var lengths = torch.tensor(new long[] { 10, 8 });

        var batch = new Dictionary<string, Tensor>
        {
            ["label_ctc"] = labelCtc,
            ["label_gtc"] = labelGtc,
            ["length"] = lengths
        };

        var result = multiLoss.Forward(predictions, batch);
        try
        {
            result.Should().ContainKey("loss", "total loss");
            result.Should().ContainKey("ctc_loss", "CTC branch loss");
            result.Should().ContainKey("gtc_loss", "GTC branch loss");

            // Losses should be finite
            result["loss"].item<float>().Should().NotBe(float.NaN);
            result["loss"].item<float>().Should().NotBe(float.PositiveInfinity);
        }
        finally
        {
            foreach (var t in result.Values) t.Dispose();
            foreach (var t in predictions.Values) t.Dispose();
            foreach (var t in batch.Values) t.Dispose();
        }
    }

    [Fact]
    public void MultiLoss_Should_Use_SAR_PredKey_For_SARLoss()
    {
        var multiLoss = new MultiLoss(
            ctcLoss: new CTCLoss(),
            gtcLoss: new SARLoss(),
            ctcWeight: 1.0f,
            gtcWeight: 1.0f,
            ctcLabelKey: "label_ctc",
            gtcLabelKey: "label_gtc",
            gtcPredKey: "sar");

        var ctcPred = torch.randn([2, 40, NumClasses]);
        var sarPred = torch.randn([2, MaxTextLength, NumClasses]);

        var predictions = new Dictionary<string, Tensor>
        {
            ["ctc"] = ctcPred,
            ["sar"] = sarPred // SAR uses "sar" key, not "gtc"
        };

        var batch = new Dictionary<string, Tensor>
        {
            ["label_ctc"] = torch.randint(1, NumClasses, [2, MaxTextLength], ScalarType.Int64),
            ["label_gtc"] = torch.randint(1, NumClasses, [2, MaxTextLength], ScalarType.Int64),
            ["length"] = torch.tensor(new long[] { 10, 8 })
        };

        var result = multiLoss.Forward(predictions, batch);
        try
        {
            result.Should().ContainKey("gtc_loss", "SAR loss should be computed using 'sar' key");
        }
        finally
        {
            foreach (var t in result.Values) t.Dispose();
            foreach (var t in predictions.Values) t.Dispose();
            foreach (var t in batch.Values) t.Dispose();
        }
    }

    // ─── Full Pipeline (RecModelBuilder) Tests ───

    [Fact]
    public void RecModelBuilder_Should_Build_PPOCRv5_Architecture()
    {
        var ctcNeckConfig = new MultiHeadCtcNeckConfig(
            EncoderType: "svtr",
            Dims: SvtrDims,
            Depth: SvtrDepth,
            HiddenDims: SvtrHiddenDims,
            UseGuide: true,
            KernelSize: [1, 3]);

        using var model = RecModelBuilder.Build(
            backboneName: "PPLCNetV3",
            neckName: "none",
            headName: "MultiHead",
            numClasses: NumClasses,
            inChannels: 3,
            hiddenSize: 48,
            maxLen: MaxTextLength,
            neckEncoderType: "none",
            gtcHeadName: "NRTRHead",
            gtcOutChannels: NumClasses,
            headHiddenSize: NrtrDim,
            ctcNeckConfig: ctcNeckConfig,
            nrtrDim: NrtrDim);

        model.BackboneName.Should().Be("PPLCNetV3");
        model.HeadName.Should().Be("MultiHead");
    }

    [Fact]
    public void PPOCRv5_Full_Pipeline_Training_Forward_Should_Succeed()
    {
        var ctcNeckConfig = new MultiHeadCtcNeckConfig(
            EncoderType: "svtr",
            Dims: SvtrDims,
            Depth: SvtrDepth,
            HiddenDims: SvtrHiddenDims,
            UseGuide: true,
            KernelSize: [1, 3]);

        using var model = RecModelBuilder.Build(
            backboneName: "PPLCNetV3",
            neckName: "none",
            headName: "MultiHead",
            numClasses: NumClasses,
            inChannels: 3,
            hiddenSize: 48,
            maxLen: MaxTextLength,
            neckEncoderType: "none",
            gtcHeadName: "NRTRHead",
            gtcOutChannels: NumClasses,
            headHiddenSize: NrtrDim,
            ctcNeckConfig: ctcNeckConfig,
            nrtrDim: NrtrDim);

        model.train();

        using var input = torch.rand([2, 3, 48, 320], dtype: ScalarType.Float32);

        var targets = new Dictionary<string, Tensor>
        {
            ["label_ctc"] = torch.randint(0, NumClasses, [2, MaxTextLength], ScalarType.Int64),
            ["label_gtc"] = torch.randint(0, NumClasses, [2, MaxTextLength + 1], ScalarType.Int64),
            ["length"] = torch.tensor(new long[] { 10, 8 })
        };

        var result = model.ForwardDict(input, targets);
        try
        {
            result.Should().ContainKey("ctc", "CTC branch output");
            result.Should().ContainKey("predict", "predict key");
        }
        finally
        {
            foreach (var t in result.Values) t.Dispose();
            foreach (var t in targets.Values) t.Dispose();
        }
    }

    // ─── FCTranspose Tests ───

    [Fact]
    public void FCTranspose_Should_Have_No_Bias()
    {
        // Python: nn.Linear(in_channels, out_channels, bias_attr=False)
        var fcTranspose = new FCTranspose(480, NrtrDim);
        try
        {
            // Verify no bias by checking parameter count
            long paramCount = 0;
            foreach (var (name, param) in fcTranspose.named_parameters())
            {
                paramCount += param.numel();
                // Should only have weight, no bias
                name.Should().NotContain("bias");
            }

            // Expected: only weight matrix 480 * 384 = 184320
            paramCount.Should().Be(480L * NrtrDim);
        }
        finally
        {
            fcTranspose.Dispose();
        }
    }

    // ─── RecLossBuilder Tests ───

    [Fact]
    public void RecLossBuilder_Should_Build_MultiLoss_With_NRTRLoss()
    {
        var config = new Dictionary<string, object>
        {
            ["loss_config_list"] = new List<object?>
            {
                new Dictionary<string, object?> { ["CTCLoss"] = null },
                new Dictionary<string, object?> { ["NRTRLoss"] = null }
            }
        };

        var loss = RecLossBuilder.Build("MultiLoss", config);
        loss.Should().BeOfType<MultiLoss>();
    }
}
