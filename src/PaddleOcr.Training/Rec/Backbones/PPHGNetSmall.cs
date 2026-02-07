using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// PPHGNetSmall backbone：HG-Net 小型化版本，用于 PP-OCRv4 等。
/// 结构：Stem -> Stage1..4（每个 Stage 包含多个 HGBlock）。
/// 参考 ppocr/modeling/backbones/rec_hgnet.py。
/// </summary>
public sealed class PPHGNetSmall : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _stem;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _stages;
    private readonly Module<Tensor, Tensor> _pool;
    public int OutChannels { get; }

    /// <param name="inChannels">输入通道数</param>
    public PPHGNetSmall(int inChannels = 3) : base(nameof(PPHGNetSmall))
    {
        // Stem: 3 层卷积做初步特征提取和下采样
        _stem = Sequential(
            ConvBnAct(inChannels, 24, 3, (2, 1), 1),
            ConvBnAct(24, 32, 3, (1, 1), 1),
            ConvBnAct(32, 64, 3, (2, 1), 1)
        );

        // Stage 配置：(inChannels, outChannels, numBlocks, strideY, strideX, useSe)
        var stageConfigs = new[]
        {
            (64, 128, 2, 2, 1, false),    // stage1
            (128, 256, 3, 2, 1, true),     // stage2
            (256, 512, 3, 2, 1, true),     // stage3
            (512, 768, 2, 1, 1, true)      // stage4（不下采样）
        };

        _stages = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        foreach (var (stageIn, stageOut, numBlocks, sy, sx, useSe) in stageConfigs)
        {
            _stages.Add(BuildStage(stageIn, stageOut, numBlocks, sy, sx, useSe));
        }

        OutChannels = stageConfigs[^1].Item2;
        _pool = AdaptiveAvgPool2d(new long[] { 1, 40 });
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _stem.call(input);
        foreach (var stage in _stages)
        {
            x = stage.call(x);
        }

        return _pool.call(x);
    }

    private static Module<Tensor, Tensor> BuildStage(int inCh, int outCh, int numBlocks, int strideY, int strideX, bool useSe)
    {
        var blocks = new List<Module<Tensor, Tensor>>();

        // 第一个 block 执行下采样
        blocks.Add(new HGBlock(inCh, outCh, strideY, strideX, useSe));

        // 剩余 blocks 不下采样
        for (var i = 1; i < numBlocks; i++)
        {
            blocks.Add(new HGBlock(outCh, outCh, 1, 1, useSe));
        }

        return Sequential(blocks.ToArray());
    }

    private static Module<Tensor, Tensor> ConvBnAct(int inCh, int outCh, int kernel, (int, int) stride, int padding)
    {
        return Sequential(
            Conv2d(inCh, outCh, (kernel, kernel), stride: (stride.Item1, stride.Item2), padding: (padding, padding), bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
    }
}

/// <summary>
/// HGBlock：HG-Net 的基本 block，包含 Depthwise + Pointwise + 残差连接 + 可选 SE。
/// </summary>
internal sealed class HGBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _dwConv;
    private readonly Module<Tensor, Tensor> _pwConv;
    private readonly Module<Tensor, Tensor>? _downsample;
    private readonly Module<Tensor, Tensor>? _se;

    public HGBlock(int inCh, int outCh, int strideY, int strideX, bool useSe) : base(nameof(HGBlock))
    {
        // Depthwise + BN + Act
        _dwConv = Sequential(
            Conv2d(inCh, inCh, (3, 3), stride: (strideY, strideX), padding: (1, 1), groups: inCh, bias: false),
            BatchNorm2d(inCh),
            ReLU()
        );

        // Pointwise + BN + Act
        _pwConv = Sequential(
            Conv2d(inCh, outCh, 1, bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );

        // 残差连接下采样（如果维度或步长不同）
        if (inCh != outCh || strideY != 1 || strideX != 1)
        {
            _downsample = Sequential(
                Conv2d(inCh, outCh, (1, 1), stride: ((long)strideY, (long)strideX), bias: false),
                BatchNorm2d(outCh)
            );
        }

        // SE 模块
        if (useSe)
        {
            var reducedCh = Math.Max(1, outCh / 4);
            _se = new SEBlock(outCh, reducedCh);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var residual = _downsample is not null ? _downsample.call(input) : input;

        var x = _dwConv.call(input);
        x = _pwConv.call(x);

        if (_se is not null)
        {
            x = _se.call(x);
        }

        return x + residual;
    }
}

/// <summary>
/// SEBlock：Squeeze-and-Excitation block（用于 HGNet）。
/// </summary>
internal sealed class SEBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fc;

    public SEBlock(int channels, int reducedCh) : base(nameof(SEBlock))
    {
        _fc = Sequential(
            Linear(channels, reducedCh),
            ReLU(),
            Linear(reducedCh, channels),
            Sigmoid()
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var shape = input.shape;
        var b = shape[0];
        var c = shape[1];
        using var pooled = functional.adaptive_avg_pool2d(input, new long[] { 1, 1 }).reshape(b, c);
        var scale = _fc.call(pooled).reshape(b, c, 1, 1);
        return input * scale;
    }
}
