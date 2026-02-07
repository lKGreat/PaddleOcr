using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// EfficientNetB3 backbone：MBConv blocks + SE 模块。
/// 用于需要更强视觉特征的 rec 任务。
/// 参考 EfficientNet-B3 架构（Tan &amp; Le, 2019）。
/// </summary>
public sealed class EfficientNetB3 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _stem;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _headConv;
    private readonly Module<Tensor, Tensor> _pool;
    public int OutChannels { get; }

    /// <param name="inChannels">输入通道数</param>
    public EfficientNetB3(int inChannels = 3) : base(nameof(EfficientNetB3))
    {
        // Stem conv
        _stem = Sequential(
            Conv2d(inChannels, 40, (3, 3), stride: (2, 1), padding: (1, 1), bias: false),
            BatchNorm2d(40),
            SiLU()
        );

        // MBConv block 配置：(expandRatio, inCh, outCh, numRepeats, strideY, strideX, kernelSize)
        // EfficientNet-B3 with compound scaling, modified strides for OCR (height downsample only)
        var blockConfigs = new[]
        {
            (1, 40, 24, 2, 1, 1, 3),
            (6, 24, 32, 3, 2, 1, 3),
            (6, 32, 48, 3, 2, 1, 5),
            (6, 48, 96, 5, 2, 1, 3),
            (6, 96, 136, 5, 1, 1, 5),
            (6, 136, 232, 6, 2, 1, 5),
            (6, 232, 384, 2, 1, 1, 3)
        };

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        foreach (var (expand, bIn, bOut, repeats, sy, sx, kernel) in blockConfigs)
        {
            // 第一个 repeat 可能有下采样
            _blocks.Add(new MBConvBlock(bIn, bOut, expand, kernel, sy, sx));
            for (var i = 1; i < repeats; i++)
            {
                _blocks.Add(new MBConvBlock(bOut, bOut, expand, kernel, 1, 1));
            }
        }

        // Head conv
        var lastOutCh = blockConfigs[^1].Item3;
        var headCh = 1536;
        _headConv = Sequential(
            Conv2d(lastOutCh, headCh, 1, bias: false),
            BatchNorm2d(headCh),
            SiLU()
        );

        OutChannels = headCh;
        _pool = AdaptiveAvgPool2d(new long[] { 1, 40 });
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _stem.call(input);
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        x = _headConv.call(x);
        return _pool.call(x);
    }
}

/// <summary>
/// MBConv（Mobile Inverted Bottleneck Convolution）block + SE。
/// </summary>
internal sealed class MBConvBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor>? _expand;
    private readonly Module<Tensor, Tensor> _depthwise;
    private readonly Module<Tensor, Tensor> _se;
    private readonly Module<Tensor, Tensor> _project;
    private readonly bool _useResidual;

    public MBConvBlock(int inCh, int outCh, int expandRatio, int kernel, int strideY, int strideX) : base(nameof(MBConvBlock))
    {
        var midCh = inCh * expandRatio;
        _useResidual = inCh == outCh && strideY == 1 && strideX == 1;
        var padding = kernel / 2;

        // Expand phase (skip if expandRatio == 1)
        if (expandRatio != 1)
        {
            _expand = Sequential(
                Conv2d(inCh, midCh, 1, bias: false),
                BatchNorm2d(midCh),
                SiLU()
            );
        }

        // Depthwise conv
        _depthwise = Sequential(
            Conv2d(midCh, midCh, (kernel, kernel), stride: (strideY, strideX), padding: (padding, padding), groups: midCh, bias: false),
            BatchNorm2d(midCh),
            SiLU()
        );

        // SE module (reduction ratio = 4)
        var seCh = Math.Max(1, inCh / 4);
        _se = new MBConvSE(midCh, seCh);

        // Project phase
        _project = Sequential(
            Conv2d(midCh, outCh, 1, bias: false),
            BatchNorm2d(outCh)
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _expand is not null ? _expand.call(input) : input;
        x = _depthwise.call(x);
        x = _se.call(x);
        x = _project.call(x);

        if (_useResidual)
        {
            x = x + input;
        }

        return x;
    }
}

/// <summary>
/// SE 模块（用于 MBConv block）。
/// </summary>
internal sealed class MBConvSE : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fc;

    public MBConvSE(int channels, int reducedCh) : base(nameof(MBConvSE))
    {
        _fc = Sequential(
            Linear(channels, reducedCh),
            SiLU(),
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
