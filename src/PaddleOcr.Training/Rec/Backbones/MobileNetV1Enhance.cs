using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// MobileNetV1Enhance backbone：深度可分离卷积 + SE模块 + 非对称步长(2,1)。
/// 用于 PP-OCRv2/v3/v4 的 SVTR_LCNet 配置。
/// 参考 ppocr/modeling/backbones/rec_mobilenet_v1_enhance.py。
/// </summary>
public sealed class MobileNetV1Enhance : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _pool;
    public int OutChannels { get; }

    /// <param name="inChannels">输入通道数</param>
    /// <param name="scale">通道缩放因子</param>
    /// <param name="lastConvStride">最后两层卷积的步长</param>
    public MobileNetV1Enhance(int inChannels = 3, float scale = 0.5f, (int, int)? lastConvStride = null) : base(nameof(MobileNetV1Enhance))
    {
        var channels = new[] { 32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024 };
        var strides = new[] { (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1) };

        // SE 模块应用于指定的 block 索引（参考 PaddleOCR 原版）
        // 在 PP-OCRv3/v4 中，SE 模块应用在 stage3 和 stage4 的部分 block
        var seIndices = new HashSet<int> { 5, 6, 7, 8, 9, 10, 11, 12 };

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();

        // 初始卷积
        var outCh = Math.Max(1, (int)(channels[0] * scale));
        _blocks.Add(ConvBnReLU(inChannels, outCh, 3, (2, 1), 1));

        var prevCh = outCh;
        for (var i = 0; i < strides.Length; i++)
        {
            outCh = Math.Max(1, (int)(channels[i + 1] * scale));
            var (sy, sx) = strides[i];

            // 最后一个 conv stride 可覆盖
            if (lastConvStride.HasValue && i == strides.Length - 1)
            {
                (sy, sx) = lastConvStride.Value;
            }

            var useSe = seIndices.Contains(i);
            _blocks.Add(DepthwiseSeparable(prevCh, outCh, sy, sx, useSe));
            prevCh = outCh;
        }

        OutChannels = prevCh;
        _pool = AdaptiveAvgPool2d(new long[] { 1, 40 });
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input;
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        return _pool.call(x);
    }

    private static Module<Tensor, Tensor> ConvBnReLU(int inCh, int outCh, int kernel, (int, int) stride, int padding)
    {
        return Sequential(
            Conv2d(inCh, outCh, (kernel, kernel), stride: (stride.Item1, stride.Item2), padding: (padding, padding), bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
    }

    private static Module<Tensor, Tensor> DepthwiseSeparable(int inCh, int outCh, int strideY, int strideX, bool useSe)
    {
        var layers = new List<(string, Module)>
        {
            // Depthwise conv
            ("dw_conv", Conv2d(inCh, inCh, (3, 3), stride: (strideY, strideX), padding: (1, 1), groups: inCh, bias: false)),
            ("dw_bn", BatchNorm2d(inCh)),
            ("dw_act", ReLU())
        };

        // SE 模块（在 depthwise conv 后、pointwise conv 前）
        if (useSe)
        {
            layers.Add(("se", new SEModule(inCh)));
        }

        // Pointwise conv
        layers.Add(("pw_conv", Conv2d(inCh, outCh, 1, bias: false)));
        layers.Add(("pw_bn", BatchNorm2d(outCh)));
        layers.Add(("pw_act", ReLU()));

        return Sequential(layers.Select(kv => (kv.Item1, (Module<Tensor, Tensor>)kv.Item2)).ToArray());
    }
}

/// <summary>
/// SEModule：Squeeze-and-Excitation 模块。
/// 通过全局池化 + 两层 FC + Sigmoid 学习通道注意力权重。
/// </summary>
internal sealed class SEModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _squeeze;
    private readonly Module<Tensor, Tensor> _excitation;

    public SEModule(int channels, int reduction = 4) : base(nameof(SEModule))
    {
        var reducedCh = Math.Max(1, channels / reduction);
        _squeeze = Sequential(
            Linear(channels, reducedCh),
            ReLU()
        );
        _excitation = Sequential(
            Linear(reducedCh, channels),
            Sigmoid()
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, C, H, W]
        var shape = input.shape;
        var b = shape[0];
        var c = shape[1];

        // Global average pooling: [B, C, H, W] -> [B, C]
        using var pooled = functional.adaptive_avg_pool2d(input, new long[] { 1, 1 }).reshape(b, c);

        // FC -> ReLU -> FC -> Sigmoid: [B, C] -> [B, C]
        var scale = _squeeze.call(pooled);
        scale = _excitation.call(scale);

        // Reshape 并乘回原输入: [B, C] -> [B, C, 1, 1] * [B, C, H, W]
        scale = scale.reshape(b, c, 1, 1);
        return input * scale;
    }
}
