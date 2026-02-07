using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// MobileNetV1Enhance backbone：深度可分离卷积 + SE模块 + 非对称步长(2,1)。
/// 用于 PP-OCRv2/v3/v4 的 SVTR_LCNet 配置。
/// </summary>
public sealed class MobileNetV1Enhance : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _pool;
    public int OutChannels { get; }

    public MobileNetV1Enhance(int inChannels = 3, float scale = 0.5f) : base(nameof(MobileNetV1Enhance))
    {
        var channels = new[] { 32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024 };
        var strides = new[] { (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1) };

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();

        // 初始卷积
        var outCh = Math.Max(1, (int)(channels[0] * scale));
        _blocks.Add(ConvBnReLU(inChannels, outCh, 3, (2, 1), 1));

        var prevCh = outCh;
        for (var i = 0; i < strides.Length; i++)
        {
            outCh = Math.Max(1, (int)(channels[i + 1] * scale));
            var (sy, sx) = strides[i];
            _blocks.Add(DepthwiseSeparable(prevCh, outCh, sy, sx));
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

    private static Module<Tensor, Tensor> DepthwiseSeparable(int inCh, int outCh, int strideY, int strideX)
    {
        return Sequential(
            // Depthwise
            Conv2d(inCh, inCh, (3, 3), stride: (strideY, strideX), padding: (1, 1), groups: inCh, bias: false),
            BatchNorm2d(inCh),
            ReLU(),
            // Pointwise
            Conv2d(inCh, outCh, 1, bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
    }
}
