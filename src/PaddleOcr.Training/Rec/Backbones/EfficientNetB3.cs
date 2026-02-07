using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// EfficientNetB3 backbone：用于 PREN 等模型。
/// </summary>
public sealed class EfficientNetB3 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public EfficientNetB3(int inChannels = 3) : base(nameof(EfficientNetB3))
    {
        OutChannels = 1536; // EfficientNet-B3 的最终通道数

        _features = Sequential(
            // Stem
            Conv2d(inChannels, 32, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(32),
            ReLU(),
            // MBConv blocks (简化版)
            MBConv(32, 24, stride: 1, expandRatio: 1),
            MBConv(24, 24, stride: 1, expandRatio: 6),
            MBConv(24, 40, stride: 2, expandRatio: 6),
            MBConv(40, 40, stride: 1, expandRatio: 6),
            MBConv(40, 80, stride: 2, expandRatio: 6),
            MBConv(80, 80, stride: 1, expandRatio: 6),
            MBConv(80, 112, stride: 1, expandRatio: 6),
            MBConv(112, 112, stride: 1, expandRatio: 6),
            MBConv(112, 192, stride: 2, expandRatio: 6),
            MBConv(192, 192, stride: 1, expandRatio: 6),
            MBConv(192, 320, stride: 1, expandRatio: 6),
            MBConv(320, 320, stride: 1, expandRatio: 6),
            // Final conv
            Conv2d(320, OutChannels, 1, bias: false),
            BatchNorm2d(OutChannels),
            ReLU()
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }

    private static Module<Tensor, Tensor> MBConv(int inCh, int outCh, int stride, int expandRatio)
    {
        var expandedCh = inCh * expandRatio;
        return Sequential(
            // Expand
            Conv2d(inCh, expandedCh, 1, bias: false),
            BatchNorm2d(expandedCh),
            ReLU(),
            // Depthwise
            Conv2d(expandedCh, expandedCh, 3, stride: stride, padding: 1, groups: expandedCh, bias: false),
            BatchNorm2d(expandedCh),
            ReLU(),
            // SE (简化)
            SEBlock(expandedCh),
            // Project
            Conv2d(expandedCh, outCh, 1, bias: false),
            BatchNorm2d(outCh)
        );
    }

    private static Module<Tensor, Tensor> SEBlock(int channels, int reduction = 4)
    {
        var reducedCh = Math.Max(1, channels / reduction);
        return Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(1),
            Linear(channels, reducedCh),
            ReLU(),
            Linear(reducedCh, channels),
            Sigmoid()
        );
    }
}
