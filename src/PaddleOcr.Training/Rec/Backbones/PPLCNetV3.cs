using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// PPLCNetV3 backbone：PaddlePaddle 轻量级 CNN v3。
/// </summary>
public sealed class PPLCNetV3 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public PPLCNetV3(int inChannels = 3) : base(nameof(PPLCNetV3))
    {
        OutChannels = 512;

        _features = Sequential(
            // Stem
            Conv2d(inChannels, 32, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(32),
            ReLU(),
            // Depthwise separable blocks
            DepthwiseSeparable(32, 64, stride: 1),
            DepthwiseSeparable(64, 128, stride: 2),
            DepthwiseSeparable(128, 128, stride: 1),
            DepthwiseSeparable(128, 256, stride: 2),
            DepthwiseSeparable(256, 256, stride: 1),
            DepthwiseSeparable(256, 512, stride: 2),
            DepthwiseSeparable(512, 512, stride: 1),
            AdaptiveAvgPool2d(new long[] { 1, 40 })
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }

    private static Module<Tensor, Tensor> DepthwiseSeparable(int inCh, int outCh, int stride)
    {
        return Sequential(
            // Depthwise
            Conv2d(inCh, inCh, 3, stride: stride, padding: 1, groups: inCh, bias: false),
            BatchNorm2d(inCh),
            ReLU(),
            // Pointwise
            Conv2d(inCh, outCh, 1, bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
    }
}
