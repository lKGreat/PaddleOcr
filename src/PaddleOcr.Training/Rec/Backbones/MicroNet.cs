using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// MicroNet backbone：超轻量级网络。
/// </summary>
public sealed class MicroNet : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public MicroNet(int inChannels = 3) : base(nameof(MicroNet))
    {
        OutChannels = 256;

        _features = Sequential(
            Conv2d(inChannels, 32, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(32),
            ReLU(),
            Conv2d(32, 64, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 128, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 256, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(256),
            ReLU(),
            AdaptiveAvgPool2d(new long[] { 1, 40 })
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }
}
