using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ShallowCNN backbone：浅层 CNN，用于轻量级模型。
/// </summary>
public sealed class ShallowCNN : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public ShallowCNN(int inChannels = 3) : base(nameof(ShallowCNN))
    {
        OutChannels = 512;

        _features = Sequential(
            Conv2d(inChannels, 64, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, stride: 2),
            Conv2d(64, 128, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, stride: 2),
            Conv2d(128, 256, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 512, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(512),
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
