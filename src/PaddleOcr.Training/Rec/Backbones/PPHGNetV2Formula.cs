using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// PPHGNetV2_B4_Formula backbone：Formula recognition variant of PPHGNetV2。
/// 参考: ppocr/modeling/backbones/rec_pphgnetv2.py - PPHGNetV2_B4_Formula
/// </summary>
public sealed class PPHGNetV2B4Formula : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly PPHGNetV2B4 _inner;
    public int OutChannels => _inner.OutChannels;

    public PPHGNetV2B4Formula(int inChannels = 3) : base(nameof(PPHGNetV2B4Formula))
    {
        _inner = new PPHGNetV2B4(inChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => _inner.call(input);
}

/// <summary>
/// PPHGNetV2_B6_Formula backbone：Larger formula recognition variant。
/// 参考: ppocr/modeling/backbones/rec_pphgnetv2.py - PPHGNetV2_B6_Formula
/// Uses wider channels than B4.
/// </summary>
public sealed class PPHGNetV2B6Formula : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public PPHGNetV2B6Formula(int inChannels = 3) : base(nameof(PPHGNetV2B6Formula))
    {
        // B6 uses wider channels: roughly 1.5x of B4
        OutChannels = 2048;
        _features = Sequential(
            Conv2d(inChannels, 64, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 64, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 128, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 256, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 512, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(512, 1024, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(1024),
            ReLU(),
            Conv2d(1024, 2048, 1, bias: false),
            BatchNorm2d(2048),
            ReLU()
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }
}
