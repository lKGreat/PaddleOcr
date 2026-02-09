using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ResNetV2 backbone：Pre-activation ResNet with Weight Standardization (BiT model)。
/// 参考: ppocr/modeling/backbones/rec_resnetv2.py
/// </summary>
public sealed class ResNetV2 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _stem;
    private readonly Module<Tensor, Tensor> _stages;
    private readonly Module<Tensor, Tensor> _norm;
    public int OutChannels { get; }

    public ResNetV2(
        int inChannels = 3,
        int[] layers = null!,
        int[] channels = null!,
        int stemChs = 64,
        float dropPathRate = 0.0f) : base(nameof(ResNetV2))
    {
        layers ??= [2, 3, 7];
        channels ??= [256, 512, 1024, 2048];

        // Stem: 7x7 conv + maxpool
        _stem = Sequential(
            Conv2d(inChannels, stemChs, 7, stride: 2, padding: 3, bias: false),
            MaxPool2d(3, stride: 2, padding: 1)
        );

        var prevChs = stemChs;
        var stageList = new List<(string, Module<Tensor, Tensor>)>();
        for (var i = 0; i < layers.Length; i++)
        {
            var outChs = channels[i];
            var stride = i == 0 ? 1 : 2;
            stageList.Add(($"stage{i}", BuildStage(prevChs, outChs, layers[i], stride)));
            prevChs = outChs;
        }
        _stages = Sequential(stageList.ToArray());

        _norm = Sequential(GroupNorm(32, prevChs), ReLU());
        OutChannels = prevChs;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _stem.call(input);
        x = _stages.call(x);
        x = _norm.call(x);
        return x;
    }

    private static Module<Tensor, Tensor> BuildStage(int inChs, int outChs, int depth, int stride)
    {
        var blocks = new List<(string, Module<Tensor, Tensor>)>();
        for (var i = 0; i < depth; i++)
        {
            var blockStride = i == 0 ? stride : 1;
            var blockInChs = i == 0 ? inChs : outChs;
            blocks.Add(($"block{i}", new PreActBottleneck(blockInChs, outChs, blockStride, i == 0)));
        }
        return Sequential(blocks.ToArray());
    }
}

internal sealed class PreActBottleneck : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly Module<Tensor, Tensor> _conv2;
    private readonly Module<Tensor, Tensor> _norm3;
    private readonly Module<Tensor, Tensor> _conv3;
    private readonly Module<Tensor, Tensor>? _downsample;

    public PreActBottleneck(int inChs, int outChs, int stride, bool hasDownsample) : base(nameof(PreActBottleneck))
    {
        var midChs = Math.Max(8, (int)(outChs * 0.25) / 8 * 8);
        _norm1 = Sequential(GroupNorm(32, inChs), ReLU());
        _conv1 = Conv2d(inChs, midChs, 1, bias: false);
        _norm2 = Sequential(GroupNorm(32, midChs), ReLU());
        _conv2 = Conv2d(midChs, midChs, 3, stride: stride, padding: 1, bias: false);
        _norm3 = Sequential(GroupNorm(32, midChs), ReLU());
        _conv3 = Conv2d(midChs, outChs, 1, bias: false);

        if (hasDownsample && (inChs != outChs || stride != 1))
        {
            _downsample = Conv2d(inChs, outChs, 1, stride: stride, bias: false);
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var xPreact = _norm1.call(input);
        var shortcut = _downsample is not null ? _downsample.call(xPreact) : input;
        var x = _conv1.call(xPreact);
        x = _conv2.call(_norm2.call(x));
        x = _conv3.call(_norm3.call(x));
        return x + shortcut;
    }
}
