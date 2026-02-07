using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// PPHGNetV2_B4 backbone for rec (text_rec=True path).
/// This follows rec_pphgnetv2.py stage topology and output shape contract [B, C, 1, 40].
/// </summary>
public sealed class PPHGNetV2B4 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly StemBlockV2 _stem;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _stages;

    public int OutChannels { get; } = 2048;

    public PPHGNetV2B4(int inChannels = 3) : base(nameof(PPHGNetV2B4))
    {
        _stem = new StemBlockV2(inChannels, midChannels: 32, outChannels: 48, textRec: true);
        _stages = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>
        {
            new HGV2Stage(48, 48, 128, blockNum: 1, layerNum: 6, downsampleStride: (2, 1), lightBlock: false, kernelSize: 3),
            new HGV2Stage(128, 96, 512, blockNum: 1, layerNum: 6, downsampleStride: (1, 2), lightBlock: false, kernelSize: 3),
            new HGV2Stage(512, 192, 1024, blockNum: 3, layerNum: 6, downsampleStride: (2, 1), lightBlock: true, kernelSize: 5),
            new HGV2Stage(1024, 384, 2048, blockNum: 1, layerNum: 6, downsampleStride: (2, 1), lightBlock: true, kernelSize: 5)
        };
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _stem.call(input);
        foreach (var stage in _stages)
        {
            x = stage.call(x);
        }

        return functional.adaptive_avg_pool2d(x, new long[] { 1, 40 });
    }
}

internal sealed class StemBlockV2 : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _stem1;
    private readonly Module<Tensor, Tensor> _stem2a;
    private readonly Module<Tensor, Tensor> _stem2b;
    private readonly Module<Tensor, Tensor> _stem3;
    private readonly Module<Tensor, Tensor> _stem4;
    private readonly Module<Tensor, Tensor> _pool;

    public StemBlockV2(int inChannels, int midChannels, int outChannels, bool textRec) : base(nameof(StemBlockV2))
    {
        _stem1 = ConvBnAct(inChannels, midChannels, 3, (2, 2), useAct: true);
        _stem2a = ConvBnAct(midChannels, midChannels / 2, 3, (1, 1), useAct: true);
        _stem2b = ConvBnAct(midChannels / 2, midChannels, 3, (1, 1), useAct: true);
        _stem3 = ConvBnAct(midChannels * 2, midChannels, 3, textRec ? (1, 1) : (2, 2), useAct: true);
        _stem4 = ConvBnAct(midChannels, outChannels, 1, (1, 1), useAct: true);
        _pool = MaxPool2d(3, stride: 1, padding: 1);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var x = _stem1.call(input);
        using var x2a = _stem2a.call(x);
        using var x2 = _stem2b.call(x2a);
        using var x1 = _pool.call(x);
        using var cat = torch.cat([x1, x2], 1);
        using var x3 = _stem3.call(cat);
        return _stem4.call(x3);
    }

    private static Module<Tensor, Tensor> ConvBnAct(
        int inChannels,
        int outChannels,
        int kernelSize,
        (long Y, long X) stride,
        bool useAct = true,
        int groups = 1)
    {
        if (useAct)
        {
            return Sequential(
                Conv2d(inChannels, outChannels, (kernelSize, kernelSize), stride: (stride.Y, stride.X), padding: ((kernelSize - 1) / 2, (kernelSize - 1) / 2), groups: groups, bias: false),
                BatchNorm2d(outChannels),
                ReLU());
        }

        return Sequential(
            Conv2d(inChannels, outChannels, (kernelSize, kernelSize), stride: (stride.Y, stride.X), padding: ((kernelSize - 1) / 2, (kernelSize - 1) / 2), groups: groups, bias: false),
            BatchNorm2d(outChannels));
    }
}

internal sealed class HGV2Stage : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _downsample;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;

    public HGV2Stage(
        int inChannels,
        int midChannels,
        int outChannels,
        int blockNum,
        int layerNum,
        (long Y, long X) downsampleStride,
        bool lightBlock,
        int kernelSize) : base(nameof(HGV2Stage))
    {
        _downsample = Sequential(
            Conv2d(inChannels, inChannels, (3, 3), stride: (downsampleStride.Y, downsampleStride.X), padding: (1, 1), groups: inChannels, bias: false),
            BatchNorm2d(inChannels));

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < blockNum; i++)
        {
            _blocks.Add(new HGV2Block(
                inChannels: i == 0 ? inChannels : outChannels,
                midChannels: midChannels,
                outChannels: outChannels,
                layerNum: layerNum,
                kernelSize: kernelSize,
                identity: i > 0,
                lightBlock: lightBlock));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _downsample.call(input);
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        return x;
    }
}

internal sealed class HGV2Block : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;
    private readonly Module<Tensor, Tensor> _squeeze;
    private readonly Module<Tensor, Tensor> _excitation;
    private readonly bool _identity;

    public HGV2Block(
        int inChannels,
        int midChannels,
        int outChannels,
        int layerNum,
        int kernelSize,
        bool identity,
        bool lightBlock) : base(nameof(HGV2Block))
    {
        _identity = identity;
        _layers = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < layerNum; i++)
        {
            var inCh = i == 0 ? inChannels : midChannels;
            _layers.Add(lightBlock
                ? new LightConvBnAct(inCh, midChannels, kernelSize)
                : ConvBnAct(inCh, midChannels, kernelSize, (1, 1), useAct: true));
        }

        var totalChannels = inChannels + layerNum * midChannels;
        _squeeze = ConvBnAct(totalChannels, outChannels / 2, 1, (1, 1), useAct: true);
        _excitation = ConvBnAct(outChannels / 2, outChannels, 1, (1, 1), useAct: true);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var identity = input;
        var outputs = new List<Tensor> { input };
        var x = input;
        foreach (var layer in _layers)
        {
            x = layer.call(x);
            outputs.Add(x);
        }

        using var cat = torch.cat(outputs.ToArray(), 1);
        foreach (var item in outputs.Skip(1))
        {
            item.Dispose();
        }

        using var squeezed = _squeeze.call(cat);
        var outTensor = _excitation.call(squeezed);
        if (_identity)
        {
            var withSkip = outTensor + identity;
            outTensor.Dispose();
            return withSkip;
        }

        return outTensor;
    }

    private static Module<Tensor, Tensor> ConvBnAct(
        int inChannels,
        int outChannels,
        int kernelSize,
        (long Y, long X) stride,
        bool useAct = true,
        int groups = 1)
    {
        if (useAct)
        {
            return Sequential(
                Conv2d(inChannels, outChannels, (kernelSize, kernelSize), stride: (stride.Y, stride.X), padding: ((kernelSize - 1) / 2, (kernelSize - 1) / 2), groups: groups, bias: false),
                BatchNorm2d(outChannels),
                ReLU());
        }

        return Sequential(
            Conv2d(inChannels, outChannels, (kernelSize, kernelSize), stride: (stride.Y, stride.X), padding: ((kernelSize - 1) / 2, (kernelSize - 1) / 2), groups: groups, bias: false),
            BatchNorm2d(outChannels));
    }
}

internal sealed class LightConvBnAct : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _pointwise;
    private readonly Module<Tensor, Tensor> _depthwise;

    public LightConvBnAct(int inChannels, int outChannels, int kernelSize) : base(nameof(LightConvBnAct))
    {
        _pointwise = Sequential(
            Conv2d(inChannels, outChannels, (1, 1), stride: (1, 1), padding: (0, 0), bias: false),
            BatchNorm2d(outChannels));

        _depthwise = Sequential(
            Conv2d(outChannels, outChannels, (kernelSize, kernelSize), stride: (1, 1), padding: ((kernelSize - 1) / 2, (kernelSize - 1) / 2), groups: outChannels, bias: false),
            BatchNorm2d(outChannels),
            ReLU());
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var x = _pointwise.call(input);
        return _depthwise.call(x);
    }
}
