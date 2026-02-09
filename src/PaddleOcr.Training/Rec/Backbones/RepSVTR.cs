using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// RepSVTR backbone (PP-OCRv4)：基于 RepViT 的识别骨干网络。
/// 参考: ppocr/modeling/backbones/rec_repvit.py - RepSVTR
/// </summary>
public sealed class RepSVTR : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _features;
    public int OutChannels { get; }

    public RepSVTR(int inChannels = 3) : base(nameof(RepSVTR))
    {
        // k, t, c, SE, HS, s
        var cfgs = new (int k, int t, int c, bool useSe, bool useHs, (int, int) s)[]
        {
            (3, 2, 96, true, false, (1, 1)),
            (3, 2, 96, false, false, (1, 1)),
            (3, 2, 96, false, false, (1, 1)),
            (3, 2, 192, false, true, (2, 1)),
            (3, 2, 192, true, true, (1, 1)),
            (3, 2, 192, false, true, (1, 1)),
            (3, 2, 192, true, true, (1, 1)),
            (3, 2, 192, false, true, (1, 1)),
            (3, 2, 192, true, true, (1, 1)),
            (3, 2, 192, false, true, (1, 1)),
            (3, 2, 384, false, true, (2, 1)),
            (3, 2, 384, true, true, (1, 1)),
            (3, 2, 384, false, true, (1, 1)),
        };

        var inputChannel = cfgs[0].c;
        _features = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();

        // Patch embed: 2 conv layers with stride 2
        _features.Add(Sequential(
            Conv2d(inChannels, inputChannel / 2, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(inputChannel / 2),
            GELU(),
            Conv2d(inputChannel / 2, inputChannel, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(inputChannel),
            GELU()
        ));

        foreach (var (k, t, c, useSe, _, s) in cfgs)
        {
            var outputChannel = MakeDivisible(c, 8);
            var expSize = MakeDivisible(inputChannel * t, 8);
            _features.Add(new RepViTBlock(inputChannel, expSize, outputChannel, k, s, useSe));
            inputChannel = outputChannel;
        }

        OutChannels = cfgs[^1].c;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input;
        foreach (var f in _features)
        {
            x = f.call(x);
        }
        // avg_pool2d([h, 2]) for rec
        var h = x.shape[2];
        x = functional.avg_pool2d(x, new long[] { h, 2 });
        return x;
    }

    private static int MakeDivisible(int v, int divisor, int? minValue = null)
    {
        var min = minValue ?? divisor;
        var newV = Math.Max(min, (v + divisor / 2) / divisor * divisor);
        if (newV < 0.9 * v) newV += divisor;
        return newV;
    }
}

internal sealed class RepViTBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _tokenMixer;
    private readonly Module<Tensor, Tensor> _channelMixer;

    public RepViTBlock(int inp, int hiddenDim, int oup, int kernelSize, (int, int) stride, bool useSe) : base(nameof(RepViTBlock))
    {
        if (stride != (1, 1))
        {
            var seModule = useSe
                ? (Module<Tensor, Tensor>)new RepViTSEModule(inp)
                : new IdentityModule();

            _tokenMixer = Sequential(
                ("dw", Conv2dBN(inp, inp, kernelSize, stride, (kernelSize - 1) / 2, groups: inp)),
                ("se", seModule),
                ("pw", Conv2dBN(inp, oup, 1, (1, 1), 0))
            );
            _channelMixer = new ResidualModule(Sequential(
                Conv2dBN(oup, 2 * oup, 1, (1, 1), 0),
                GELU(),
                Conv2dBN(2 * oup, oup, 1, (1, 1), 0, bnWeightInit: 0)
            ));
        }
        else
        {
            var seModule = useSe
                ? (Module<Tensor, Tensor>)new RepViTSEModule(inp)
                : new IdentityModule();

            _tokenMixer = Sequential(
                ("dw", new RepVGGDW(inp)),
                ("se", seModule)
            );
            _channelMixer = new ResidualModule(Sequential(
                Conv2dBN(inp, hiddenDim, 1, (1, 1), 0),
                GELU(),
                Conv2dBN(hiddenDim, oup, 1, (1, 1), 0, bnWeightInit: 0)
            ));
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _channelMixer.call(_tokenMixer.call(input));
    }

    private static Module<Tensor, Tensor> Conv2dBN(int a, int b, int ks, (int, int) stride, int pad, int groups = 1, int bnWeightInit = 1)
    {
        var conv = Conv2d(a, b, ((long)ks, (long)ks), stride: ((long)stride.Item1, (long)stride.Item2), padding: ((long)pad, (long)pad), groups: groups, bias: false);
        var bn = BatchNorm2d(b);
        return Sequential(conv, bn);
    }
}

internal sealed class RepViTSEModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fc1;
    private readonly Module<Tensor, Tensor> _fc2;
    private readonly Module<Tensor, Tensor> _act;

    public RepViTSEModule(int channels, float rdRatio = 0.25f) : base(nameof(RepViTSEModule))
    {
        var rdChannels = Math.Max(8, (int)(channels * rdRatio) / 8 * 8);
        _fc1 = Conv2d(channels, rdChannels, 1, bias: true);
        _act = ReLU();
        _fc2 = Conv2d(rdChannels, channels, 1, bias: true);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var xSe = input.mean(new long[] { 2, 3 }, keepdim: true);
        using var fc1 = _fc1.call(xSe);
        using var act = _act.call(fc1);
        using var fc2 = _fc2.call(act);
        using var sig = functional.sigmoid(fc2);
        return input * sig;
    }
}

internal sealed class RepVGGDW : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv;
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly Module<Tensor, Tensor> _bn;

    public RepVGGDW(int ed) : base(nameof(RepVGGDW))
    {
        _conv = Sequential(
            Conv2d(ed, ed, 3, stride: 1, padding: 1, groups: ed, bias: false),
            BatchNorm2d(ed)
        );
        _conv1 = Conv2d(ed, ed, 1, stride: 1, padding: 0L, groups: ed);
        _bn = BatchNorm2d(ed);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var c1 = _conv.call(input);
        using var c2 = _conv1.call(input);
        using var sum = c1 + c2 + input;
        return _bn.call(sum);
    }
}

internal sealed class ResidualModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _m;

    public ResidualModule(Module<Tensor, Tensor> m) : base(nameof(ResidualModule))
    {
        _m = m;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return input + _m.call(input);
    }
}

internal sealed class IdentityModule : Module<Tensor, Tensor>
{
    public IdentityModule() : base(nameof(IdentityModule)) { }
    public override Tensor forward(Tensor input) => input;
}
