using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Necks;

/// <summary>
/// DBFPN (Feature Pyramid Network for DB text detection).
/// 接收 4 个阶段的特征图 [c2,c3,c4,c5]，通过 top-down 路径融合后 concat 输出。
/// 参考: ppocr/modeling/necks/db_fpn.py
/// </summary>
public sealed class DBFPN : Module<Tensor[], Tensor>, IDetNeck
{
    private readonly Conv2d _in2Conv;
    private readonly Conv2d _in3Conv;
    private readonly Conv2d _in4Conv;
    private readonly Conv2d _in5Conv;
    private readonly Conv2d _p2Conv;
    private readonly Conv2d _p3Conv;
    private readonly Conv2d _p4Conv;
    private readonly Conv2d _p5Conv;

    public int OutChannels { get; }

    public DBFPN(int[] inChannels, int outChannels = 256) : base(nameof(DBFPN))
    {
        if (inChannels.Length != 4)
            throw new ArgumentException("DBFPN expects 4 input channel sizes", nameof(inChannels));

        OutChannels = outChannels;

        // 1x1 lateral connections
        _in2Conv = Conv2d(inChannels[0], outChannels, 1, bias: false);
        _in3Conv = Conv2d(inChannels[1], outChannels, 1, bias: false);
        _in4Conv = Conv2d(inChannels[2], outChannels, 1, bias: false);
        _in5Conv = Conv2d(inChannels[3], outChannels, 1, bias: false);

        // 3x3 smooth convolutions (out_channels / 4 each, then concat = out_channels)
        _p2Conv = Conv2d(outChannels, outChannels / 4, 3, padding: 1, bias: false);
        _p3Conv = Conv2d(outChannels, outChannels / 4, 3, padding: 1, bias: false);
        _p4Conv = Conv2d(outChannels, outChannels / 4, 3, padding: 1, bias: false);
        _p5Conv = Conv2d(outChannels, outChannels / 4, 3, padding: 1, bias: false);

        RegisterComponents();
    }

    public override Tensor forward(Tensor[] x)
    {
        var c2 = x[0]; // 1/4
        var c3 = x[1]; // 1/8
        var c4 = x[2]; // 1/16
        var c5 = x[3]; // 1/32

        // Lateral connections
        var in5 = _in5Conv.call(c5);
        var in4 = _in4Conv.call(c4);
        var in3 = _in3Conv.call(c3);
        var in2 = _in2Conv.call(c2);

        // Top-down pathway
        var out4 = in4 + functional.interpolate(in5, scale_factor: [2, 2], mode: InterpolationMode.Nearest);
        var out3 = in3 + functional.interpolate(out4, scale_factor: [2, 2], mode: InterpolationMode.Nearest);
        var out2 = in2 + functional.interpolate(out3, scale_factor: [2, 2], mode: InterpolationMode.Nearest);

        // Smooth convolutions
        var p5 = _p5Conv.call(in5);
        var p4 = _p4Conv.call(out4);
        var p3 = _p3Conv.call(out3);
        var p2 = _p2Conv.call(out2);

        // Upsample all to the same size (1/4 scale)
        p5 = functional.interpolate(p5, scale_factor: [8, 8], mode: InterpolationMode.Nearest);
        p4 = functional.interpolate(p4, scale_factor: [4, 4], mode: InterpolationMode.Nearest);
        p3 = functional.interpolate(p3, scale_factor: [2, 2], mode: InterpolationMode.Nearest);

        // Concat → out_channels
        var fuse = torch.cat([p5, p4, p3, p2], dim: 1);
        return fuse;
    }
}

/// <summary>
/// RSEFPN — FPN with Residual Squeeze-Excitation (used in DB++ / PP-OCRv3 det).
/// 参考: ppocr/modeling/necks/db_fpn.py RSELayer/RSEFPN
/// </summary>
public sealed class RSEFPN : Module<Tensor[], Tensor>, IDetNeck
{
    private readonly ModuleList<RSELayer> _insConv;
    private readonly ModuleList<RSELayer> _inpConv;

    public int OutChannels { get; }

    public RSEFPN(int[] inChannels, int outChannels = 96, bool shortcut = true) : base(nameof(RSEFPN))
    {
        if (inChannels.Length != 4)
            throw new ArgumentException("RSEFPN expects 4 input channel sizes");

        OutChannels = outChannels;
        _insConv = new ModuleList<RSELayer>();
        _inpConv = new ModuleList<RSELayer>();

        for (int i = 0; i < 4; i++)
        {
            _insConv.Add(new RSELayer(inChannels[i], outChannels, 1, shortcut));
            _inpConv.Add(new RSELayer(outChannels, outChannels / 4, 3, shortcut));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor[] x)
    {
        var c2 = x[0]; var c3 = x[1]; var c4 = x[2]; var c5 = x[3];

        var in5 = _insConv[3].call(c5);
        var in4 = _insConv[2].call(c4);
        var in3 = _insConv[1].call(c3);
        var in2 = _insConv[0].call(c2);

        var out4 = in4 + functional.interpolate(in5, scale_factor: [2, 2], mode: InterpolationMode.Nearest);
        var out3 = in3 + functional.interpolate(out4, scale_factor: [2, 2], mode: InterpolationMode.Nearest);
        var out2 = in2 + functional.interpolate(out3, scale_factor: [2, 2], mode: InterpolationMode.Nearest);

        var p5 = _inpConv[3].call(in5);
        var p4 = _inpConv[2].call(out4);
        var p3 = _inpConv[1].call(out3);
        var p2 = _inpConv[0].call(out2);

        p5 = functional.interpolate(p5, scale_factor: [8, 8], mode: InterpolationMode.Nearest);
        p4 = functional.interpolate(p4, scale_factor: [4, 4], mode: InterpolationMode.Nearest);
        p3 = functional.interpolate(p3, scale_factor: [2, 2], mode: InterpolationMode.Nearest);

        return torch.cat([p5, p4, p3, p2], dim: 1);
    }
}

/// <summary>
/// RSE (Residual Squeeze-Excitation) layer for RSEFPN.
/// </summary>
internal sealed class RSELayer : Module<Tensor, Tensor>
{
    private readonly Conv2d _inConv;
    private readonly BatchNorm2d _bn;
    private readonly Module<Tensor, Tensor> _seAvgPool;
    private readonly Conv2d _se1;
    private readonly Conv2d _se2;
    private readonly bool _shortcut;

    public RSELayer(int inChannels, int outChannels, int kernelSize, bool shortcut)
        : base(nameof(RSELayer))
    {
        _inConv = Conv2d(inChannels, outChannels, kernelSize, padding: kernelSize / 2, bias: false);
        _bn = BatchNorm2d(outChannels);
        _seAvgPool = AdaptiveAvgPool2d(1);
        _se1 = Conv2d(outChannels, outChannels / 4, 1);
        _se2 = Conv2d(outChannels / 4, outChannels, 1);
        _shortcut = shortcut;

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _bn.call(_inConv.call(input));
        // SE block
        using var avg = ((Module<Tensor, Tensor>)_seAvgPool).call(x);
        using var se1 = functional.relu(_se1.call(avg));
        using var se2 = torch.sigmoid(_se2.call(se1));
        var seOut = x * se2;

        if (_shortcut)
        {
            return x + seOut;
        }
        return seOut;
    }
}
