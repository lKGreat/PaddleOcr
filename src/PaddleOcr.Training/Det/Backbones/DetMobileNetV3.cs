using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Backbones;

/// <summary>
/// MobileNetV3 for Detection：输出多尺度特征图列表 [c2, c3, c4, c5]。
/// 与 Rec 版本不同：stride 为对称 (s,s) 用于检测，且输出为多阶段特征列表。
/// 参考: ppocr/modeling/backbones/det_mobilenet_v3.py
/// </summary>
public sealed class DetMobileNetV3 : Module<Tensor, Tensor[]>, IDetBackbone
{
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly ModuleList<Sequential> _stages;

    public int[] OutChannels { get; }

    public DetMobileNetV3(
        int inChannels = 3,
        string modelName = "large",
        float scale = 0.5f,
        bool disableSe = false) : base(nameof(DetMobileNetV3))
    {
        // cfg: kernel, exp, channels, useSe, activation, stride
        (int k, int exp, int c, bool se, string act, int s)[] cfg;
        int clsChSqueeze;

        if (modelName == "large")
        {
            cfg =
            [
                (3, 16, 16, false, "relu", 1),
                (3, 64, 24, false, "relu", 2),
                (3, 72, 24, false, "relu", 1),
                (5, 72, 40, true, "relu", 2),
                (5, 120, 40, true, "relu", 1),
                (5, 120, 40, true, "relu", 1),
                (3, 240, 80, false, "hardswish", 2),
                (3, 200, 80, false, "hardswish", 1),
                (3, 184, 80, false, "hardswish", 1),
                (3, 184, 80, false, "hardswish", 1),
                (3, 480, 112, true, "hardswish", 1),
                (3, 672, 112, true, "hardswish", 1),
                (5, 672, 160, true, "hardswish", 2),
                (5, 960, 160, true, "hardswish", 1),
                (5, 960, 160, true, "hardswish", 1)
            ];
            clsChSqueeze = 960;
        }
        else // small
        {
            cfg =
            [
                (3, 16, 16, true, "relu", 2),
                (3, 72, 24, false, "relu", 2),
                (3, 88, 24, false, "relu", 1),
                (5, 96, 40, true, "hardswish", 2),
                (5, 240, 40, true, "hardswish", 1),
                (5, 240, 40, true, "hardswish", 1),
                (5, 120, 48, true, "hardswish", 1),
                (5, 144, 48, true, "hardswish", 1),
                (5, 288, 96, true, "hardswish", 2),
                (5, 576, 96, true, "hardswish", 1),
                (5, 576, 96, true, "hardswish", 1)
            ];
            clsChSqueeze = 576;
        }

        var inplanes = MakeDivisible((int)(16 * scale));
        _conv1 = MakeConvBnAct(inChannels, inplanes, 3, 2, 1, 1, "hardswish");

        // 按 stride==2 分拆阶段，收集各阶段输出通道
        _stages = new ModuleList<Sequential>();
        var stageOutChannels = new List<int>();
        var blockList = new List<Module<Tensor, Tensor>>();
        int startIdx = modelName == "large" ? 2 : 0;
        int i = 0;

        foreach (var (k, exp, c, se, act, s) in cfg)
        {
            var useSe = se && !disableSe;
            var midCh = MakeDivisible((int)(scale * exp));
            var outCh = MakeDivisible((int)(scale * c));

            if (s == 2 && i > startIdx)
            {
                // 当前阶段结束，保存
                stageOutChannels.Add(inplanes);
                _stages.Add(Sequential(blockList.ToArray()));
                blockList.Clear();
            }

            blockList.Add(new DetResidualUnit(inplanes, midCh, outCh, k, s, useSe, act));
            inplanes = outCh;
            i++;
        }

        // 最后一个阶段加上 1x1 conv
        var conv2OutCh = MakeDivisible((int)(scale * clsChSqueeze));
        blockList.Add(MakeConvBnAct(inplanes, conv2OutCh, 1, 1, 0, 1, "hardswish"));
        _stages.Add(Sequential(blockList.ToArray()));
        stageOutChannels.Add(conv2OutCh);

        OutChannels = stageOutChannels.ToArray();
        RegisterComponents();
    }

    public override Tensor[] forward(Tensor x)
    {
        x = ((Module<Tensor, Tensor>)_conv1).call(x);
        var outputs = new Tensor[_stages.Count];
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].call(x);
            outputs[i] = x;
        }
        return outputs;
    }

    private static int MakeDivisible(int v, int divisor = 8)
    {
        var newV = Math.Max(divisor, (v + divisor / 2) / divisor * divisor);
        if (newV < (int)(0.9 * v)) newV += divisor;
        return newV;
    }

    private static Module<Tensor, Tensor> MakeConvBnAct(
        int inCh, int outCh, int kernel, int stride, int padding, int groups, string act)
    {
        var conv = Conv2d(inCh, outCh, kernel, stride: stride, padding: padding, groups: groups, bias: false);
        var bn = BatchNorm2d(outCh);
        return act switch
        {
            "relu" => Sequential(conv, bn, ReLU()),
            "hardswish" => Sequential(conv, bn, Hardswish()),
            _ => Sequential(conv, bn)
        };
    }
}

/// <summary>
/// MobileNetV3 InvertedResidual block for detection.
/// </summary>
internal sealed class DetResidualUnit : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _expand;
    private readonly Module<Tensor, Tensor> _depthwise;
    private readonly DetSEModule? _se;
    private readonly Module<Tensor, Tensor> _linear;
    private readonly bool _useShortcut;

    public DetResidualUnit(
        int inChannels, int midChannels, int outChannels,
        int kernelSize, int stride, bool useSe, string act)
        : base(nameof(DetResidualUnit))
    {
        _useShortcut = stride == 1 && inChannels == outChannels;

        _expand = MakeConvBnAct(inChannels, midChannels, 1, 1, 0, 1, act);
        _depthwise = MakeConvBnAct(midChannels, midChannels, kernelSize, stride, kernelSize / 2, midChannels, act);
        _se = useSe ? new DetSEModule(midChannels) : null;
        _linear = MakeConvBnAct(midChannels, outChannels, 1, 1, 0, 1, "none");

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = ((Module<Tensor, Tensor>)_expand).call(input);
        x = ((Module<Tensor, Tensor>)_depthwise).call(x);
        if (_se is not null)
        {
            x = _se.call(x);
        }
        x = ((Module<Tensor, Tensor>)_linear).call(x);
        if (_useShortcut)
        {
            x = x + input;
        }
        return x;
    }

    private static Module<Tensor, Tensor> MakeConvBnAct(
        int inCh, int outCh, int kernel, int stride, int padding, int groups, string act)
    {
        var conv = Conv2d(inCh, outCh, kernel, stride: stride, padding: padding, groups: groups, bias: false);
        var bn = BatchNorm2d(outCh);
        return act switch
        {
            "relu" => Sequential(conv, bn, ReLU()),
            "hardswish" => Sequential(conv, bn, Hardswish()),
            _ => Sequential(conv, bn)
        };
    }
}

/// <summary>
/// Squeeze-and-Excitation module for Det MobileNetV3.
/// </summary>
internal sealed class DetSEModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _avgPool;
    private readonly Conv2d _conv1;
    private readonly Conv2d _conv2;

    public DetSEModule(int channels, int reduction = 4) : base(nameof(DetSEModule))
    {
        _avgPool = AdaptiveAvgPool2d(1);
        _conv1 = Conv2d(channels, channels / reduction, 1);
        _conv2 = Conv2d(channels / reduction, channels, 1);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var avg = ((Module<Tensor, Tensor>)_avgPool).call(input);
        using var fc1 = functional.relu(_conv1.call(avg));
        using var fc2 = functional.hardsigmoid(_conv2.call(fc1));
        return input * fc2;
    }
}
