using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// MobileNetV3 for Rec：CRNN 默认 backbone。
/// 支持 large/small 两种模式，含 h-swish/h-sigmoid 和 SE 模块。
/// 参考: ppocr/modeling/backbones/rec_mobilenet_v3.py
/// </summary>
public sealed class MobileNetV3 : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _conv2;
    private readonly MaxPool2d _pool;

    public int OutChannels { get; }

    public MobileNetV3(
        int inChannels = 3,
        string modelName = "small",
        float scale = 0.5f,
        int[]? largeStride = null,
        int[]? smallStride = null,
        bool disableSe = false) : base(nameof(MobileNetV3))
    {
        largeStride ??= [1, 2, 2, 2];
        smallStride ??= [2, 2, 2, 2];

        // cfg: kernel, exp, channels, useSe, activation, stride
        (int k, int exp, int c, bool se, string act, (int sh, int sw))[] cfg;
        int clsChSqueeze;

        if (modelName == "large")
        {
            cfg =
            [
                (3, 16, 16, false, "relu", (largeStride[0], 1)),
                (3, 64, 24, false, "relu", (largeStride[1], 1)),
                (3, 72, 24, false, "relu", (1, 1)),
                (5, 72, 40, true, "relu", (largeStride[2], 1)),
                (5, 120, 40, true, "relu", (1, 1)),
                (5, 120, 40, true, "relu", (1, 1)),
                (3, 240, 80, false, "hardswish", (1, 1)),
                (3, 200, 80, false, "hardswish", (1, 1)),
                (3, 184, 80, false, "hardswish", (1, 1)),
                (3, 184, 80, false, "hardswish", (1, 1)),
                (3, 480, 112, true, "hardswish", (1, 1)),
                (3, 672, 112, true, "hardswish", (1, 1)),
                (5, 672, 160, true, "hardswish", (largeStride[3], 1)),
                (5, 960, 160, true, "hardswish", (1, 1)),
                (5, 960, 160, true, "hardswish", (1, 1))
            ];
            clsChSqueeze = 960;
        }
        else // small
        {
            cfg =
            [
                (3, 16, 16, true, "relu", (smallStride[0], 1)),
                (3, 72, 24, false, "relu", (smallStride[1], 1)),
                (3, 88, 24, false, "relu", (1, 1)),
                (5, 96, 40, true, "hardswish", (smallStride[2], 1)),
                (5, 240, 40, true, "hardswish", (1, 1)),
                (5, 240, 40, true, "hardswish", (1, 1)),
                (5, 120, 48, true, "hardswish", (1, 1)),
                (5, 144, 48, true, "hardswish", (1, 1)),
                (5, 288, 96, true, "hardswish", (smallStride[3], 1)),
                (5, 576, 96, true, "hardswish", (1, 1)),
                (5, 576, 96, true, "hardswish", (1, 1))
            ];
            clsChSqueeze = 576;
        }

        var inplanes = MakeDivisible((int)(16 * scale));
        _conv1 = MakeConvBnAct(inChannels, inplanes, 3, (2, 1), 1, 1, "hardswish");

        _blocks = new ModuleList<Module<Tensor, Tensor>>();
        foreach (var (k, exp, c, se, act, s) in cfg)
        {
            var useSe = se && !disableSe;
            var midCh = MakeDivisible((int)(scale * exp));
            var outCh = MakeDivisible((int)(scale * c));
            _blocks.Add(new ResidualUnit(inplanes, midCh, outCh, k, s, useSe, act));
            inplanes = outCh;
        }

        var conv2OutCh = MakeDivisible((int)(scale * clsChSqueeze));
        _conv2 = MakeConvBnAct(inplanes, conv2OutCh, 1, (1, 1), 0, 1, "hardswish");
        _pool = MaxPool2d(kernel_size: 2, stride: 2);

        OutChannels = conv2OutCh;
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _conv1.call(x);
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }
        x = _conv2.call(x);
        x = _pool.call(x);
        return x;
    }

    private static int MakeDivisible(int v, int divisor = 8)
    {
        var minValue = divisor;
        var newV = Math.Max(minValue, (v + divisor / 2) / divisor * divisor);
        if (newV < (int)(0.9 * v))
        {
            newV += divisor;
        }
        return newV;
    }

    private static Module<Tensor, Tensor> MakeConvBnAct(int inCh, int outCh, int kernel, (int h, int w) stride, int padding, int groups, string act)
    {
        var layers = new List<Module<Tensor, Tensor>>
        {
            Conv2d(inCh, outCh, (kernel, kernel), stride: (stride.h, stride.w), padding: (padding, padding), groups: groups, bias: false),
            BatchNorm2d(outCh)
        };

        if (act == "relu")
        {
            layers.Add(ReLU(inplace: true));
        }
        else if (act == "hardswish")
        {
            layers.Add(Hardswish(inplace: true));
        }

        return Sequential(layers);
    }

    /// <summary>
    /// SE (Squeeze-and-Excitation) 模块。
    /// </summary>
    private sealed class SEBlock : Module<Tensor, Tensor>
    {
        private readonly AdaptiveAvgPool2d _pool;
        private readonly Conv2d _conv1;
        private readonly Conv2d _conv2;

        public SEBlock(int channels, int reduction = 4) : base(nameof(SEBlock))
        {
            _pool = AdaptiveAvgPool2d(1);
            _conv1 = Conv2d(channels, channels / reduction, 1);
            _conv2 = Conv2d(channels / reduction, channels, 1);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var scale = _pool.call(x);
            using var s1 = _conv1.call(scale);
            using var s2 = functional.relu(s1);
            using var s3 = _conv2.call(s2);
            using var s4 = functional.hardsigmoid(s3);
            return x * s4;
        }
    }

    /// <summary>
    /// MobileNetV3 ResidualUnit：expand conv + depthwise conv + SE + pointwise conv + residual。
    /// </summary>
    private sealed class ResidualUnit : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _expandConv;
        private readonly Module<Tensor, Tensor> _bottleneckConv;
        private readonly SEBlock? _se;
        private readonly Module<Tensor, Tensor> _linearConv;
        private readonly bool _useShortcut;

        public ResidualUnit(int inCh, int midCh, int outCh, int kernel, (int h, int w) stride, bool useSe, string act)
            : base(nameof(ResidualUnit))
        {
            _useShortcut = stride.h == 1 && stride.w == 1 && inCh == outCh;

            var padding = (kernel - 1) / 2;
            _expandConv = MakeConvBnAct(inCh, midCh, 1, (1, 1), 0, 1, act);
            _bottleneckConv = MakeConvBnAct(midCh, midCh, kernel, stride, padding, midCh, act);
            _se = useSe ? new SEBlock(midCh) : null;
            _linearConv = MakeConvBnAct(midCh, outCh, 1, (1, 1), 0, 1, "none");
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _expandConv.call(input);
            x = _bottleneckConv.call(x);
            if (_se is not null)
            {
                x = _se.call(x);
            }
            x = _linearConv.call(x);
            if (_useShortcut)
            {
                x = x + input;
            }
            return x;
        }
    }
}
