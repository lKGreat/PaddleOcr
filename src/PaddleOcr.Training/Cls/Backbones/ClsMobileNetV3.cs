using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using PaddleOcr.Training.Det.Backbones;

namespace PaddleOcr.Training.Cls.Backbones;

/// <summary>
/// MobileNetV3 for Classification: outputs single final feature tensor.
/// Reuses DetResidualUnit and DetSEModule building blocks but with sequential architecture.
/// Reference: PaddleOCR ppocr/modeling/backbones/rec_mobilenet_v3.py
/// </summary>
public sealed class ClsMobileNetV3 : Module<Tensor, Tensor>, IClsBackbone
{
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor>? _conv2;

    public int OutChannels { get; }

    /// <summary>
    /// Creates a new ClsMobileNetV3 backbone.
    /// </summary>
    /// <param name="inChannels">Number of input channels (default: 3 for RGB)</param>
    /// <param name="modelName">Model variant: "large" or "small" (default: "small")</param>
    /// <param name="scale">Channel scaling factor (default: 1.0)</param>
    /// <param name="disableSe">Whether to disable Squeeze-and-Excitation modules (default: false)</param>
    public ClsMobileNetV3(
        int inChannels = 3,
        string modelName = "small",
        float scale = 1.0f,
        bool disableSe = false) : base(nameof(ClsMobileNetV3))
    {
        // Configuration: (kernel, exp_channels, out_channels, use_se, activation, stride)
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

        // Initial convolution
        var inplanes = MakeDivisible((int)(16 * scale));
        _conv1 = MakeConvBnAct(inChannels, inplanes, 3, 2, 1, 1, "hardswish");

        // Build all residual blocks sequentially (no stage separation)
        _blocks = new ModuleList<Module<Tensor, Tensor>>();

        foreach (var (k, exp, c, se, act, s) in cfg)
        {
            var useSe = se && !disableSe;
            var midCh = MakeDivisible((int)(scale * exp));
            var outCh = MakeDivisible((int)(scale * c));

            // Reuse DetResidualUnit from Det.Backbones
            _blocks.Add(new DetResidualUnit(inplanes, midCh, outCh, k, s, useSe, act));
            inplanes = outCh;
        }

        // Optional final 1x1 conv (for feature refinement)
        var conv2OutCh = MakeDivisible((int)(scale * clsChSqueeze));
        if (conv2OutCh != inplanes)
        {
            _conv2 = MakeConvBnAct(inplanes, conv2OutCh, 1, 1, 0, 1, "hardswish");
            OutChannels = conv2OutCh;
        }
        else
        {
            _conv2 = null;
            OutChannels = inplanes;
        }

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through the backbone.
    /// </summary>
    /// <param name="x">Input image tensor. Shape: [B, 3, H, W]</param>
    /// <returns>Feature tensor. Shape: [B, C, H', W'] where H'=H/32, W'=W/32</returns>
    public override Tensor forward(Tensor x)
    {
        // Initial convolution
        x = _conv1.call(x);

        // Apply all residual blocks sequentially
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        // Optional final convolution
        if (_conv2 is not null)
        {
            x = _conv2.call(x);
        }

        return x;
    }

    /// <summary>
    /// Makes the channel count divisible by a divisor (for hardware efficiency).
    /// </summary>
    private static int MakeDivisible(int v, int divisor = 8)
    {
        var newV = Math.Max(divisor, (v + divisor / 2) / divisor * divisor);
        if (newV < (int)(0.9 * v)) newV += divisor;
        return newV;
    }

    /// <summary>
    /// Helper to create Conv + BatchNorm + Activation block.
    /// </summary>
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
