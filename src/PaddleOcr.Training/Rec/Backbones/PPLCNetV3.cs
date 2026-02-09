using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// PPLCNetV3 backbone for recognition (rec mode).
/// 1:1 port of ppocr/modeling/backbones/rec_lcnetv3.py PPLCNetV3(det=False).
///
/// Architecture: conv1 -> blocks2 -> blocks3 -> blocks4 -> blocks5 -> blocks6 -> pool
/// Uses LearnableRepLayer (reparameterizable convolutions) and LearnableAffineBlock.
/// Training output: adaptive_avg_pool2d([1, 40])
/// Inference output: avg_pool2d([3, 2])
/// </summary>
public sealed class PPLCNetV3 : Module<Tensor, Tensor>, IRecBackbone
{
    // NET_CONFIG_rec: k, in_c, out_c, stride_h, stride_w, use_se
    private static readonly (int K, int InC, int OutC, int SH, int SW, bool UseSe)[][] NetConfigRec =
    [
        // blocks2
        [(3, 16, 32, 1, 1, false)],
        // blocks3
        [(3, 32, 64, 1, 1, false), (3, 64, 64, 1, 1, false)],
        // blocks4
        [(3, 64, 128, 2, 1, false), (3, 128, 128, 1, 1, false)],
        // blocks5
        [(3, 128, 256, 1, 2, false), (5, 256, 256, 1, 1, false), (5, 256, 256, 1, 1, false), (5, 256, 256, 1, 1, false), (5, 256, 256, 1, 1, false)],
        // blocks6
        [(5, 256, 512, 2, 1, true), (5, 512, 512, 1, 1, true), (5, 512, 512, 2, 1, false), (5, 512, 512, 1, 1, false)],
    ];

    private readonly LCNetV3ConvBNLayer _conv1;
    private readonly Sequential _blocks2;
    private readonly Sequential _blocks3;
    private readonly Sequential _blocks4;
    private readonly Sequential _blocks5;
    private readonly Sequential _blocks6;

    public int OutChannels { get; }

    public PPLCNetV3(
        int inChannels = 3,
        float scale = 0.95f,
        int convKxkNum = 4,
        float[]? lrMultList = null,
        float labLr = 0.1f) : base(nameof(PPLCNetV3))
    {
        lrMultList ??= [1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f];
        if (lrMultList.Length != 6)
            throw new ArgumentException($"lrMultList length should be 6 but got {lrMultList.Length}");

        OutChannels = MakeDivisible((int)(512 * scale));

        _conv1 = new LCNetV3ConvBNLayer(
            inChannels, MakeDivisible((int)(16 * scale)), 3, (2, 2));

        _blocks2 = BuildBlockGroup(NetConfigRec[0], scale, convKxkNum, lrMultList[1], labLr);
        _blocks3 = BuildBlockGroup(NetConfigRec[1], scale, convKxkNum, lrMultList[2], labLr);
        _blocks4 = BuildBlockGroup(NetConfigRec[2], scale, convKxkNum, lrMultList[3], labLr);
        _blocks5 = BuildBlockGroup(NetConfigRec[3], scale, convKxkNum, lrMultList[4], labLr);
        _blocks6 = BuildBlockGroup(NetConfigRec[4], scale, convKxkNum, lrMultList[5], labLr);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _conv1.call(input);
        x = _blocks2.call(x);
        x = _blocks3.call(x);
        x = _blocks4.call(x);
        x = _blocks5.call(x);
        x = _blocks6.call(x);

        // Rec output pooling: training uses adaptive, inference uses fixed kernel
        if (training)
        {
            x = functional.adaptive_avg_pool2d(x, [1, 40]);
        }
        else
        {
            x = functional.avg_pool2d(x, [3, 2]);
        }

        return x;
    }

    private static Sequential BuildBlockGroup(
        (int K, int InC, int OutC, int SH, int SW, bool UseSe)[] blockConfigs,
        float scale, int convKxkNum, float lrMult, float labLr)
    {
        var blocks = new List<Module<Tensor, Tensor>>();
        foreach (var (k, inC, outC, sh, sw, useSe) in blockConfigs)
        {
            blocks.Add(new LCNetV3Block(
                MakeDivisible((int)(inC * scale)),
                MakeDivisible((int)(outC * scale)),
                dwSize: k,
                stride: (sh, sw),
                useSe: useSe,
                convKxkNum: convKxkNum,
                lrMult: lrMult,
                labLr: labLr));
        }
        return Sequential(blocks);
    }

    /// <summary>
    /// make_divisible with divisor=16 (different from MobileNetV3 which uses 8).
    /// </summary>
    internal static int MakeDivisible(int v, int divisor = 16)
    {
        var minValue = divisor;
        var newV = Math.Max(minValue, (v + divisor / 2) / divisor * divisor);
        if (newV < (int)(0.9 * v))
        {
            newV += divisor;
        }
        return newV;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal layer implementations (1:1 port of rec_lcnetv3.py)
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Learnable affine transformation: scale * x + bias.
/// Both scale and bias are learnable scalar parameters.
/// Reference: rec_lcnetv3.py LearnableAffineBlock
/// </summary>
internal sealed class LearnableAffineBlock : Module<Tensor, Tensor>
{
    private readonly Parameter _scale;
    private readonly Parameter _bias;

    public LearnableAffineBlock(float scaleValue = 1.0f, float biasValue = 0.0f)
        : base(nameof(LearnableAffineBlock))
    {
        _scale = Parameter(torch.tensor(scaleValue));
        _bias = Parameter(torch.tensor(biasValue));
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _scale * input + _bias;
    }
}

/// <summary>
/// Conv2D + BatchNorm2D (no activation).
/// Supports asymmetric stride via (strideH, strideW) tuple.
/// Reference: rec_lcnetv3.py ConvBNLayer
/// </summary>
internal sealed class LCNetV3ConvBNLayer : Module<Tensor, Tensor>
{
    private readonly Conv2d _conv;
    private readonly BatchNorm2d _bn;

    public LCNetV3ConvBNLayer(
        int inChannels, int outChannels, int kernelSize, (int H, int W) stride,
        int groups = 1)
        : base(nameof(LCNetV3ConvBNLayer))
    {
        var padding = (kernelSize - 1) / 2;
        _conv = Conv2d(
            inChannels, outChannels, (kernelSize, kernelSize),
            stride: ((long)stride.H, (long)stride.W),
            padding: (padding, padding),
            groups: groups,
            bias: false);
        _bn = BatchNorm2d(outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _bn.call(_conv.call(input));
    }

    /// <summary>Access the underlying Conv2d (used in rep fusion).</summary>
    internal Conv2d Conv => _conv;

    /// <summary>Access the underlying BatchNorm2d (used in rep fusion).</summary>
    internal BatchNorm2d BN => _bn;
}

/// <summary>
/// Activation wrapper: Hardswish or ReLU followed by LearnableAffineBlock.
/// Reference: rec_lcnetv3.py Act
/// </summary>
internal sealed class LCNetV3Act : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _act;
    private readonly LearnableAffineBlock _lab;

    public LCNetV3Act(string act = "hswish")
        : base(nameof(LCNetV3Act))
    {
        _act = act == "hswish" ? Hardswish() : (Module<Tensor, Tensor>)ReLU();
        _lab = new LearnableAffineBlock();
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _lab.call(_act.call(input));
    }
}

/// <summary>
/// Squeeze-and-Excitation layer for PPLCNetV3.
/// Uses Hardsigmoid (not Sigmoid like MobileNetV3 det SE).
/// Reference: rec_lcnetv3.py SELayer
/// </summary>
internal sealed class LCNetV3SELayer : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _avgPool;
    private readonly Conv2d _conv1;
    private readonly Conv2d _conv2;

    public LCNetV3SELayer(int channels, int reduction = 4)
        : base(nameof(LCNetV3SELayer))
    {
        _avgPool = AdaptiveAvgPool2d(1);
        _conv1 = Conv2d(channels, channels / reduction, 1);
        _conv2 = Conv2d(channels / reduction, channels, 1);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var avg = _avgPool.call(input);
        using var fc1 = functional.relu(_conv1.call(avg));
        using var fc2 = functional.hardsigmoid(_conv2.call(fc1));
        return input * fc2;
    }
}

/// <summary>
/// Reparameterizable convolution layer.
/// Training: sum of N kxk conv branches + optional 1x1 branch + optional identity BN branch,
/// followed by LearnableAffineBlock + Act (skipped when stride contains 2).
/// Export: all branches fused into a single Conv2d via rep().
/// Reference: rec_lcnetv3.py LearnableRepLayer
/// </summary>
internal sealed class LearnableRepLayer : Module<Tensor, Tensor>
{
    private readonly int _groups;
    private readonly int _strideH;
    private readonly int _strideW;
    private readonly int _kernelSize;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _padding;

    private readonly BatchNorm2d? _identity;
    private readonly ModuleList<LCNetV3ConvBNLayer> _convKxk;
    private readonly LCNetV3ConvBNLayer? _conv1x1;
    private readonly LearnableAffineBlock _lab;
    private readonly LCNetV3Act _act;

    // Rep fusion state
    private bool _isRepped;
    private Conv2d? _reparamConv;

    public LearnableRepLayer(
        int inChannels,
        int outChannels,
        int kernelSize,
        (int H, int W) stride,
        int groups = 1,
        int numConvBranches = 1)
        : base(nameof(LearnableRepLayer))
    {
        _isRepped = false;
        _groups = groups;
        _strideH = stride.H;
        _strideW = stride.W;
        _kernelSize = kernelSize;
        _inChannels = inChannels;
        _outChannels = outChannels;
        _padding = (kernelSize - 1) / 2;

        // Identity branch: BN when in_ch == out_ch and stride == (1,1)
        _identity = (outChannels == inChannels && stride.H == 1 && stride.W == 1)
            ? BatchNorm2d(inChannels)
            : null;

        // N kxk conv branches
        _convKxk = new ModuleList<LCNetV3ConvBNLayer>();
        for (int i = 0; i < numConvBranches; i++)
        {
            _convKxk.Add(new LCNetV3ConvBNLayer(
                inChannels, outChannels, kernelSize, stride, groups));
        }

        // 1x1 conv branch (when kernel_size > 1)
        _conv1x1 = kernelSize > 1
            ? new LCNetV3ConvBNLayer(inChannels, outChannels, 1, stride, groups)
            : null;

        _lab = new LearnableAffineBlock();
        _act = new LCNetV3Act();

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Export (fused) mode
        if (_isRepped && _reparamConv is not null)
        {
            var outRepped = _lab.call(_reparamConv.call(input));
            if (_strideH != 2 && _strideW != 2)
            {
                outRepped = _act.call(outRepped);
            }
            return outRepped;
        }

        // Training mode: sum all branches
        Tensor? sum = null;

        if (_identity is not null)
        {
            sum = _identity.call(input);
        }

        if (_conv1x1 is not null)
        {
            var c1 = _conv1x1.call(input);
            sum = sum is null ? c1 : sum + c1;
        }

        foreach (var conv in _convKxk)
        {
            var ck = conv.call(input);
            sum = sum is null ? ck : sum + ck;
        }

        // sum should never be null because _convKxk always has at least 1 element
        var result = _lab.call(sum!);
        if (_strideH != 2 && _strideW != 2)
        {
            result = _act.call(result);
        }
        return result;
    }

    /// <summary>
    /// Fuse all branches into a single Conv2d for export/inference.
    /// After calling rep(), forward uses the fused conv instead of multi-branch summation.
    /// Reference: rec_lcnetv3.py LearnableRepLayer.rep()
    /// </summary>
    public void Rep()
    {
        if (_isRepped) return;

        var (kernel, bias) = GetKernelBias();

        _reparamConv = Conv2d(
            _inChannels, _outChannels, (_kernelSize, _kernelSize),
            stride: ((long)_strideH, (long)_strideW),
            padding: (_padding, _padding),
            groups: _groups);

        // Copy fused parameters
        using (torch.no_grad())
        {
            _reparamConv.weight!.copy_(kernel);
            _reparamConv.bias!.copy_(bias);
        }

        kernel.Dispose();
        bias.Dispose();

        _isRepped = true;
        register_module("_reparamConv", _reparamConv);
    }

    /// <summary>Whether this layer has been fused for export.</summary>
    public bool IsRepped => _isRepped;

    // ── Rep fusion internals ──

    private (Tensor kernel, Tensor bias) GetKernelBias()
    {
        // Fuse 1x1 branch
        var (kernel1x1, bias1x1) = FuseBnTensor(_conv1x1);
        kernel1x1 = PadKernel1x1ToKxk(kernel1x1, _kernelSize / 2);

        // Fuse identity branch
        var (kernelId, biasId) = FuseBnTensorIdentity(_identity);

        // Fuse kxk branches (conv is always non-null, so k and b are always non-null)
        Tensor? kernelKxk = null;
        Tensor? biasKxk = null;
        foreach (var conv in _convKxk)
        {
            var (k, b) = FuseBnTensor(conv);
            kernelKxk = kernelKxk is null ? k : kernelKxk + k!;
            biasKxk = biasKxk is null ? b : biasKxk + b!;
        }

        // Sum all branches
        var kernelReparam = kernelKxk!;
        var biasReparam = biasKxk!;

        if (kernel1x1 is not null && bias1x1 is not null)
        {
            kernelReparam = kernelReparam + kernel1x1;
            biasReparam = biasReparam + bias1x1;
        }

        if (kernelId is not null && biasId is not null)
        {
            kernelReparam = kernelReparam + kernelId;
            biasReparam = biasReparam + biasId;
        }

        return (kernelReparam, biasReparam);
    }

    /// <summary>
    /// Fuse ConvBNLayer: fold BN into conv weights.
    /// kernel_fused = kernel * (gamma / std), bias_fused = beta - mean * gamma / std
    /// </summary>
    private static (Tensor? kernel, Tensor? bias) FuseBnTensor(LCNetV3ConvBNLayer? branch)
    {
        if (branch is null) return (null, null);

        var kernel = branch.Conv.weight!;
        var runningMean = branch.BN.running_mean!;
        var runningVar = branch.BN.running_var!;
        var gamma = branch.BN.weight!;
        var beta = branch.BN.bias!;
        var eps = branch.BN.eps;

        using var std = (runningVar + eps).sqrt();
        using var t = (gamma / std).reshape(-1, 1, 1, 1);

        var fusedKernel = kernel * t;
        var fusedBias = beta - runningMean * gamma / std;

        return (fusedKernel, fusedBias);
    }

    /// <summary>
    /// Fuse identity BN branch: create identity kernel and fold BN.
    /// </summary>
    private (Tensor? kernel, Tensor? bias) FuseBnTensorIdentity(BatchNorm2d? branch)
    {
        if (branch is null) return (null, null);

        var inputDim = _inChannels / _groups;
        var kernelValue = torch.zeros(
            _inChannels, inputDim, _kernelSize, _kernelSize,
            dtype: branch.weight!.dtype);

        // Build identity kernel
        for (int i = 0; i < _inChannels; i++)
        {
            kernelValue[i, i % inputDim, _kernelSize / 2, _kernelSize / 2] = 1;
        }

        var runningMean = branch.running_mean!;
        var runningVar = branch.running_var!;
        var gamma = branch.weight!;
        var beta = branch.bias!;
        var eps = branch.eps;

        using var std = (runningVar + eps).sqrt();
        using var t = (gamma / std).reshape(-1, 1, 1, 1);

        var fusedKernel = kernelValue * t;
        var fusedBias = beta - runningMean * gamma / std;

        return (fusedKernel, fusedBias);
    }

    /// <summary>
    /// Pad a 1x1 kernel to kxk by zero-padding.
    /// </summary>
    private static Tensor? PadKernel1x1ToKxk(Tensor? kernel1x1, int pad)
    {
        if (kernel1x1 is null) return null;
        if (pad == 0) return kernel1x1;
        return functional.pad(kernel1x1, [pad, pad, pad, pad]);
    }
}

/// <summary>
/// LCNetV3 building block: depthwise conv (LearnableRepLayer) + optional SE + pointwise conv (LearnableRepLayer).
/// Reference: rec_lcnetv3.py LCNetV3Block
/// </summary>
internal sealed class LCNetV3Block : Module<Tensor, Tensor>
{
    private readonly LearnableRepLayer _dwConv;
    private readonly LCNetV3SELayer? _se;
    private readonly LearnableRepLayer _pwConv;

    public LCNetV3Block(
        int inChannels,
        int outChannels,
        int dwSize,
        (int H, int W) stride,
        bool useSe = false,
        int convKxkNum = 4,
        float lrMult = 1.0f,
        float labLr = 0.1f)
        : base(nameof(LCNetV3Block))
    {
        // Depthwise conv: groups=in_channels
        _dwConv = new LearnableRepLayer(
            inChannels, inChannels, dwSize, stride,
            groups: inChannels,
            numConvBranches: convKxkNum);

        // SE after depthwise (on in_channels, since dw output == in_channels)
        _se = useSe ? new LCNetV3SELayer(inChannels) : null;

        // Pointwise conv: 1x1, stride=(1,1)
        _pwConv = new LearnableRepLayer(
            inChannels, outChannels, 1, (1, 1),
            numConvBranches: convKxkNum);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _dwConv.call(input);
        if (_se is not null)
        {
            x = _se.call(x);
        }
        x = _pwConv.call(x);
        return x;
    }
}
