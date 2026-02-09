using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// SVTRv2 backbone (PP-OCRv4 second gen)：多阶段 Conv + Global attention。
/// 参考: ppocr/modeling/backbones/rec_svtrv2.py
/// </summary>
public sealed class SVTRv2 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _stages;
    public int OutChannels { get; }

    public SVTRv2(
        int inChannels = 3,
        int[] maxSz = null!,
        int[]? depths = null,
        int[]? dims = null,
        string[][]? mixer = null,
        int[]? numHeads = null,
        int mlpRatio = 4,
        bool usePosEmbed = false,
        float dropPathRate = 0.1f,
        bool lastStage = false,
        int outChannels = 192,
        float lastDrop = 0.1f,
        int outCharNum = 25) : base(nameof(SVTRv2))
    {
        maxSz ??= [32, 128];
        depths ??= [3, 6, 3];
        dims ??= [64, 128, 256];
        mixer ??= [
            ["Conv", "Conv", "Conv"],
            ["Conv", "Conv", "Conv", "Global", "Global", "Global"],
            ["Global", "Global", "Global"]
        ];
        numHeads ??= [2, 4, 8];

        var numStages = depths.Length;
        OutChannels = dims[^1];

        // Patch embedding: 2 ConvBN layers with stride 2
        _patchEmbed = Sequential(
            Conv2d(inChannels, dims[0] / 2, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(dims[0] / 2),
            GELU(),
            Conv2d(dims[0] / 2, dims[0], 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(dims[0]),
            GELU()
        );

        // Stochastic depth
        var totalDepth = depths.Sum();
        var dpr = Enumerable.Range(0, totalDepth)
            .Select(i => (double)dropPathRate * i / Math.Max(totalDepth - 1, 1))
            .ToArray();

        _stages = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        var dprOffset = 0;
        for (var i = 0; i < numStages; i++)
        {
            var stageDpr = dpr.Skip(dprOffset).Take(depths[i]).ToArray();
            dprOffset += depths[i];
            var hasDownsample = i < numStages - 1;
            var outDim = hasDownsample ? dims[i + 1] : 0;
            var stage = new SVTRv2Stage(
                dim: dims[i],
                outDim: outDim,
                depth: depths[i],
                mixerTypes: mixer[i],
                numHeads: numHeads[i],
                mlpRatio: mlpRatio,
                dropPath: stageDpr,
                downsample: hasDownsample);
            _stages.Add(stage);
        }

        if (lastStage)
        {
            OutChannels = outChannels;
            _stages.Add(new SVTRv2LastStage(dims[^1], outChannels, lastDrop));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Patch embed
        var x = _patchEmbed.call(input);
        var shape = x.shape;
        var h = (int)shape[2];
        var w = (int)shape[3];

        foreach (var stage in _stages)
        {
            if (stage is SVTRv2Stage s)
            {
                (x, h, w) = s.ForwardWithSize(x, h, w);
            }
            else if (stage is SVTRv2LastStage ls)
            {
                (x, h, w) = ls.ForwardWithSize(x, h, w);
            }
            else
            {
                x = stage.call(x);
            }
        }

        return x;
    }
}

internal sealed class SVTRv2Stage : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor>? _downsampleConv;
    private readonly Module<Tensor, Tensor>? _downsampleNorm;
    private readonly bool _hasDownsample;
    private readonly bool _lastBlockIsConv;
    private readonly int _convBlockCount;

    public SVTRv2Stage(
        int dim, int outDim, int depth, string[] mixerTypes,
        int numHeads, int mlpRatio, double[] dropPath,
        bool downsample) : base(nameof(SVTRv2Stage))
    {
        _hasDownsample = downsample;
        _convBlockCount = mixerTypes.Count(m => m == "Conv");
        _lastBlockIsConv = mixerTypes[^1] == "Conv";

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            if (mixerTypes[i] == "Conv")
            {
                _blocks.Add(new SVTRv2ConvBlock(dim, numHeads, mlpRatio, (float)dropPath[i]));
            }
            else
            {
                _blocks.Add(new SVTRv2AttnBlock(dim, numHeads, mlpRatio, (float)dropPath[i]));
            }
        }

        if (downsample && outDim > 0)
        {
            _downsampleConv = Conv2d(dim, outDim, (3L, 3L), stride: (2L, 1L), padding: (1L, 1L));
            _downsampleNorm = LayerNorm(outDim);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var (x, _, _) = ForwardWithSize(input, (int)input.shape[2], (int)input.shape[3]);
        return x;
    }

    public (Tensor x, int h, int w) ForwardWithSize(Tensor input, int h, int w)
    {
        var x = input;
        var isConvMode = true;
        var convIdx = 0;

        for (var i = 0; i < _blocks.Count; i++)
        {
            var block = _blocks[i];
            if (block is SVTRv2ConvBlock convBlock)
            {
                x = convBlock.call(x);
                convIdx++;
                // After last conv block, if next blocks are attention, flatten
                if (convIdx == _convBlockCount && !_lastBlockIsConv)
                {
                    // x: [B, C, H, W] -> [B, H*W, C]
                    var s = x.shape;
                    x = x.flatten(2).permute(0, 2, 1);
                    isConvMode = false;
                }
            }
            else if (block is SVTRv2AttnBlock attnBlock)
            {
                if (isConvMode)
                {
                    // Convert from 2D to sequence
                    x = x.flatten(2).permute(0, 2, 1);
                    isConvMode = false;
                }
                x = attnBlock.call(x);
            }
        }

        // Downsample
        if (_hasDownsample && _downsampleConv is not null && _downsampleNorm is not null)
        {
            if (!isConvMode)
            {
                // [B, H*W, C] -> [B, C, H, W]
                var c = x.shape[2];
                x = x.permute(0, 2, 1).reshape(-1, c, h, w);
            }
            x = _downsampleConv.call(x);
            var shape = x.shape;
            h = (int)shape[2];
            w = (int)shape[3];
            x = x.flatten(2).permute(0, 2, 1);
            x = _downsampleNorm.call(x);
            // Back to 2D for next stage
            var c2 = x.shape[2];
            x = x.permute(0, 2, 1).reshape(-1, c2, h, w);
        }
        else if (!isConvMode)
        {
            // Keep as sequence for final output
        }

        return (x, h, w);
    }
}

internal sealed class SVTRv2ConvBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _mixer;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly Module<Tensor, Tensor> _mlp;

    public SVTRv2ConvBlock(int dim, int numHeads, int mlpRatio, float dropPath) : base(nameof(SVTRv2ConvBlock))
    {
        _mixer = Conv2d(dim, dim, 5, stride: 1, padding: 2, groups: numHeads);
        _norm1 = LayerNorm(dim);
        _norm2 = LayerNorm(dim);
        var mlpHidden = dim * mlpRatio;
        _mlp = Sequential(
            Linear(dim, mlpHidden),
            GELU(),
            Linear(mlpHidden, dim)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, C, H, W]
        var shape = input.shape;
        var c = shape[1];
        var h = shape[2];
        var w = shape[3];

        using var mixed = _mixer.call(input);
        var x = input + mixed;
        // Flatten and norm
        var flat = x.flatten(2).permute(0, 2, 1); // [B, H*W, C]
        flat = _norm1.call(flat);
        using var mlpOut = _mlp.call(flat);
        flat = _norm2.call(flat + mlpOut);
        // Back to 2D
        return flat.permute(0, 2, 1).reshape(-1, c, h, w);
    }
}

internal sealed class SVTRv2AttnBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly Module<Tensor, Tensor> _qkv;
    private readonly Module<Tensor, Tensor> _proj;
    private readonly Module<Tensor, Tensor> _mlp;
    private readonly int _numHeads;
    private readonly int _dim;

    public SVTRv2AttnBlock(int dim, int numHeads, int mlpRatio, float dropPath) : base(nameof(SVTRv2AttnBlock))
    {
        _dim = dim;
        _numHeads = numHeads;
        _norm1 = LayerNorm(dim);
        _norm2 = LayerNorm(dim);
        _qkv = Linear(dim, dim * 3, hasBias: true);
        _proj = Linear(dim, dim);
        var mlpHidden = dim * mlpRatio;
        _mlp = Sequential(
            Linear(dim, mlpHidden),
            GELU(),
            Linear(mlpHidden, dim)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, seqLen, dim]
        // Pre-norm attention
        var normed = _norm1.call(input);
        var attnOut = MultiHeadAttention(normed);
        var x = input + attnOut;

        normed = _norm2.call(x);
        using var mlpOut = _mlp.call(normed);
        x = x + mlpOut;

        // Post-norm style from SVTRv2: norm after add
        x = _norm1.call(x);
        x = _norm2.call(x);

        return x;
    }

    private Tensor MultiHeadAttention(Tensor x)
    {
        var b = x.shape[0];
        var n = x.shape[1];
        var headDim = _dim / _numHeads;
        var scale = 1.0 / Math.Sqrt(headDim);

        using var qkv = _qkv.call(x);
        var chunks = qkv.chunk(3, dim: -1);
        var q = chunks[0].reshape(b, n, _numHeads, headDim).permute(0, 2, 1, 3);
        var k = chunks[1].reshape(b, n, _numHeads, headDim).permute(0, 2, 1, 3);
        var v = chunks[2].reshape(b, n, _numHeads, headDim).permute(0, 2, 1, 3);

        using var scores = torch.matmul(q, k.transpose(-2, -1)) * scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v);
        output = output.permute(0, 2, 1, 3).reshape(b, n, _dim);
        return _proj.call(output);
    }
}

internal sealed class SVTRv2LastStage : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _lastConv;
    private readonly Module<Tensor, Tensor> _hardswish;
    private readonly Module<Tensor, Tensor> _dropout;

    public SVTRv2LastStage(int inChannels, int outChannels, float lastDrop) : base(nameof(SVTRv2LastStage))
    {
        _lastConv = Linear(inChannels, outChannels, hasBias: false);
        _hardswish = Hardswish();
        _dropout = Dropout(lastDrop);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var (x, _, _) = ForwardWithSize(input, 1, (int)input.shape[1]);
        return x;
    }

    public (Tensor x, int h, int w) ForwardWithSize(Tensor input, int h, int w)
    {
        // input: [B, seqLen, C]
        // Reshape to [B, H, W, C], mean over H
        var x = input.reshape(-1, h, w, input.shape[^1]);
        x = x.mean(new long[] { 1 }); // [B, W, C]
        x = _lastConv.call(x);
        x = _hardswish.call(x);
        x = _dropout.call(x);
        return (x, 1, w);
    }
}
