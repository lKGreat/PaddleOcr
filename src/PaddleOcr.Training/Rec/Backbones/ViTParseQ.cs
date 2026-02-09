using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ViTParseQ backbone：Vision Transformer for ParseQ (patch-based ViT without cls head).
/// 参考: ppocr/modeling/backbones/rec_vit_parseq.py
/// </summary>
public sealed class ViTParseQ : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly TorchSharp.Modules.Parameter _posEmbed;
    private readonly Module<Tensor, Tensor> _posDrop;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _norm;
    public int OutChannels { get; }

    public ViTParseQ(
        int inChannels = 3,
        int[] imgSize = null!,
        int[] patchSize = null!,
        int embedDim = 384,
        int depth = 12,
        int numHeads = 6,
        float mlpRatio = 4.0f,
        bool qkvBias = true,
        float dropRate = 0.0f,
        float attnDropRate = 0.0f,
        float dropPathRate = 0.0f) : base(nameof(ViTParseQ))
    {
        imgSize ??= [32, 128];
        patchSize ??= [4, 8];
        OutChannels = embedDim;

        var numPatches = (imgSize[0] / patchSize[0]) * (imgSize[1] / patchSize[1]);

        _patchEmbed = Conv2d(inChannels, embedDim, ((long)patchSize[0], (long)patchSize[1]), stride: ((long)patchSize[0], (long)patchSize[1]));
        _posEmbed = Parameter(torch.zeros(1, numPatches, embedDim));
        _posDrop = Dropout(dropRate);

        var dpr = Enumerable.Range(0, depth)
            .Select(i => dropPathRate * i / Math.Max(depth - 1, 1))
            .ToArray();

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            _blocks.Add(new ViTBlock(embedDim, numHeads, mlpRatio, qkvBias, dropRate, attnDropRate, (float)dpr[i]));
        }

        _norm = LayerNorm(embedDim);

        // Initialize pos_embed with truncated normal
        init.trunc_normal_(_posEmbed, std: 0.02);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Patch embed: [B, C, H, W] -> [B, numPatches, embedDim]
        var x = _patchEmbed.call(input).flatten(2).permute(0, 2, 1);
        x = x + _posEmbed;
        x = _posDrop.call(x);

        foreach (var blk in _blocks)
        {
            x = blk.call(x);
        }

        x = _norm.call(x);
        return x;
    }
}

internal sealed class ViTBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly ViTAttention _attn;
    private readonly Module<Tensor, Tensor> _mlp;

    public ViTBlock(int dim, int numHeads, float mlpRatio, bool qkvBias,
        float dropRate, float attnDropRate, float dropPath) : base(nameof(ViTBlock))
    {
        _norm1 = LayerNorm(dim);
        _attn = new ViTAttention(dim, numHeads, qkvBias, attnDropRate, dropRate);
        _norm2 = LayerNorm(dim);
        var mlpHiddenDim = (int)(dim * mlpRatio);
        _mlp = Sequential(
            Linear(dim, mlpHiddenDim),
            GELU(),
            Dropout(dropRate),
            Linear(mlpHiddenDim, dim),
            Dropout(dropRate)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Pre-norm attention
        var x = input + _attn.call(_norm1.call(input));
        x = x + _mlp.call(_norm2.call(x));
        return x;
    }
}

internal sealed class ViTAttention : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _qkv;
    private readonly Module<Tensor, Tensor> _proj;
    private readonly Module<Tensor, Tensor> _attnDrop;
    private readonly Module<Tensor, Tensor> _projDrop;
    private readonly int _numHeads;
    private readonly double _scale;

    public ViTAttention(int dim, int numHeads, bool qkvBias, float attnDrop, float projDrop) : base(nameof(ViTAttention))
    {
        _numHeads = numHeads;
        var headDim = dim / numHeads;
        _scale = 1.0 / Math.Sqrt(headDim);
        _qkv = Linear(dim, dim * 3, hasBias: qkvBias);
        _attnDrop = Dropout(attnDrop);
        _proj = Linear(dim, dim);
        _projDrop = Dropout(projDrop);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var b = input.shape[0];
        var n = input.shape[1];
        var c = input.shape[2];
        var headDim = c / _numHeads;

        using var qkv = _qkv.call(input)
            .reshape(b, n, 3, _numHeads, headDim)
            .permute(2, 0, 3, 1, 4);
        var q = qkv[0];
        var k = qkv[1];
        var v = qkv[2];

        using var attn = torch.matmul(q, k.transpose(-2, -1)) * _scale;
        using var attnSoftmax = functional.softmax(attn, dim: -1);
        using var attnDropped = _attnDrop.call(attnSoftmax);

        var x = torch.matmul(attnDropped, v)
            .permute(0, 2, 1, 3)
            .reshape(b, n, c);

        x = _proj.call(x);
        x = _projDrop.call(x);
        return x;
    }
}
