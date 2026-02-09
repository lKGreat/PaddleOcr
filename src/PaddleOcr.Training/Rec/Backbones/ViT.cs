using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ViT backbone：Generic Vision Transformer for recognition (CPPD etc.)。
/// 支持 prenorm/postnorm 两种模式。
/// 参考: ppocr/modeling/backbones/rec_vit.py
/// </summary>
public sealed class ViT : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly TorchSharp.Modules.Parameter _posEmbed;
    private readonly Module<Tensor, Tensor> _posDrop;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor>? _norm;
    private readonly Module<Tensor, Tensor> _avgPool;
    private readonly Module<Tensor, Tensor> _lastConv;
    private readonly Module<Tensor, Tensor> _hardswish;
    private readonly Module<Tensor, Tensor> _dropout;
    private readonly bool _prenorm;
    public int OutChannels { get; }

    public ViT(
        int inChannels = 3,
        int[] imgSize = null!,
        int[] patchSize = null!,
        int embedDim = 384,
        int depth = 12,
        int numHeads = 6,
        int mlpRatio = 4,
        bool qkvBias = false,
        float dropRate = 0.0f,
        float attnDropRate = 0.0f,
        float dropPathRate = 0.1f,
        bool prenorm = false) : base(nameof(ViT))
    {
        imgSize ??= [32, 128];
        patchSize ??= [4, 4];
        _prenorm = prenorm;
        OutChannels = embedDim;

        _patchEmbed = Conv2d(inChannels, embedDim, ((long)patchSize[0], (long)patchSize[1]), stride: ((long)patchSize[0], (long)patchSize[1]));
        _posEmbed = Parameter(torch.zeros(1, 257, embedDim));
        _posDrop = Dropout(dropRate);

        var dpr = Enumerable.Range(0, depth)
            .Select(i => dropPathRate * i / Math.Max(depth - 1, 1))
            .ToArray();

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            _blocks.Add(new ViTBlockWithPrenorm(embedDim, numHeads, mlpRatio, qkvBias, dropRate, attnDropRate, (float)dpr[i], prenorm));
        }

        if (!prenorm)
        {
            _norm = LayerNorm(embedDim);
        }

        _avgPool = AdaptiveAvgPool2d(new long[] { 1, 25 });
        _lastConv = Conv2d(embedDim, embedDim, 1, bias: false);
        _hardswish = Hardswish();
        _dropout = Dropout(0.1);

        init.trunc_normal_(_posEmbed, std: 0.02);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // [B, C, H, W] -> [B, N, D]
        var x = _patchEmbed.call(input).flatten(2).permute(0, 2, 1);
        var n = x.shape[1];
        x = x + _posEmbed.slice(1, 1, 1 + n, 1);
        x = _posDrop.call(x);

        foreach (var blk in _blocks)
        {
            x = blk.call(x);
        }

        if (!_prenorm && _norm is not null)
        {
            x = _norm.call(x);
        }

        // [B, N, D] -> [B, D, H', W'] -> avg_pool -> last_conv
        var d = x.shape[2];
        x = _avgPool.call(x.permute(0, 2, 1).reshape(-1, d, -1, 25));
        x = _lastConv.call(x);
        x = _hardswish.call(x);
        x = _dropout.call(x);
        return x;
    }
}

internal sealed class ViTBlockWithPrenorm : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly ViTAttention _attn;
    private readonly Module<Tensor, Tensor> _mlp;
    private readonly bool _prenorm;

    public ViTBlockWithPrenorm(int dim, int numHeads, int mlpRatio, bool qkvBias,
        float dropRate, float attnDropRate, float dropPath, bool prenorm) : base(nameof(ViTBlockWithPrenorm))
    {
        _prenorm = prenorm;
        _norm1 = LayerNorm(dim);
        _attn = new ViTAttention(dim, numHeads, qkvBias, attnDropRate, dropRate);
        _norm2 = LayerNorm(dim);
        _mlp = Sequential(
            Linear(dim, dim * mlpRatio),
            GELU(),
            Dropout(dropRate),
            Linear(dim * mlpRatio, dim),
            Dropout(dropRate)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        if (_prenorm)
        {
            // Post-norm: norm(x + attn(x)), norm(x + mlp(x))
            var x = _norm1.call(input + _attn.call(input));
            x = _norm2.call(x + _mlp.call(x));
            return x;
        }
        else
        {
            // Pre-norm: x + attn(norm(x)), x + mlp(norm(x))
            var x = input + _attn.call(_norm1.call(input));
            x = x + _mlp.call(_norm2.call(x));
            return x;
        }
    }
}
