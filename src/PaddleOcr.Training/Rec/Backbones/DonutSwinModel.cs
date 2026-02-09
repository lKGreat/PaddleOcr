using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// DonutSwinModel backbone：Swin Transformer for document understanding (Donut architecture).
/// 简化实现，保留核心 patch embed + Swin stage 结构。
/// 参考: ppocr/modeling/backbones/rec_donut_swin.py
/// </summary>
public sealed class DonutSwinModel : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _stages;
    private readonly Module<Tensor, Tensor> _norm;
    public int OutChannels { get; }

    public DonutSwinModel(
        int inChannels = 3,
        int embedDim = 96,
        int[] depths = null!,
        int[] numHeads = null!,
        int windowSize = 7,
        float mlpRatio = 4.0f,
        float dropPathRate = 0.1f,
        int patchSize = 4) : base(nameof(DonutSwinModel))
    {
        depths ??= [2, 2, 6, 2];
        numHeads ??= [3, 6, 12, 24];

        // Patch embedding
        _patchEmbed = Sequential(
            Conv2d(inChannels, embedDim, patchSize, stride: patchSize),
            LayerNorm(embedDim)
        );

        // Build stages
        _stages = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        var dim = embedDim;
        for (var i = 0; i < depths.Length; i++)
        {
            var stageBlocks = new List<(string, Module<Tensor, Tensor>)>();
            for (var j = 0; j < depths[i]; j++)
            {
                // Simplified: use standard attention blocks instead of window attention
                stageBlocks.Add(($"block{j}", new SwinBlock(dim, numHeads[i], mlpRatio)));
            }

            _stages.Add(Sequential(stageBlocks.ToArray()));

            // Patch merging (downsample) between stages
            if (i < depths.Length - 1)
            {
                _stages.Add(new PatchMerging(dim));
                dim *= 2;
            }
        }

        _norm = LayerNorm(dim);
        OutChannels = dim;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // [B, C, H, W] -> [B, H*W, C]
        var x = _patchEmbed.call(input);
        var shape = x.shape;
        if (x.dim() == 4)
        {
            x = x.flatten(2).permute(0, 2, 1);
        }

        foreach (var stage in _stages)
        {
            x = stage.call(x);
        }

        x = _norm.call(x);
        return x;
    }
}

internal sealed class SwinBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly ViTAttention _attn;
    private readonly Module<Tensor, Tensor> _mlp;

    public SwinBlock(int dim, int numHeads, float mlpRatio) : base(nameof(SwinBlock))
    {
        _norm1 = LayerNorm(dim);
        _attn = new ViTAttention(dim, numHeads, true, 0, 0);
        _norm2 = LayerNorm(dim);
        _mlp = Sequential(
            Linear(dim, (int)(dim * mlpRatio)),
            GELU(),
            Linear((int)(dim * mlpRatio), dim)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input + _attn.call(_norm1.call(input));
        x = x + _mlp.call(_norm2.call(x));
        return x;
    }
}

internal sealed class PatchMerging : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _reduction;
    private readonly Module<Tensor, Tensor> _norm;

    public PatchMerging(int dim) : base(nameof(PatchMerging))
    {
        _reduction = Linear(4 * dim, 2 * dim, hasBias: false);
        _norm = LayerNorm(4 * dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Simplified patch merging: just reshape and project
        // input: [B, L, C]
        var b = input.shape[0];
        var l = input.shape[1];
        var c = input.shape[2];

        // Approximate: halve sequence length, double channels
        var newL = l / 2;
        if (newL * 2 < l)
        {
            input = input.slice(1, 0, newL * 2, 1);
        }

        var x = input.reshape(b, newL, 2 * c);
        // Pad to 4*C for the norm and reduction
        x = functional.pad(x, new long[] { 0, 2 * c });
        x = _norm.call(x);
        x = _reduction.call(x);
        return x;
    }
}
