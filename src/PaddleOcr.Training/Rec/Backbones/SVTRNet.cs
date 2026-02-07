using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// SVTRNet backbone：Local/Global 混合注意力 + ConvMixer。
/// 简化实现，用于 SVTR 算法。
/// </summary>
public sealed class SVTRNet : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patch_embed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _norm;
    public int OutChannels { get; }

    public SVTRNet(int inChannels = 3, int embedDim = 192, int depth = 12) : base(nameof(SVTRNet))
    {
        OutChannels = embedDim;

        // Patch embedding: 将图像转换为 patch 序列
        _patch_embed = Sequential(
            Conv2d(inChannels, embedDim / 2, (3, 3), stride: (2, 1), padding: (1, 1), bias: false),
            BatchNorm2d(embedDim / 2),
            ReLU(),
            Conv2d(embedDim / 2, embedDim, (3, 3), stride: (2, 1), padding: (1, 1), bias: false),
            BatchNorm2d(embedDim),
            ReLU()
        );

        // Transformer blocks (简化为 ConvMixer 风格)
        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            _blocks.Add(MixerBlock(embedDim));
        }

        _norm = LayerNorm(embedDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _patch_embed.call(input); // [B, C, H', W']
        var shape = x.shape;
        var b = shape[0];
        var c = shape[1];
        var h = shape[2];
        var w = shape[3];

        // reshape to [B, H*W, C]
        x = x.reshape(b, c, h * w).permute(0, 2, 1);

        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        x = _norm.call(x);
        // reshape back to [B, C, 1, W]（用于 CTC head）
        x = x.permute(0, 2, 1).reshape(b, c, h, w);
        using var pooled = functional.adaptive_avg_pool2d(x, new long[] { 1, w });
        return pooled;
    }

    private static Module<Tensor, Tensor> MixerBlock(int dim)
    {
        return Sequential(
            LayerNorm(dim),
            Linear(dim, dim * 4),
            GELU(),
            Linear(dim * 4, dim)
        );
    }
}
