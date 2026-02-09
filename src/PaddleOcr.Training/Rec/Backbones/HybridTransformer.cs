using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// HybridTransformer backbone：CNN (ResNetV2) + ViT hybrid encoder for LaTeX-OCR.
/// 参考: ppocr/modeling/backbones/rec_hybridvit.py
/// </summary>
public sealed class HybridTransformer : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _proj;
    private readonly TorchSharp.Modules.Parameter _clsToken;
    private readonly TorchSharp.Modules.Parameter _posEmbed;
    private readonly Module<Tensor, Tensor> _posDrop;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _norm;
    private readonly int _patchSize;
    private readonly int _width;
    public int OutChannels { get; }

    public HybridTransformer(
        int inputChannel = 1,
        int[] backboneLayers = null!,
        int[] imgSize = null!,
        int patchSize = 16,
        int embedDim = 256,
        int depth = 4,
        int numHeads = 8,
        float mlpRatio = 4.0f,
        bool qkvBias = true,
        float dropRate = 0.0f,
        float attnDropRate = 0.0f,
        float dropPathRate = 0.0f) : base(nameof(HybridTransformer))
    {
        backboneLayers ??= [2, 3, 7];
        imgSize ??= [224, 224];
        OutChannels = embedDim;
        _patchSize = patchSize;
        _width = imgSize[1];

        // CNN backbone (ResNetV2-like)
        _backbone = new ResNetV2(inputChannel, backboneLayers);

        // Hybrid embedding: project CNN features to embed_dim
        var featureDim = 1024;
        _proj = Conv2d(featureDim, embedDim, 1);

        var numPatches = 42 * 12; // feature_size from HybridEmbed
        _clsToken = Parameter(torch.zeros(1, 1, embedDim));
        _posEmbed = Parameter(torch.zeros(1, numPatches + 1, embedDim));
        _posDrop = Dropout(dropRate);

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            _blocks.Add(new ViTBlock(embedDim, numHeads, mlpRatio, qkvBias, dropRate, attnDropRate, 0));
        }

        _norm = LayerNorm(embedDim);
        init.trunc_normal_(_posEmbed, std: 0.02);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var b = input.shape[0];

        // CNN features
        var x = _backbone.call(input);
        x = _proj.call(x).flatten(2).permute(0, 2, 1);

        // Prepend cls token
        var clsTokens = _clsToken.expand(b, -1, -1);
        x = torch.cat([clsTokens, x], dim: 1);

        // Add positional embedding (truncated to actual length)
        var seqLen = x.shape[1];
        if (seqLen <= _posEmbed.shape[1])
        {
            x = x + _posEmbed.slice(1, 0, seqLen, 1);
        }

        x = _posDrop.call(x);

        foreach (var blk in _blocks)
        {
            x = blk.call(x);
        }

        x = _norm.call(x);
        return x;
    }
}
