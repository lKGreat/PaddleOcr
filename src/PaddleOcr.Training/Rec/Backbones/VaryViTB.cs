using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// Vary_VIT_B backbone：SAM-like ViT encoder for formula recognition.
/// 参考: ppocr/modeling/backbones/rec_vary_vit.py - ImageEncoderViT
/// </summary>
public sealed class VaryViTB : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly TorchSharp.Modules.Parameter? _posEmbed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _neck;
    public int OutChannels { get; }

    public VaryViTB(
        int inChannels = 3,
        int imgSize = 1024,
        int patchSize = 16,
        int embedDim = 768,
        int depth = 12,
        int numHeads = 12,
        float mlpRatio = 4.0f,
        int outChans = 256,
        bool usePosEmbed = true) : base(nameof(VaryViTB))
    {
        _patchEmbed = Conv2d(inChannels, embedDim, patchSize, stride: patchSize);

        if (usePosEmbed)
        {
            var numPatches = (imgSize / patchSize) * (imgSize / patchSize);
            _posEmbed = Parameter(torch.zeros(1, numPatches, embedDim));
            init.trunc_normal_(_posEmbed, std: 0.02);
        }

        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            _blocks.Add(new ViTBlock(embedDim, numHeads, mlpRatio, true, 0, 0, 0));
        }

        _neck = Sequential(
            Conv2d(embedDim, outChans, 1, bias: false),
            LayerNorm(outChans),
            Conv2d(outChans, outChans, 3, padding: 1, bias: false),
            LayerNorm(outChans)
        );

        OutChannels = outChans;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _patchEmbed.call(input).flatten(2).permute(0, 2, 1);
        if (_posEmbed is not null)
        {
            x = x + _posEmbed;
        }
        foreach (var blk in _blocks)
        {
            x = blk.call(x);
        }
        // Reshape back to 2D for neck
        var c = x.shape[2];
        var hw = (int)Math.Sqrt(x.shape[1]);
        x = x.permute(0, 2, 1).reshape(-1, c, hw, hw);
        x = _neck.call(x);
        return x;
    }
}

/// <summary>
/// Vary_VIT_B_Formula backbone：Formula recognition variant。
/// </summary>
public sealed class VaryViTBFormula : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly VaryViTB _inner;
    public int OutChannels => _inner.OutChannels;

    public VaryViTBFormula(int inChannels = 3) : base(nameof(VaryViTBFormula))
    {
        _inner = new VaryViTB(inChannels, imgSize: 768, patchSize: 16, embedDim: 768, depth: 12, numHeads: 12, outChans: 256);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => _inner.call(input);
}
