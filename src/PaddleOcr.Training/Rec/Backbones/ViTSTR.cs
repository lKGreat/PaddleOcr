using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ViTSTR backbone：Vision Transformer for Scene Text Recognition。
/// </summary>
public sealed class ViTSTR : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly Module<Tensor, Tensor> _posEmbed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _norm;
    public int OutChannels { get; }

    public ViTSTR(int inChannels = 3, int embedDim = 768, int depth = 12, int numHeads = 12) : base(nameof(ViTSTR))
    {
        OutChannels = embedDim;

        // Patch embedding: 将图像切分为 patches
        _patchEmbed = Sequential(
            Conv2d(inChannels, embedDim, (16, 16), stride: (16, 16), bias: false),
            Flatten(2, 3) // [B, embedDim, H/16, W/16] -> [B, embedDim, H*W/256]
        );

        // Position embedding (可学习)
        _posEmbed = Embedding(256, embedDim); // 假设最大序列长度 256

        // Transformer encoder blocks
        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            _blocks.Add(TransformerBlock(embedDim, numHeads));
        }

        _norm = LayerNorm(embedDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, C, H, W]
        var shape = input.shape;
        var b = shape[0];
        var h = shape[2];
        var w = shape[3];

        // Patch embedding: [B, embedDim, numPatches]
        var x = _patchEmbed.call(input);
        var numPatches = x.shape[2];

        // Transpose: [B, numPatches, embedDim]
        x = x.permute(0, 2, 1);

        // Add position embedding
        var posIds = arange(numPatches, ScalarType.Int64, device: input.device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);
        x = x + posEmb;

        // Transformer blocks
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        x = _norm.call(x);

        // Reshape back: [B, embedDim, 1, numPatches]
        x = x.permute(0, 2, 1).unsqueeze(2);
        return x;
    }

    private static Module<Tensor, Tensor> TransformerBlock(int embedDim, int numHeads)
    {
        return Sequential(
            LayerNorm(embedDim),
            MultiHeadAttention(embedDim, numHeads),
            LayerNorm(embedDim),
            FeedForward(embedDim)
        );
    }

    private static Module<Tensor, Tensor> MultiHeadAttention(int embedDim, int numHeads)
    {
        var headDim = embedDim / numHeads;
        return Sequential(
            ("linear1", Linear(embedDim, embedDim * 3)), // Q, K, V
            ("attn", new AttentionModule(embedDim, numHeads, headDim)),
            ("linear2", Linear(embedDim, embedDim))
        );
    }

    private static Module<Tensor, Tensor> FeedForward(int embedDim)
    {
        return Sequential(
            Linear(embedDim, embedDim * 4),
            GELU(),
            Linear(embedDim * 4, embedDim)
        );
    }

    private sealed class AttentionModule : Module<Tensor, Tensor>
    {
        private readonly int _numHeads;
        private readonly int _headDim;

        public AttentionModule(int embedDim, int numHeads, int headDim) : base("Attention")
        {
            _numHeads = numHeads;
            _headDim = headDim;
        }

        public override Tensor forward(Tensor input)
        {
            // input: [B, seqLen, embedDim*3]
            var shape = input.shape;
            var b = shape[0];
            var seqLen = shape[1];
            var dim = shape[2] / 3;

            // Split into Q, K, V
            var qkv = input.chunk(3, dim: -1);
            var q = qkv[0].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3); // [B, numHeads, seqLen, headDim]
            var k = qkv[1].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
            var v = qkv[2].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);

            // Scaled dot-product attention
            var scale = Math.Sqrt(_headDim);
            using var scores = torch.bmm(q.reshape(b * _numHeads, seqLen, _headDim),
                k.reshape(b * _numHeads, seqLen, _headDim).transpose(-2, -1)) / scale;
            using var attn = functional.softmax(scores, dim: -1);
            var output = torch.bmm(attn, v.reshape(b * _numHeads, seqLen, _headDim));
            output = output.reshape(b, _numHeads, seqLen, _headDim).permute(0, 2, 1, 3).reshape(b, seqLen, dim);

            return output;
        }
    }
}
