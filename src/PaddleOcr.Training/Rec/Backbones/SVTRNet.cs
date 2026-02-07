using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// SVTRNet backbone：Local/Global 混合注意力。
/// 参考 ppocr/modeling/backbones/rec_svtrnet.py。
/// </summary>
public sealed class SVTRNet : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _patchEmbed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly Module<Tensor, Tensor> _norm;
    private readonly int _embedDim;
    public int OutChannels { get; }

    /// <param name="inChannels">输入通道数</param>
    /// <param name="embedDim">Embedding 维度</param>
    /// <param name="depth">Transformer block 总层数</param>
    /// <param name="numHeads">注意力头数</param>
    /// <param name="localWindowSize">Local attention 窗口大小（高度维度方向的 patch 数）</param>
    public SVTRNet(int inChannels = 3, int embedDim = 192, int depth = 12, int numHeads = 6, int localWindowSize = 7) : base(nameof(SVTRNet))
    {
        _embedDim = embedDim;
        OutChannels = embedDim;

        // Patch embedding: 两层卷积，步长 (2,1) 实现高度下采样
        _patchEmbed = Sequential(
            Conv2d(inChannels, embedDim / 2, (3, 3), stride: (2, 1), padding: (1, 1), bias: false),
            BatchNorm2d(embedDim / 2),
            GELU(),
            Conv2d(embedDim / 2, embedDim, (3, 3), stride: (2, 1), padding: (1, 1), bias: false),
            BatchNorm2d(embedDim),
            GELU()
        );

        // Local/Global 混合 Transformer blocks
        _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < depth; i++)
        {
            // 交替使用 Local 和 Global 注意力：偶数层 Local，奇数层 Global
            var useLocal = i % 2 == 0;
            _blocks.Add(new SVTRBlock(embedDim, numHeads, useLocal, localWindowSize));
        }

        _norm = LayerNorm(embedDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _patchEmbed.call(input); // [B, C, H', W']
        var shape = x.shape;
        var b = shape[0];
        var c = shape[1];
        var h = shape[2];
        var w = shape[3];

        // [B, C, H', W'] -> [B, H'*W', C]
        x = x.reshape(b, c, h * w).permute(0, 2, 1);

        foreach (var block in _blocks)
        {
            x = block.call(x);
        }

        x = _norm.call(x);

        // [B, H'*W', C] -> [B, C, H', W'] -> adaptive pool -> [B, C, 1, W']
        x = x.permute(0, 2, 1).reshape(b, c, h, w);
        using var pooled = functional.adaptive_avg_pool2d(x, new long[] { 1, w });
        return pooled;
    }
}

/// <summary>
/// SVTRBlock：Local 或 Global 注意力 block（Pre-Norm + Attention + FFN）。
/// </summary>
internal sealed class SVTRBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly SVTRAttention _attn;
    private readonly Module<Tensor, Tensor> _ffn;

    public SVTRBlock(int dim, int numHeads, bool useLocal, int windowSize) : base(nameof(SVTRBlock))
    {
        _norm1 = LayerNorm(dim);
        _attn = new SVTRAttention(dim, numHeads, useLocal, windowSize);
        _norm2 = LayerNorm(dim);
        _ffn = Sequential(
            Linear(dim, dim * 4),
            GELU(),
            Linear(dim * 4, dim)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Pre-Norm: norm -> attn -> residual
        var normed = _norm1.call(input);
        var attnOut = _attn.call(normed);
        var x = input + attnOut;

        normed = _norm2.call(x);
        var ffnOut = _ffn.call(normed);
        return x + ffnOut;
    }
}

/// <summary>
/// SVTRAttention：支持 Local 窗口注意力和 Global 全局注意力。
/// Local：将序列分为不重叠窗口，在窗口内计算注意力。
/// Global：在整个序列上计算标准注意力。
/// </summary>
internal sealed class SVTRAttention : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _qkvProj;
    private readonly Module<Tensor, Tensor> _outProj;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly bool _useLocal;
    private readonly int _windowSize;

    public SVTRAttention(int dim, int numHeads, bool useLocal, int windowSize) : base(nameof(SVTRAttention))
    {
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _useLocal = useLocal;
        _windowSize = windowSize;
        _qkvProj = Linear(dim, dim * 3);
        _outProj = Linear(dim, dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, seqLen, dim]
        if (_useLocal)
        {
            return LocalAttention(input);
        }

        return GlobalAttention(input);
    }

    private Tensor GlobalAttention(Tensor input)
    {
        var b = input.shape[0];
        var seqLen = input.shape[1];
        var dim = _numHeads * _headDim;

        var qkv = _qkvProj.call(input);
        var chunks = qkv.chunk(3, dim: -1);
        var q = chunks[0].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var k = chunks[1].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = chunks[2].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        var scale = Math.Sqrt(_headDim);
        using var scores = torch.matmul(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v);
        output = output.permute(0, 2, 1, 3).reshape(b, seqLen, dim);
        return _outProj.call(output);
    }

    private Tensor LocalAttention(Tensor input)
    {
        var b = input.shape[0];
        var seqLen = input.shape[1];
        var dim = _numHeads * _headDim;
        var ws = _windowSize;

        // 如果序列长度小于等于窗口大小，退化为全局注意力
        if (seqLen <= ws)
        {
            return GlobalAttention(input);
        }

        // Pad 序列到窗口大小的倍数
        var padLen = (ws - (int)(seqLen % ws)) % ws;
        Tensor x;
        if (padLen > 0)
        {
            x = functional.pad(input, new long[] { 0, 0, 0, padLen });
        }
        else
        {
            x = input;
        }

        var paddedLen = seqLen + padLen;
        var numWindows = paddedLen / ws;

        // [B, paddedLen, dim] -> [B, numWindows, ws, dim]
        x = x.reshape(b, numWindows, ws, dim);

        var qkv = _qkvProj.call(x); // [B, numWindows, ws, dim*3]
        var chunks = qkv.chunk(3, dim: -1);
        // [B, numWindows, ws, numHeads, headDim] -> [B, numWindows, numHeads, ws, headDim]
        var q = chunks[0].reshape(b, numWindows, ws, _numHeads, _headDim).permute(0, 1, 3, 2, 4);
        var k = chunks[1].reshape(b, numWindows, ws, _numHeads, _headDim).permute(0, 1, 3, 2, 4);
        var v = chunks[2].reshape(b, numWindows, ws, _numHeads, _headDim).permute(0, 1, 3, 2, 4);

        // 合并 batch 和 numWindows 维度以便批量计算
        q = q.reshape(b * numWindows, _numHeads, ws, _headDim);
        k = k.reshape(b * numWindows, _numHeads, ws, _headDim);
        v = v.reshape(b * numWindows, _numHeads, ws, _headDim);

        var scale = Math.Sqrt(_headDim);
        using var scores = torch.matmul(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v); // [B*numWindows, numHeads, ws, headDim]

        // 恢复形状
        output = output.reshape(b, numWindows, _numHeads, ws, _headDim);
        output = output.permute(0, 1, 3, 2, 4).reshape(b, paddedLen, dim);

        // 移除 padding
        if (padLen > 0)
        {
            output = output.slice(1, 0, seqLen, 1);
        }

        return _outProj.call(output);
    }
}
