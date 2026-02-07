using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// NRTRHead：Encoder-Decoder Transformer + 位置编码。
/// 用于 NRTR 算法。
/// </summary>
public sealed class NRTRHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly TransformerEncoder _encoder;
    private readonly TransformerDecoder _decoder;
    private readonly Module<Tensor, Tensor> _outputProj;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public NRTRHead(int inChannels, int outChannels, int hiddenSize = 512, int numHeads = 8, int numLayers = 3, int maxLen = 25) : base(nameof(NRTRHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;
        _encoder = new TransformerEncoder(inChannels, hiddenSize, numHeads, numLayers);
        _decoder = new TransformerDecoder(hiddenSize, outChannels, numHeads, numLayers, maxLen);
        _outputProj = Linear(hiddenSize, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var encoded = _encoder.call(input); // [B, W, hiddenSize]
        var decoded = _decoder.Forward(encoded); // [B, maxLen, hiddenSize]
        return _outputProj.call(decoded); // [B, maxLen, outChannels]
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}

/// <summary>
/// TransformerEncoder：Transformer 编码器。
/// </summary>
internal sealed class TransformerEncoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _posEmbed;
    private readonly Module<Tensor, Tensor>? _inputProj;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;
    private readonly int _hiddenSize;

    public TransformerEncoder(int inChannels, int hiddenSize, int numHeads, int numLayers) : base(nameof(TransformerEncoder))
    {
        _hiddenSize = hiddenSize;
        _posEmbed = Embedding(256, hiddenSize); // 最大序列长度 256

        // 在构造函数中预创建投影层（如果 inChannels != hiddenSize）
        _inputProj = inChannels != hiddenSize ? Linear(inChannels, hiddenSize) : null;

        _layers = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < numLayers; i++)
        {
            _layers.Add(new TransformerEncoderLayer(hiddenSize, numHeads));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var shape = input.shape;
        var b = shape[0];
        var w = shape[1];

        // 如果 inChannels != hiddenSize，使用预注册的投影层
        var x = _inputProj is not null ? _inputProj.call(input) : input;

        // 添加位置编码
        var posLen = Math.Min(w, 256);
        var posIds = arange(posLen, ScalarType.Int64, device: input.device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);

        // 如果序列长度超过位置编码范围，只对前部分加位置编码
        if (w <= 256)
        {
            x = x + posEmb;
        }
        else
        {
            using var posEmbPadded = functional.pad(posEmb, new long[] { 0, 0, 0, w - 256 });
            x = x + posEmbPadded;
        }

        foreach (var layer in _layers)
        {
            x = layer.call(x);
        }

        return x;
    }
}

/// <summary>
/// TransformerEncoderLayer：Transformer 编码层（Pre-Norm 结构）。
/// </summary>
internal sealed class TransformerEncoderLayer : Module<Tensor, Tensor>
{
    private readonly MultiHeadSelfAttention _selfAttn;
    private readonly Module<Tensor, Tensor> _ffn;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    public int HiddenSize { get; }

    public TransformerEncoderLayer(int hiddenSize, int numHeads) : base(nameof(TransformerEncoderLayer))
    {
        HiddenSize = hiddenSize;
        _selfAttn = new MultiHeadSelfAttention(hiddenSize, numHeads);
        _ffn = Sequential(
            Linear(hiddenSize, hiddenSize * 4),
            GELU(),
            Linear(hiddenSize * 4, hiddenSize)
        );
        _norm1 = LayerNorm(hiddenSize);
        _norm2 = LayerNorm(hiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Pre-Norm: norm -> attn -> residual
        var normed = _norm1.call(input);
        var attnOut = _selfAttn.call(normed);
        var x = input + attnOut;

        normed = _norm2.call(x);
        var ffnOut = _ffn.call(normed);
        return x + ffnOut;
    }
}

/// <summary>
/// MultiHeadSelfAttention：多头自注意力模块（Q/K/V 来自同一输入）。
/// </summary>
internal sealed class MultiHeadSelfAttention : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _qkvProj;
    private readonly Module<Tensor, Tensor> _outProj;
    private readonly int _numHeads;
    private readonly int _headDim;

    public MultiHeadSelfAttention(int hiddenSize, int numHeads) : base(nameof(MultiHeadSelfAttention))
    {
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _qkvProj = Linear(hiddenSize, hiddenSize * 3);
        _outProj = Linear(hiddenSize, hiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, seqLen, hiddenSize]
        var shape = input.shape;
        var b = shape[0];
        var seqLen = shape[1];
        var dim = _numHeads * _headDim;

        var qkv = _qkvProj.call(input);
        var chunks = qkv.chunk(3, dim: -1);
        var q = chunks[0].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var k = chunks[1].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = chunks[2].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        var scale = Math.Sqrt(_headDim);
        // [B, numHeads, seqLen, headDim] x [B, numHeads, headDim, seqLen] -> [B, numHeads, seqLen, seqLen]
        using var scores = torch.matmul(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v); // [B, numHeads, seqLen, headDim]
        output = output.permute(0, 2, 1, 3).reshape(b, seqLen, dim);
        return _outProj.call(output);
    }
}

/// <summary>
/// MultiHeadCrossAttention：多头交叉注意力模块（Q 来自 decoder，K/V 来自 encoder）。
/// </summary>
internal sealed class MultiHeadCrossAttention : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _qProj;
    private readonly Module<Tensor, Tensor> _kvProj;
    private readonly Module<Tensor, Tensor> _outProj;
    private readonly int _numHeads;
    private readonly int _headDim;
    private Tensor? _encoderOutput;

    public MultiHeadCrossAttention(int hiddenSize, int numHeads) : base(nameof(MultiHeadCrossAttention))
    {
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _qProj = Linear(hiddenSize, hiddenSize);
        _kvProj = Linear(hiddenSize, hiddenSize * 2);
        _outProj = Linear(hiddenSize, hiddenSize);
        RegisterComponents();
    }

    /// <summary>
    /// 设置编码器输出（用于提供 K/V）。
    /// </summary>
    public void SetEncoderOutput(Tensor encoderOutput)
    {
        _encoderOutput = encoderOutput;
    }

    public override Tensor forward(Tensor decoderInput)
    {
        if (_encoderOutput is null)
        {
            throw new InvalidOperationException("Encoder output must be set before cross-attention forward");
        }

        var b = decoderInput.shape[0];
        var tgtLen = decoderInput.shape[1];
        var srcLen = _encoderOutput.shape[1];
        var dim = _numHeads * _headDim;

        // Q 来自 decoder input
        var q = _qProj.call(decoderInput).reshape(b, tgtLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        // K, V 来自 encoder output
        var kv = _kvProj.call(_encoderOutput);
        var kvChunks = kv.chunk(2, dim: -1);
        var k = kvChunks[0].reshape(b, srcLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = kvChunks[1].reshape(b, srcLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        var scale = Math.Sqrt(_headDim);
        // [B, numHeads, tgtLen, headDim] x [B, numHeads, headDim, srcLen] -> [B, numHeads, tgtLen, srcLen]
        using var scores = torch.matmul(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v); // [B, numHeads, tgtLen, headDim]
        output = output.permute(0, 2, 1, 3).reshape(b, tgtLen, dim);
        return _outProj.call(output);
    }
}

/// <summary>
/// TransformerDecoder：Transformer 解码器。
/// </summary>
internal sealed class TransformerDecoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _embedding;
    private readonly Module<Tensor, Tensor> _posEmbed;
    private readonly TorchSharp.Modules.ModuleList<TransformerDecoderLayer> _layers;
    private readonly int _maxLen;

    public TransformerDecoder(int hiddenSize, int vocabSize, int numHeads, int numLayers, int maxLen) : base(nameof(TransformerDecoder))
    {
        _maxLen = maxLen;
        _embedding = Embedding(vocabSize, hiddenSize);
        _posEmbed = Embedding(maxLen, hiddenSize);
        _layers = new TorchSharp.Modules.ModuleList<TransformerDecoderLayer>();
        for (var i = 0; i < numLayers; i++)
        {
            _layers.Add(new TransformerDecoderLayer(hiddenSize, numHeads));
        }

        RegisterComponents();
    }

    /// <summary>
    /// 解码器前向传播（接受 encoderOutput 作为参数，避免手动 Set 状态）。
    /// </summary>
    public Tensor Forward(Tensor encoderOutput)
    {
        var b = encoderOutput.shape[0];
        var device = encoderOutput.device;

        // 创建解码器输入（SOS tokens）
        var sosIds = zeros(new long[] { b, _maxLen }, ScalarType.Int64, device: device);
        var x = _embedding.call(sosIds);

        // 添加位置编码
        var posIds = arange(_maxLen, ScalarType.Int64, device: device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);
        x = x + posEmb;

        foreach (var layer in _layers)
        {
            x = layer.Forward(x, encoderOutput);
        }

        return x;
    }

    public override Tensor forward(Tensor encoderOutput) => Forward(encoderOutput);
}

/// <summary>
/// TransformerDecoderLayer：Transformer 解码层（Pre-Norm 结构，标准 Cross-Attention）。
/// </summary>
internal sealed class TransformerDecoderLayer : Module<Tensor, Tensor>
{
    private readonly MultiHeadSelfAttention _selfAttn;
    private readonly MultiHeadCrossAttention _crossAttn;
    private readonly Module<Tensor, Tensor> _ffn;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly Module<Tensor, Tensor> _norm3;

    public TransformerDecoderLayer(int hiddenSize, int numHeads) : base(nameof(TransformerDecoderLayer))
    {
        _selfAttn = new MultiHeadSelfAttention(hiddenSize, numHeads);
        _crossAttn = new MultiHeadCrossAttention(hiddenSize, numHeads);
        _ffn = Sequential(
            Linear(hiddenSize, hiddenSize * 4),
            GELU(),
            Linear(hiddenSize * 4, hiddenSize)
        );
        _norm1 = LayerNorm(hiddenSize);
        _norm2 = LayerNorm(hiddenSize);
        _norm3 = LayerNorm(hiddenSize);
        RegisterComponents();
    }

    /// <summary>
    /// 解码层前向传播（通过参数传递 encoderOutput，而非 Set 方法）。
    /// </summary>
    public Tensor Forward(Tensor input, Tensor encoderOutput)
    {
        // Self-attention
        var normed = _norm1.call(input);
        var selfAttnOut = _selfAttn.call(normed);
        var x = input + selfAttnOut;

        // Cross-attention（Q 来自 decoder，K/V 来自 encoder）
        normed = _norm2.call(x);
        _crossAttn.SetEncoderOutput(encoderOutput);
        var crossAttnOut = _crossAttn.call(normed);
        x = x + crossAttnOut;

        // FFN
        normed = _norm3.call(x);
        var ffnOut = _ffn.call(normed);
        return x + ffnOut;
    }

    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use Forward(input, encoderOutput) instead");
}
