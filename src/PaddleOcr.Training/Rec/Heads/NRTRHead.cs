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
        var decoded = _decoder.call(encoded); // [B, maxLen, hiddenSize]
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
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;

    public TransformerEncoder(int inChannels, int hiddenSize, int numHeads, int numLayers) : base(nameof(TransformerEncoder))
    {
        _posEmbed = Embedding(256, hiddenSize); // 假设最大序列长度 256
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

        // 投影到 hiddenSize
        var x = input;
        if (input.shape[2] != ((TransformerEncoderLayer)_layers[0]).HiddenSize)
        {
            x = Linear(input.shape[2], ((TransformerEncoderLayer)_layers[0]).HiddenSize).call(input);
        }

        // 添加位置编码
        var posIds = arange(w, ScalarType.Int64, device: input.device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);
        x = x + posEmb;

        foreach (var layer in _layers)
        {
            x = layer.call(x);
        }

        return x;
    }
}

/// <summary>
/// TransformerEncoderLayer：Transformer 编码层。
/// </summary>
internal sealed class TransformerEncoderLayer : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _selfAttn;
    private readonly Module<Tensor, Tensor> _ffn;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    public int HiddenSize { get; }

    public TransformerEncoderLayer(int hiddenSize, int numHeads) : base(nameof(TransformerEncoderLayer))
    {
        HiddenSize = hiddenSize;
        _selfAttn = MultiHeadAttention(hiddenSize, numHeads);
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
        using var residual = input;
        var x = _norm1.call(input);
        x = _selfAttn.call(x);
        x = x + residual;

        using var residual2 = x;
        x = _norm2.call(x);
        x = _ffn.call(x);
        return x + residual2;
    }

    private static Module<Tensor, Tensor> MultiHeadAttention(int hiddenSize, int numHeads)
    {
        var headDim = hiddenSize / numHeads;
        return Sequential(
            ("linear1", Linear(hiddenSize, hiddenSize * 3)),
            ("attn", new MultiHeadAttentionModule(hiddenSize, numHeads, headDim)),
            ("linear2", Linear(hiddenSize, hiddenSize))
        );
    }
}

/// <summary>
/// MultiHeadAttentionModule：多头注意力模块。
/// </summary>
internal sealed class MultiHeadAttentionModule : Module<Tensor, Tensor>
{
    private readonly int _numHeads;
    private readonly int _headDim;

    public MultiHeadAttentionModule(int hiddenSize, int numHeads, int headDim) : base("MultiHeadAttention")
    {
        _numHeads = numHeads;
        _headDim = headDim;
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, seqLen, hiddenSize*3]
        var shape = input.shape;
        var b = shape[0];
        var seqLen = shape[1];
        var dim = shape[2] / 3;

        var qkv = input.chunk(3, dim: -1);
        var q = qkv[0].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var k = qkv[1].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = qkv[2].reshape(b, seqLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        var scale = Math.Sqrt(_headDim);
        using var scores = torch.bmm(q.reshape(b * _numHeads, seqLen, _headDim),
            k.reshape(b * _numHeads, seqLen, _headDim).transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.bmm(attn, v.reshape(b * _numHeads, seqLen, _headDim));
        return output.reshape(b, _numHeads, seqLen, _headDim).permute(0, 2, 1, 3).reshape(b, seqLen, dim);
    }
}

/// <summary>
/// TransformerDecoder：Transformer 解码器。
/// </summary>
internal sealed class TransformerDecoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _embedding;
    private readonly Module<Tensor, Tensor> _posEmbed;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;
    private readonly int _maxLen;

    public TransformerDecoder(int hiddenSize, int vocabSize, int numHeads, int numLayers, int maxLen) : base(nameof(TransformerDecoder))
    {
        _maxLen = maxLen;
        _embedding = Embedding(vocabSize, hiddenSize);
        _posEmbed = Embedding(maxLen, hiddenSize);
        _layers = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < numLayers; i++)
        {
            _layers.Add(new TransformerDecoderLayer(hiddenSize, numHeads));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor encoderOutput)
    {
        // encoderOutput: [B, W, hiddenSize]
        var b = encoderOutput.shape[0];
        var device = encoderOutput.device;

        // 创建解码器输入（SOS tokens）
        var sosIds = zeros(new long[] { b, _maxLen }, ScalarType.Int64, device: device);
        var x = _embedding.call(sosIds); // [B, maxLen, hiddenSize]

        // 添加位置编码
        var posIds = arange(_maxLen, ScalarType.Int64, device: device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);
        x = x + posEmb;

        foreach (var layer in _layers)
        {
            ((TransformerDecoderLayer)layer).SetEncoderOutput(encoderOutput);
            x = layer.call(x);
        }

        return x;
    }
}

/// <summary>
/// TransformerDecoderLayer：Transformer 解码层。
/// </summary>
internal sealed class TransformerDecoderLayer : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _selfAttn;
    private readonly Module<Tensor, Tensor> _crossAttn;
    private readonly Module<Tensor, Tensor> _ffn;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;
    private readonly Module<Tensor, Tensor> _norm3;
    private Tensor? _encoderOutput;

    public TransformerDecoderLayer(int hiddenSize, int numHeads) : base(nameof(TransformerDecoderLayer))
    {
        _selfAttn = MultiHeadAttention(hiddenSize, numHeads);
        _crossAttn = MultiHeadAttention(hiddenSize, numHeads);
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

    public void SetEncoderOutput(Tensor encoderOutput)
    {
        _encoderOutput = encoderOutput;
    }

    public override Tensor forward(Tensor input)
    {
        if (_encoderOutput is null)
        {
            throw new InvalidOperationException("Encoder output must be set before forward");
        }

        // Self-attention
        using var residual = input;
        var x = _norm1.call(input);
        x = _selfAttn.call(x);
        x = x + residual;

        // Cross-attention
        using var residual2 = x;
        x = _norm2.call(x);
        // 简化：使用 encoderOutput 作为 K, V
        x = _crossAttn.call(cat(new[] { x, _encoderOutput }, dim: 1));
        x = x + residual2;

        // FFN
        using var residual3 = x;
        x = _norm3.call(x);
        x = _ffn.call(x);
        return x + residual3;
    }

    private static Module<Tensor, Tensor> MultiHeadAttention(int hiddenSize, int numHeads)
    {
        var headDim = hiddenSize / numHeads;
        return Sequential(
            ("linear1", Linear(hiddenSize, hiddenSize * 3)),
            ("attn", new MultiHeadAttentionModule(hiddenSize, numHeads, headDim)),
            ("linear2", Linear(hiddenSize, hiddenSize))
        );
    }
}
