using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// NRTR head with Transformer encoder/decoder.
/// Training mode supports token input from label_gtc for decoder teacher forcing.
/// </summary>
public sealed class NRTRHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly TransformerEncoder _encoder;
    private readonly TransformerDecoder _decoder;
    private readonly Module<Tensor, Tensor> _outputProj;

    public NRTRHead(
        int inChannels,
        int outChannels,
        int hiddenSize = 512,
        int numHeads = 8,
        int numLayers = 3,
        int maxLen = 25) : base(nameof(NRTRHead))
    {
        _encoder = new TransformerEncoder(inChannels, hiddenSize, numHeads, numLayers);
        _decoder = new TransformerDecoder(hiddenSize, outChannels, numHeads, numLayers, maxLen);
        _outputProj = Linear(hiddenSize, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return ForwardInternal(input, targetTokens: null);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        Tensor? targetTokens = null;
        if (training && targets is not null)
        {
            if (!targets.TryGetValue("label_gtc", out targetTokens))
            {
                targets.TryGetValue("label", out targetTokens);
            }
        }

        var logits = ForwardInternal(input, targetTokens);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }

    private Tensor ForwardInternal(Tensor input, Tensor? targetTokens)
    {
        var encoded = _encoder.call(input);                     // [B, S, H]
        var decoded = _decoder.Forward(encoded, targetTokens); // [B, T, H]
        return _outputProj.call(decoded);                       // [B, T, C]
    }
}

internal sealed class TransformerEncoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _posEmbed;
    private readonly Module<Tensor, Tensor>? _inputProj;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;

    public TransformerEncoder(int inChannels, int hiddenSize, int numHeads, int numLayers) : base(nameof(TransformerEncoder))
    {
        _posEmbed = Embedding(256, hiddenSize);
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
        // input: [B, S, C]
        var b = input.shape[0];
        var s = input.shape[1];
        var x = _inputProj is not null ? _inputProj.call(input) : input;

        var posLen = Math.Min(s, 256);
        var posIds = arange(posLen, ScalarType.Int64, device: input.device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);
        if (s <= 256)
        {
            x = x + posEmb;
        }
        else
        {
            using var posEmbPadded = functional.pad(posEmb, new long[] { 0, 0, 0, s - 256 });
            x = x + posEmbPadded;
        }

        foreach (var layer in _layers)
        {
            x = layer.call(x);
        }

        return x;
    }
}

internal sealed class TransformerEncoderLayer : Module<Tensor, Tensor>
{
    private readonly MultiHeadSelfAttention _selfAttn;
    private readonly Module<Tensor, Tensor> _ffn;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor> _norm2;

    public TransformerEncoderLayer(int hiddenSize, int numHeads) : base(nameof(TransformerEncoderLayer))
    {
        _selfAttn = new MultiHeadSelfAttention(hiddenSize, numHeads);
        _ffn = Sequential(
            Linear(hiddenSize, hiddenSize * 4),
            GELU(),
            Linear(hiddenSize * 4, hiddenSize));
        _norm1 = LayerNorm(hiddenSize);
        _norm2 = LayerNorm(hiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var normed = _norm1.call(input);
        var attnOut = _selfAttn.call(normed);
        var x = input + attnOut;

        normed = _norm2.call(x);
        var ffnOut = _ffn.call(normed);
        return x + ffnOut;
    }
}

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
        var b = input.shape[0];
        var s = input.shape[1];
        var dim = _numHeads * _headDim;

        var qkv = _qkvProj.call(input);
        var chunks = qkv.chunk(3, dim: -1);
        var q = chunks[0].reshape(b, s, _numHeads, _headDim).permute(0, 2, 1, 3);
        var k = chunks[1].reshape(b, s, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = chunks[2].reshape(b, s, _numHeads, _headDim).permute(0, 2, 1, 3);

        var scale = Math.Sqrt(_headDim);
        using var scores = torch.matmul(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v);
        output = output.permute(0, 2, 1, 3).reshape(b, s, dim);
        return _outProj.call(output);
    }
}

internal sealed class MultiHeadCrossAttention : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _qProj;
    private readonly Module<Tensor, Tensor> _kvProj;
    private readonly Module<Tensor, Tensor> _outProj;
    private readonly int _numHeads;
    private readonly int _headDim;

    public MultiHeadCrossAttention(int hiddenSize, int numHeads) : base(nameof(MultiHeadCrossAttention))
    {
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _qProj = Linear(hiddenSize, hiddenSize);
        _kvProj = Linear(hiddenSize, hiddenSize * 2);
        _outProj = Linear(hiddenSize, hiddenSize);
        RegisterComponents();
    }

    public Tensor Forward(Tensor decoderInput, Tensor encoderOutput)
    {
        var b = decoderInput.shape[0];
        var tgtLen = decoderInput.shape[1];
        var srcLen = encoderOutput.shape[1];
        var dim = _numHeads * _headDim;

        var q = _qProj.call(decoderInput).reshape(b, tgtLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var kv = _kvProj.call(encoderOutput);
        var kvChunks = kv.chunk(2, dim: -1);
        var k = kvChunks[0].reshape(b, srcLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = kvChunks[1].reshape(b, srcLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        var scale = Math.Sqrt(_headDim);
        using var scores = torch.matmul(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.matmul(attn, v);
        output = output.permute(0, 2, 1, 3).reshape(b, tgtLen, dim);
        return _outProj.call(output);
    }

    public override Tensor forward(Tensor input)
    {
        throw new InvalidOperationException("Use Forward(decoderInput, encoderOutput) instead.");
    }
}

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

    public Tensor Forward(Tensor encoderOutput, Tensor? targetTokens = null)
    {
        var b = encoderOutput.shape[0];
        var device = encoderOutput.device;

        var tokenIds = zeros(new long[] { b, _maxLen }, ScalarType.Int64, device: device);
        if (targetTokens is not null && targetTokens.shape.Length == 2)
        {
            using var target64 = targetTokens.to_type(ScalarType.Int64).to(device);
            if (target64.shape[0] == b)
            {
                var copyLen = Math.Min((int)target64.shape[1], _maxLen);
                if (copyLen > 0)
                {
                    tokenIds.narrow(1, 0, copyLen).copy_(target64.narrow(1, 0, copyLen));
                }
            }
        }

        var x = _embedding.call(tokenIds);
        var posIds = arange(_maxLen, ScalarType.Int64, device: device).unsqueeze(0).expand(b, -1);
        using var posEmb = _posEmbed.call(posIds);
        x = x + posEmb;

        foreach (var layer in _layers)
        {
            x = layer.Forward(x, encoderOutput);
        }

        return x;
    }

    public override Tensor forward(Tensor encoderOutput) => Forward(encoderOutput, targetTokens: null);
}

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
            Linear(hiddenSize * 4, hiddenSize));
        _norm1 = LayerNorm(hiddenSize);
        _norm2 = LayerNorm(hiddenSize);
        _norm3 = LayerNorm(hiddenSize);
        RegisterComponents();
    }

    public Tensor Forward(Tensor input, Tensor encoderOutput)
    {
        var normed = _norm1.call(input);
        var selfAttnOut = _selfAttn.call(normed);
        var x = input + selfAttnOut;

        normed = _norm2.call(x);
        var crossAttnOut = _crossAttn.Forward(normed, encoderOutput);
        x = x + crossAttnOut;

        normed = _norm3.call(x);
        var ffnOut = _ffn.call(normed);
        return x + ffnOut;
    }

    public override Tensor forward(Tensor input)
    {
        throw new InvalidOperationException("Use Forward(input, encoderOutput) instead.");
    }
}
