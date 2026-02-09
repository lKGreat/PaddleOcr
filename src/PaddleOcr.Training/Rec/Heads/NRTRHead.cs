using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// NRTR head with Transformer encoder/decoder.
/// 1:1 port of ppocr/modeling/heads/rec_nrtr_head.py Transformer.
///
/// When numEncoderLayers &lt;= 0 (default for MultiHead usage), the encoder is skipped
/// and the input is used directly as memory for the decoder cross-attention.
/// This matches the Python behavior where MultiHead passes num_encoder_layers=-1.
///
/// Training mode supports token input from label_gtc for decoder teacher forcing.
/// Uses causal mask for decoder self-attention (matching Python generate_square_subsequent_mask).
/// </summary>
public sealed class NRTRHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly TransformerEncoder? _encoder;
    private readonly PositionalEncoding _positionalEncoding;
    private readonly NRTRDecoder _decoder;
    private readonly Module<Tensor, Tensor> _outputProj;
    private readonly NRTREmbeddings _embedding;
    private readonly int _outChannelsInternal;
    private readonly int _maxLen;
    private readonly int _dModel;

    public NRTRHead(
        int inChannels,
        int outChannels,
        int hiddenSize = 512,
        int numHeads = 8,
        int numEncoderLayers = -1,
        int numDecoderLayers = 6,
        int maxLen = 25,
        bool scaleEmbedding = true) : base(nameof(NRTRHead))
    {
        _dModel = hiddenSize;
        _maxLen = maxLen;
        // Python: self.out_channels = out_channels + 1 (adds padding token)
        _outChannelsInternal = outChannels + 1;

        _embedding = new NRTREmbeddings(_dModel, _outChannelsInternal, paddingIdx: 0, scaleEmbedding: scaleEmbedding);
        _positionalEncoding = new PositionalEncoding(dropout: 0.1f, dim: _dModel);

        // Encoder: optional (when numEncoderLayers <= 0, skip encoder like Python num_encoder_layers=-1)
        if (numEncoderLayers > 0)
        {
            _encoder = new TransformerEncoder(hiddenSize, numHeads, numEncoderLayers);
        }

        // Decoder: always present
        _decoder = new NRTRDecoder(hiddenSize, numHeads, numDecoderLayers);

        // Output projection (no bias, matching Python)
        _outputProj = Linear(hiddenSize, _outChannelsInternal, hasBias: false);

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
        // Memory (encoder output or raw input)
        Tensor memory;
        if (_encoder is not null)
        {
            var src = _positionalEncoding.call(input);
            memory = _encoder.call(src);
        }
        else
        {
            // No encoder: use input directly as memory (Python: memory = src)
            memory = input;
        }

        if (training && targetTokens is not null)
        {
            return ForwardTrain(memory, targetTokens);
        }

        return ForwardTest(memory);
    }

    /// <summary>
    /// Training forward: teacher forcing with target tokens.
    /// Python: Transformer.forward_train(src, tgt)
    /// </summary>
    private Tensor ForwardTrain(Tensor memory, Tensor targetTokens)
    {
        // Python: tgt = tgt[:, :-1] — remove last token (EOS is target, not input)
        var tgt = targetTokens.to_type(ScalarType.Int64);
        if (tgt.shape[1] > 1)
        {
            tgt = tgt[.., ..^1]; // Remove last column
        }

        var tgtEmbed = _embedding.call(tgt);
        tgtEmbed = _positionalEncoding.call(tgtEmbed);

        // Causal mask for decoder self-attention
        using var tgtMask = GenerateSquareSubsequentMask((int)tgtEmbed.shape[1], tgtEmbed.device);

        var output = _decoder.Forward(tgtEmbed, memory, tgtMask);
        return _outputProj.call(output);
    }

    /// <summary>
    /// Test forward: autoregressive decoding.
    /// Python: Transformer.forward_test(src)
    /// </summary>
    private Tensor ForwardTest(Tensor memory)
    {
        var bs = (int)memory.shape[0];
        var device = memory.device;

        // Start with BOS token (index 2 in Python)
        var decSeq = torch.full(bs, 1, 2L, ScalarType.Int64, device: device);

        for (int step = 1; step < _maxLen; step++)
        {
            var decSeqEmbed = _embedding.call(decSeq);
            decSeqEmbed = _positionalEncoding.call(decSeqEmbed);
            using var tgtMask = GenerateSquareSubsequentMask((int)decSeqEmbed.shape[1], device);

            var output = _decoder.Forward(decSeqEmbed, memory, tgtMask);
            var lastStep = output[.., ^1, ..]; // [B, dModel]
            var wordProb = functional.softmax(_outputProj.call(lastStep), dim: -1);
            var predsIdx = wordProb.argmax(dim: -1); // [B]

            // Check if all predictions are EOS (index 3 in Python)
            using var eosCheck = predsIdx.eq(torch.tensor(3L, ScalarType.Int64, device: device));
            if (eosCheck.all().item<bool>())
            {
                break;
            }

            decSeq = torch.cat([decSeq, predsIdx.unsqueeze(-1)], dim: 1);
        }

        // Return logits for full sequence
        var finalEmbed = _embedding.call(decSeq);
        finalEmbed = _positionalEncoding.call(finalEmbed);
        using var finalMask = GenerateSquareSubsequentMask((int)finalEmbed.shape[1], device);
        var finalOutput = _decoder.Forward(finalEmbed, memory, finalMask);
        return _outputProj.call(finalOutput);
    }

    /// <summary>
    /// Generate causal mask for decoder self-attention.
    /// Python: self.generate_square_subsequent_mask(sz)
    /// Upper-triangular matrix of -inf, diagonal and below are 0.
    /// </summary>
    private static Tensor GenerateSquareSubsequentMask(int sz, Device device)
    {
        var mask = torch.full(sz, sz, float.NegativeInfinity, device: device);
        return mask.triu(1); // Zero out diagonal and below
    }
}

/// <summary>
/// Token embedding with optional scaling.
/// Python: Embeddings class in rec_nrtr_head.py
/// </summary>
internal sealed class NRTREmbeddings : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _lut;
    private readonly float _dModelSqrt;

    public NRTREmbeddings(int dModel, int vocab, int paddingIdx = 0, bool scaleEmbedding = true)
        : base(nameof(NRTREmbeddings))
    {
        _lut = Embedding(vocab, dModel, padding_idx: paddingIdx);
        _dModelSqrt = scaleEmbedding ? MathF.Sqrt(dModel) : 1.0f;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _lut.call(input) * _dModelSqrt;
    }
}

/// <summary>
/// Sinusoidal positional encoding.
/// Python: PositionalEncoding in rec_nrtr_head.py
/// </summary>
internal sealed class PositionalEncoding : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Dropout _dropout;
    private Tensor _pe; // [1, maxLen, dim]

    public PositionalEncoding(float dropout = 0.1f, int dim = 512, int maxLen = 5000)
        : base(nameof(PositionalEncoding))
    {
        _dropout = nn.Dropout(dropout);

        // Compute sinusoidal positional encoding
        using var position = torch.arange(maxLen, ScalarType.Float32).unsqueeze(1);
        using var divTerm = torch.exp(
            torch.arange(0, dim, 2, ScalarType.Float32) * (-MathF.Log(10000.0f) / dim));
        var pe = torch.zeros(maxLen, dim);
        pe[.., TensorIndex.Slice(0, null, 2)] = torch.sin(position * divTerm);
        pe[.., TensorIndex.Slice(1, null, 2)] = torch.cos(position * divTerm);
        _pe = pe.unsqueeze(0); // [1, maxLen, dim]

        // Register as buffer (not a parameter)
        register_buffer("_pe", _pe);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var seqLen = (int)input.shape[1];
        var x = input + _pe[.., ..seqLen, ..].to(input.device);
        return _dropout.call(x);
    }
}

/// <summary>
/// Transformer encoder: stack of self-attention blocks.
/// Only used when num_encoder_layers > 0.
/// </summary>
internal sealed class TransformerEncoder : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.ModuleList<NRTRTransformerBlock> _layers;

    public TransformerEncoder(int dModel, int numHeads, int numLayers) : base(nameof(TransformerEncoder))
    {
        _layers = new TorchSharp.Modules.ModuleList<NRTRTransformerBlock>();
        for (var i = 0; i < numLayers; i++)
        {
            _layers.Add(new NRTRTransformerBlock(dModel, numHeads, dModel * 4, withSelfAttn: true, withCrossAttn: false));
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input;
        foreach (var layer in _layers)
        {
            x = layer.Forward(x, memory: null, selfMask: null);
        }
        return x;
    }
}

/// <summary>
/// Transformer decoder: stack of self-attention + cross-attention blocks.
/// </summary>
internal sealed class NRTRDecoder : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.ModuleList<NRTRTransformerBlock> _layers;

    public NRTRDecoder(int dModel, int numHeads, int numLayers) : base(nameof(NRTRDecoder))
    {
        _layers = new TorchSharp.Modules.ModuleList<NRTRTransformerBlock>();
        for (var i = 0; i < numLayers; i++)
        {
            _layers.Add(new NRTRTransformerBlock(dModel, numHeads, dModel * 4, withSelfAttn: true, withCrossAttn: true));
        }
        RegisterComponents();
    }

    public Tensor Forward(Tensor tgt, Tensor memory, Tensor? selfMask = null)
    {
        var x = tgt;
        foreach (var layer in _layers)
        {
            x = layer.Forward(x, memory, selfMask);
        }
        return x;
    }

    public override Tensor forward(Tensor input)
    {
        throw new InvalidOperationException("Use Forward(tgt, memory, selfMask) instead.");
    }
}

/// <summary>
/// Unified transformer block with optional self-attention, optional cross-attention, and FFN.
/// Python: TransformerBlock in rec_nrtr_head.py
/// Pre-norm architecture.
/// </summary>
internal sealed class NRTRTransformerBlock : Module<Tensor, Tensor>
{
    private readonly MultiHeadAttentionNRTR? _selfAttn;
    private readonly MultiHeadAttentionNRTR? _crossAttn;
    private readonly Module<Tensor, Tensor> _ffn;
    private readonly Module<Tensor, Tensor> _norm1;
    private readonly Module<Tensor, Tensor>? _norm2;
    private readonly Module<Tensor, Tensor> _normFfn;

    public NRTRTransformerBlock(int dModel, int numHeads, int dimFeedforward,
        bool withSelfAttn = true, bool withCrossAttn = false,
        float attentionDropout = 0.0f, float residualDropout = 0.1f)
        : base(nameof(NRTRTransformerBlock))
    {
        _selfAttn = withSelfAttn ? new MultiHeadAttentionNRTR(dModel, numHeads, attentionDropout) : null;
        _crossAttn = withCrossAttn ? new MultiHeadAttentionNRTR(dModel, numHeads, attentionDropout) : null;
        _ffn = Sequential(
            Linear(dModel, dimFeedforward),
            ReLU(),
            nn.Dropout(residualDropout),
            Linear(dimFeedforward, dModel),
            nn.Dropout(residualDropout));
        _norm1 = LayerNorm(dModel);
        _norm2 = withCrossAttn ? LayerNorm(dModel) : null;
        _normFfn = LayerNorm(dModel);
        RegisterComponents();
    }

    public Tensor Forward(Tensor input, Tensor? memory, Tensor? selfMask)
    {
        var x = input;

        // Self-attention with residual
        if (_selfAttn is not null)
        {
            var normed = _norm1.call(x);
            var selfAttnOut = _selfAttn.Forward(normed, normed, normed, selfMask);
            x = x + selfAttnOut;
        }

        // Cross-attention with residual (decoder only)
        if (_crossAttn is not null && memory is not null && _norm2 is not null)
        {
            var normed = _norm2.call(x);
            var crossAttnOut = _crossAttn.Forward(normed, memory, memory, mask: null);
            x = x + crossAttnOut;
        }

        // FFN with residual
        var normedFfn = _normFfn.call(x);
        var ffnOut = _ffn.call(normedFfn);
        x = x + ffnOut;

        return x;
    }

    public override Tensor forward(Tensor input)
    {
        throw new InvalidOperationException("Use Forward(input, memory, selfMask) instead.");
    }
}

/// <summary>
/// Multi-head attention supporting Q, K, V from different sources (for cross-attention).
/// Python: MultiHeadAttention in rec_nrtr_head.py
/// </summary>
internal sealed class MultiHeadAttentionNRTR : Module<Tensor, Tensor>
{
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _scale;
    private readonly TorchSharp.Modules.Linear _wQ;
    private readonly TorchSharp.Modules.Linear _wK;
    private readonly TorchSharp.Modules.Linear _wV;
    private readonly TorchSharp.Modules.Linear _outProj;
    private readonly TorchSharp.Modules.Dropout _attnDropout;

    public MultiHeadAttentionNRTR(int dModel, int numHeads, float dropoutRate = 0.0f)
        : base(nameof(MultiHeadAttentionNRTR))
    {
        _numHeads = numHeads;
        _headDim = dModel / numHeads;
        _scale = 1.0f / MathF.Sqrt(_headDim);

        _wQ = nn.Linear(dModel, dModel);
        _wK = nn.Linear(dModel, dModel);
        _wV = nn.Linear(dModel, dModel);
        _outProj = nn.Linear(dModel, dModel);
        _attnDropout = nn.Dropout(dropoutRate);

        RegisterComponents();
    }

    public Tensor Forward(Tensor query, Tensor key, Tensor value, Tensor? mask = null)
    {
        var b = query.shape[0];
        var tgtLen = query.shape[1];
        var srcLen = key.shape[1];

        var q = _wQ.call(query).reshape(b, tgtLen, _numHeads, _headDim).permute(0, 2, 1, 3) * _scale;
        var k = _wK.call(key).reshape(b, srcLen, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = _wV.call(value).reshape(b, srcLen, _numHeads, _headDim).permute(0, 2, 1, 3);

        var attn = torch.matmul(q, k.transpose(-2, -1)); // [B, heads, tgtLen, srcLen]

        if (mask is not null)
        {
            attn = attn + mask;
        }

        attn = functional.softmax(attn, dim: -1);
        attn = _attnDropout.call(attn);

        var output = torch.matmul(attn, v); // [B, heads, tgtLen, headDim]
        output = output.permute(0, 2, 1, 3).reshape(b, tgtLen, -1); // [B, tgtLen, dModel]
        return _outProj.call(output);
    }

    public override Tensor forward(Tensor input)
    {
        return Forward(input, input, input, mask: null);
    }
}
