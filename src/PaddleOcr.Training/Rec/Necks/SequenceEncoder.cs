using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace PaddleOcr.Training.Rec.Necks;

/// <summary>
/// SequenceEncoder: converts backbone feature map to sequence features.
/// Follows Paddle behavior:
/// - non-svtr: Im2Seq -> encoder
/// - svtr: encoder(4D) -> Im2Seq
/// </summary>
public sealed class SequenceEncoder : Module<Tensor, Tensor>, IRecNeck
{
    private readonly Im2Seq _encoderReshape;
    private readonly Module<Tensor, Tensor>? _encoder;
    private readonly string _encoderType;
    private readonly bool _onlyReshape;
    public int OutChannels { get; }

    /// <summary>
    /// Legacy constructor for simple encoders (rnn, fc, cascadernn, reshape).
    /// </summary>
    public SequenceEncoder(int inChannels, string encoderType = "rnn", int hiddenSize = 48)
        : this(inChannels, encoderType, dims: 0, depth: 1, hiddenDims: 0, hiddenSize: hiddenSize)
    {
    }

    /// <summary>
    /// Enhanced constructor supporting SVTR with dims/depth/hidden_dims.
    /// </summary>
    public SequenceEncoder(int inChannels, string encoderType = "rnn", int dims = 0, int depth = 1, int hiddenDims = 0, int hiddenSize = 48)
        : base(nameof(SequenceEncoder))
    {
        _encoderType = encoderType.ToLowerInvariant();
        _encoderReshape = new Im2Seq(inChannels);
        OutChannels = _encoderReshape.OutChannels;

        switch (_encoderType)
        {
            case "rnn":
                _encoder = new EncoderWithRNN(_encoderReshape.OutChannels, hiddenSize);
                OutChannels = hiddenSize * 2;
                _onlyReshape = false;
                break;
            case "fc":
                _encoder = new EncoderWithFC(_encoderReshape.OutChannels, hiddenSize);
                OutChannels = hiddenSize;
                _onlyReshape = false;
                break;
            case "svtr":
                // Enhanced SVTR with dims/depth support
                _encoder = dims > 0
                    ? new EncoderWithSVTR(inChannels, dims, depth, hiddenDims)
                    : new EncoderWithSVTR(inChannels);
                OutChannels = ((EncoderWithSVTR)_encoder).OutChannels;
                _onlyReshape = false;
                break;
            case "cascadernn":
                _encoder = new EncoderWithCascadeRNN(_encoderReshape.OutChannels, hiddenSize);
                OutChannels = hiddenSize * 2;
                _onlyReshape = false;
                break;
            default: // reshape
                _encoder = null;
                _onlyReshape = true;
                break;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        if (_encoderType != "svtr")
        {
            var x = _encoderReshape.call(input);
            if (!_onlyReshape && _encoder is not null)
            {
                x = _encoder.call(x);
            }

            return x;
        }

        // MultiHead internal CTC encoder may receive sequence features [B,T,C].
        // In that case SVTR works directly in sequence space and no Im2Seq is needed.
        if (input.shape.Length == 3)
        {
            return _encoder!.call(input);
        }

        var svtrFeat = _encoder!.call(input);
        return _encoderReshape.call(svtrFeat);
    }
}

/// <summary>
/// Im2Seq: [B, C, 1, W] -> [B, W, C]
/// </summary>
internal sealed class Im2Seq : Module<Tensor, Tensor>
{
    public int OutChannels { get; }

    public Im2Seq(int inChannels) : base(nameof(Im2Seq))
    {
        OutChannels = inChannels;
    }

    public override Tensor forward(Tensor input)
    {
        if (input.shape.Length != 4)
        {
            throw new InvalidOperationException($"Im2Seq expects rank-4 input [B,C,H,W], but got rank-{input.shape.Length}");
        }

        var h = input.shape[2];
        if (h != 1)
        {
            throw new InvalidOperationException($"Im2Seq expects feature height H=1, but got H={h}");
        }

        var squeezed = input.squeeze(2);
        return squeezed.permute(0, 2, 1);
    }
}

/// <summary>
/// EncoderWithRNN: 2-layer bidirectional LSTM.
/// </summary>
internal sealed class EncoderWithRNN : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.LSTM _lstm;

    public EncoderWithRNN(int inChannels, int hiddenSize) : base(nameof(EncoderWithRNN))
    {
        _lstm = nn.LSTM(inChannels, hiddenSize, numLayers: 2, bidirectional: true, batchFirst: true);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var (output, _, _) = _lstm.call(input);
        return output;
    }
}

/// <summary>
/// EncoderWithFC: linear projection encoder.
/// </summary>
internal sealed class EncoderWithFC : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fc;

    public EncoderWithFC(int inChannels, int hiddenSize) : base(nameof(EncoderWithFC))
    {
        _fc = Sequential(
            Linear(inChannels, hiddenSize),
            ReLU());
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _fc.call(input);
    }
}

/// <summary>
/// EncoderWithSVTR: Full SVTR implementation with Conv + Attention + MLP.
/// Matches the Python PaddleOCR-3.3.2 implementation exactly.
/// Input: [B, C, H, W], Output: [B, dims, H, W]
/// </summary>
internal sealed class EncoderWithSVTR : Module<Tensor, Tensor>
{
    private readonly ConvBNLayer _conv1;
    private readonly ConvBNLayer _conv2;
    private readonly TorchSharp.Modules.ModuleList<SVTRBlock> _svtrBlocks;
    private readonly LayerNorm _norm;
    private readonly ConvBNLayer _conv3;
    private readonly ConvBNLayer _conv4;
    private readonly ConvBNLayer _conv1x1;
    private readonly bool _useGuide;
    public int OutChannels { get; }

    public EncoderWithSVTR(
        int inChannels,
        int dims = 64,
        int depth = 2,
        int hiddenDims = 120,
        bool useGuide = false,
        int numHeads = 8,
        bool qkvBias = true,
        float mlpRatio = 2.0f,
        float dropRate = 0.1f,
        float attnDropRate = 0.1f,
        float dropPath = 0.0f,
        int[]? kernelSize = null)
        : base(nameof(EncoderWithSVTR))
    {
        kernelSize ??= new[] { 3, 3 };
        _useGuide = useGuide;
        OutChannels = dims;

        // Dimension reduction: inChannels → inChannels/8
        _conv1 = new ConvBNLayer(
            inChannels, inChannels / 8, kernelSize,
            new[] { kernelSize[0] / 2, kernelSize[1] / 2 });

        // inChannels/8 → hidden_dims
        _conv2 = new ConvBNLayer(inChannels / 8, hiddenDims, new[] { 1, 1 });

        // SVTR blocks (Attention)
        _svtrBlocks = new TorchSharp.Modules.ModuleList<SVTRBlock>();
        for (var i = 0; i < depth; i++)
        {
            _svtrBlocks.Add(new SVTRBlock(
                hiddenDims, numHeads, mlpRatio, qkvBias, attnDropRate,
                dropPath, dropRate));
        }

        // Output normalization
        _norm = LayerNorm(hiddenDims);

        // hidden_dims → inChannels
        _conv3 = new ConvBNLayer(hiddenDims, inChannels, new[] { 1, 1 });

        // Concat([input, conv3_out]) → inChannels/8
        _conv4 = new ConvBNLayer(
            2 * inChannels, inChannels / 8, kernelSize,
            new[] { kernelSize[0] / 2, kernelSize[1] / 2 });

        // inChannels/8 → dims
        _conv1x1 = new ConvBNLayer(inChannels / 8, dims, new[] { 1, 1 });

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Stop gradient for use_guide
        var z = _useGuide ? input.detach() : input;

        // Shortcut
        var h = z;

        // Dimension reduction
        z = _conv1.forward(z);
        z = _conv2.forward(z);

        // SVTR global blocks (Transformer)
        var B = z.shape[0];
        var C = z.shape[1];
        var H = z.shape[2];
        var W = z.shape[3];

        z = z.flatten(2).transpose(1, 2); // [B, H*W, C]

        foreach (var block in _svtrBlocks)
        {
            z = block.forward(z);
        }

        z = _norm.forward(z);

        // Reshape back to 4D
        z = z.transpose(1, 2).reshape(B, C, H, W);
        z = _conv3.forward(z);

        // Concatenate with shortcut
        z = torch.cat(new[] { h, z }, dim: 1);
        z = _conv4.forward(z);
        z = _conv1x1.forward(z);

        return z;
    }
}

/// <summary>
/// ConvBNLayer: Conv2D + BatchNorm2D + Swish activation.
/// </summary>
internal sealed class ConvBNLayer : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.Conv2d _conv;
    private readonly BatchNorm2d _norm;
    private readonly Module<Tensor, Tensor> _act;

    public ConvBNLayer(
        int inChannels,
        int outChannels,
        int[] kernelSize,
        int[]? padding = null,
        int stride = 1,
        bool useActivation = true)
        : base(nameof(ConvBNLayer))
    {
        padding ??= new[] { kernelSize[0] / 2, kernelSize[1] / 2 };

        _conv = Conv2d(inChannels, outChannels, kernel_size: kernelSize,
            stride: stride, padding: padding, bias: false);
        _norm = BatchNorm2d(outChannels);
        _act = useActivation ? SiLU() : Identity();

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _conv.forward(input);
        x = _norm.forward(x);
        x = _act.forward(x);
        return x;
    }
}

/// <summary>
/// SVTRBlock: Multi-head Self-Attention + MLP + Residual connections.
/// </summary>
internal sealed class SVTRBlock : Module<Tensor, Tensor>
{
    private readonly LayerNorm _norm1;
    private readonly MultiHeadAttention _attention;
    private readonly LayerNorm _norm2;
    private readonly Module<Tensor, Tensor> _mlp;

    public SVTRBlock(
        int dim,
        int numHeads = 8,
        float mlpRatio = 2.0f,
        bool qkvBias = true,
        float attnDropRate = 0.1f,
        float dropPathRate = 0.0f,
        float dropRate = 0.1f)
        : base(nameof(SVTRBlock))
    {
        _norm1 = LayerNorm(dim);
        _attention = new MultiHeadAttention(dim, numHeads, qkvBias, attnDropRate);
        _norm2 = LayerNorm(dim);

        var hiddenDim = (int)(dim * mlpRatio);
        _mlp = Sequential(
            Linear(dim, hiddenDim),
            SiLU(),
            Dropout(dropRate),
            Linear(hiddenDim, dim),
            Dropout(dropRate));

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input;

        // Attention block with residual
        var attnOut = _attention.forward(_norm1.forward(x));
        x = x + attnOut;

        // MLP block with residual
        var mlpOut = _mlp.forward(_norm2.forward(x));
        x = x + mlpOut;

        return x;
    }
}

/// <summary>
/// MultiHeadAttention: Standard scaled dot-product attention.
/// </summary>
internal sealed class MultiHeadAttention : Module<Tensor, Tensor>
{
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _scale;
    private readonly Linear _qkvProj;
    private readonly Linear _outProj;
    private readonly Dropout _attnDropout;
    private readonly Dropout _projDropout;

    public MultiHeadAttention(
        int dim,
        int numHeads = 8,
        bool qkvBias = true,
        float attnDropRate = 0.1f,
        float projDropRate = 0.1f)
        : base(nameof(MultiHeadAttention))
    {
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _scale = 1.0f / (float)Math.Sqrt(_headDim);

        _qkvProj = Linear(dim, dim * 3, bias: qkvBias);
        _outProj = Linear(dim, dim);
        _attnDropout = Dropout(attnDropRate);
        _projDropout = Dropout(projDropRate);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var B = input.shape[0];
        var N = input.shape[1];

        // Project to Q, K, V
        var qkv = _qkvProj.forward(input); // [B, N, 3*dim]
        qkv = qkv.reshape(B, N, 3, _numHeads, _headDim); // [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4); // [3, B, num_heads, N, head_dim]

        var q = qkv[0] * _scale;
        var k = qkv[1];
        var v = qkv[2];

        // Attention
        var attn = torch.matmul(q, k.transpose(-2, -1)); // [B, num_heads, N, N]
        attn = torch.softmax(attn, dim: -1);
        attn = _attnDropout.forward(attn);

        // Combine heads
        var x = torch.matmul(attn, v); // [B, num_heads, N, head_dim]
        x = x.permute(0, 2, 1, 3); // [B, N, num_heads, head_dim]
        x = x.reshape(B, N, -1); // [B, N, dim]

        // Output projection
        x = _outProj.forward(x);
        x = _projDropout.forward(x);

        return x;
    }
}

/// <summary>
/// Identity layer (no-op).
/// </summary>
internal sealed class Identity : Module<Tensor, Tensor>
{
    public Identity() : base(nameof(Identity)) { }

    public override Tensor forward(Tensor input) => input;
}

/// <summary>
/// EncoderWithCascadeRNN: two stacked bidirectional LSTMs.
/// </summary>
internal sealed class EncoderWithCascadeRNN : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.LSTM _lstm1;
    private readonly TorchSharp.Modules.LSTM _lstm2;

    public EncoderWithCascadeRNN(int inChannels, int hiddenSize) : base(nameof(EncoderWithCascadeRNN))
    {
        _lstm1 = nn.LSTM(inChannels, hiddenSize, bidirectional: true, batchFirst: true);
        _lstm2 = nn.LSTM(hiddenSize * 2, hiddenSize, bidirectional: true, batchFirst: true);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var (x, _, _) = _lstm1.call(input);
        var (output, _, _) = _lstm2.call(x);
        return output;
    }
}
