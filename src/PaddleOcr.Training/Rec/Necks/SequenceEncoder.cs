using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

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
/// EncoderWithSVTR: lightweight token-mixing block over flattened [H*W] tokens.
/// Supports both simple FFN mode (for backward compatibility) and advanced mode with dims/depth/hiddenDims.
/// Input/Output are both [B, C, H, W] (unless dims is specified, then output channels = dims).
///
/// When dims > 0: applies channel reduction (inChannels → dims) and stacks depth blocks.
/// When dims == 0: uses legacy simple FFN (inChannels → inChannels).
/// </summary>
internal sealed class EncoderWithSVTR : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor>? _block;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>? _blocks;
    private readonly bool _isSimpleMode;
    public int OutChannels { get; }

    /// <summary>
    /// Legacy constructor for simple FFN mode (backward compatible).
    /// </summary>
    public EncoderWithSVTR(int inChannels) : base(nameof(EncoderWithSVTR))
    {
        OutChannels = inChannels;
        _isSimpleMode = true;
        _block = Sequential(
            LayerNorm(inChannels),
            Linear(inChannels, inChannels * 2),
            GELU(),
            Linear(inChannels * 2, inChannels));
        RegisterComponents();
    }

    /// <summary>
    /// Enhanced constructor with channel reduction and depth.
    /// </summary>
    /// <param name="inChannels">Input channels from backbone/neck</param>
    /// <param name="dims">Output dimension; if 0, behaves like legacy mode</param>
    /// <param name="depth">Number of stacked blocks (each block: LayerNorm + FFN)</param>
    /// <param name="hiddenDims">FFN hidden dimension (inner expansion)</param>
    public EncoderWithSVTR(int inChannels, int dims = 0, int depth = 1, int hiddenDims = 0)
        : base(nameof(EncoderWithSVTR))
    {
        if (dims <= 0)
        {
            // Legacy mode
            OutChannels = inChannels;
            _isSimpleMode = true;
            _block = Sequential(
                LayerNorm(inChannels),
                Linear(inChannels, inChannels * 2),
                GELU(),
                Linear(inChannels * 2, inChannels));
            RegisterComponents();
        }
        else
        {
            // Advanced mode with channel reduction
            OutChannels = dims;
            _isSimpleMode = false;
            var effectiveHiddenDims = hiddenDims > 0 ? hiddenDims : dims * 2;

            // Initial projection: inChannels → dims
            _block = Linear(inChannels, dims);

            // Stack depth blocks
            _blocks = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
            for (var i = 0; i < depth; i++)
            {
                var block = Sequential(
                    LayerNorm(dims),
                    Linear(dims, effectiveHiddenDims),
                    GELU(),
                    Linear(effectiveHiddenDims, dims)
                );
                _blocks.Add(block);
            }

            RegisterComponents();
        }
    }

    public override Tensor forward(Tensor input)
    {
        if (input.shape.Length != 4)
        {
            throw new InvalidOperationException($"EncoderWithSVTR expects rank-4 input [B,C,H,W], but got rank-{input.shape.Length}");
        }

        var b = input.shape[0];
        var c = input.shape[1];
        var h = input.shape[2];
        var w = input.shape[3];

        var x = input.flatten(2).transpose(1, 2); // [B, H*W, C]

        if (_isSimpleMode)
        {
            // Legacy: simple FFN with residual
            var y = _block!.call(x);
            y = y + x;
            return y.transpose(1, 2).reshape(b, c, h, w);
        }
        else
        {
            // Advanced: initial projection + stacked blocks
            var y = _block!.call(x); // [B, H*W, dims]

            if (_blocks != null)
            {
                foreach (var block in _blocks)
                {
                    var blockOut = block.call(y);
                    y = y + blockOut; // Residual connection
                }
            }

            // Return [B, dims, H, W]
            return y.transpose(1, 2).reshape(b, OutChannels, h, w);
        }
    }
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
