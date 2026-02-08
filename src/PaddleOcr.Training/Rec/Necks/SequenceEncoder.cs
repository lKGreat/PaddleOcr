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

    public SequenceEncoder(int inChannels, string encoderType = "rnn", int hiddenSize = 48) : base(nameof(SequenceEncoder))
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
                _encoder = new EncoderWithSVTR(inChannels);
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
/// Input/Output are both [B, C, H, W].
/// </summary>
internal sealed class EncoderWithSVTR : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _block;
    public int OutChannels { get; }

    public EncoderWithSVTR(int inChannels) : base(nameof(EncoderWithSVTR))
    {
        OutChannels = inChannels;
        _block = Sequential(
            LayerNorm(inChannels),
            Linear(inChannels, inChannels * 2),
            GELU(),
            Linear(inChannels * 2, inChannels));
        RegisterComponents();
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
        var y = _block.call(x);
        y = y + x;
        return y.transpose(1, 2).reshape(b, c, h, w);
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
