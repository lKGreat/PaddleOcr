using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Necks;

/// <summary>
/// SequenceEncoder：核心 Neck，将 backbone 输出转换为序列特征。
/// 支持多种编码方式：reshape, rnn, fc, svtr, cascadernn。
/// </summary>
public sealed class SequenceEncoder : Module<Tensor, Tensor>, IRecNeck
{
    private readonly Module<Tensor, Tensor> _encoder;
    public int OutChannels { get; }

    public SequenceEncoder(int inChannels, string encoderType = "rnn", int hiddenSize = 48) : base(nameof(SequenceEncoder))
    {
        switch (encoderType.ToLowerInvariant())
        {
            case "rnn":
                _encoder = new EncoderWithRNN(inChannels, hiddenSize);
                OutChannels = hiddenSize * 2; // 双向
                break;
            case "fc":
                _encoder = new EncoderWithFC(inChannels, hiddenSize);
                OutChannels = hiddenSize;
                break;
            case "svtr":
                _encoder = new EncoderWithSVTR(inChannels);
                OutChannels = inChannels;
                break;
            case "cascadernn":
                _encoder = new EncoderWithCascadeRNN(inChannels, hiddenSize);
                OutChannels = hiddenSize * 2;
                break;
            default: // reshape
                _encoder = new Im2Seq();
                OutChannels = inChannels;
                break;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // 输入 [B, C, H, W] -> 先做 Im2Seq 转换
        var shape = input.shape;
        var b = shape[0];
        var c = shape[1];
        var w = shape[3];

        // squeeze height -> [B, C, W] -> [B, W, C]
        using var squeezed = input.squeeze(2);
        var x = squeezed.permute(0, 2, 1); // [B, W, C]

        return _encoder.call(x);
    }
}

/// <summary>
/// Im2Seq：仅做 reshape，[B,C,1,W] -> [B,W,C]。
/// </summary>
internal sealed class Im2Seq : Module<Tensor, Tensor>
{
    public Im2Seq() : base(nameof(Im2Seq))
    {
    }

    public override Tensor forward(Tensor input)
    {
        return input; // 已在 SequenceEncoder.forward 中完成 reshape
    }
}

/// <summary>
/// EncoderWithRNN：双向 LSTM 编码器。
/// </summary>
internal sealed class EncoderWithRNN : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.LSTM _lstm;
    private readonly int _hiddenSize;

    public EncoderWithRNN(int inChannels, int hiddenSize) : base(nameof(EncoderWithRNN))
    {
        _hiddenSize = hiddenSize;
        _lstm = nn.LSTM(inChannels, hiddenSize, numLayers: 2, bidirectional: true, batchFirst: true);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var (output, _, _) = _lstm.call(input);
        return output; // [B, W, hiddenSize*2]
    }
}

/// <summary>
/// EncoderWithFC：Linear 投影编码器。
/// </summary>
internal sealed class EncoderWithFC : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fc;

    public EncoderWithFC(int inChannels, int hiddenSize) : base(nameof(EncoderWithFC))
    {
        _fc = Sequential(
            Linear(inChannels, hiddenSize),
            ReLU()
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _fc.call(input); // [B, W, hiddenSize]
    }
}

/// <summary>
/// EncoderWithSVTR：SVTR 编码块。
/// </summary>
internal sealed class EncoderWithSVTR : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _block;

    public EncoderWithSVTR(int inChannels) : base(nameof(EncoderWithSVTR))
    {
        _block = Sequential(
            LayerNorm(inChannels),
            Linear(inChannels, inChannels * 2),
            GELU(),
            Linear(inChannels * 2, inChannels)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var residual = input;
        var x = _block.call(input);
        return x + residual;
    }
}

/// <summary>
/// EncoderWithCascadeRNN：多层双向 LSTM 编码器。
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
        return output; // [B, W, hiddenSize*2]
    }
}
