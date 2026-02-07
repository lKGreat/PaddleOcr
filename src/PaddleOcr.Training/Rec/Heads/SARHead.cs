using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// SARHead：SAREncoder + ParallelSARDecoder (2D attention)。
/// 用于 SAR 算法。
/// </summary>
public sealed class SARHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly SAREncoder _encoder;
    private readonly ParallelSARDecoder _decoder;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public SARHead(int inChannels, int outChannels, int hiddenSize = 512, int maxLen = 25) : base(nameof(SARHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;
        _encoder = new SAREncoder(inChannels, hiddenSize);
        _decoder = new ParallelSARDecoder(hiddenSize, outChannels, maxLen);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var encoded = _encoder.call(input); // [B, W, hiddenSize]
        var decoded = _decoder.call(encoded); // [B, maxLen, outChannels]
        return decoded;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}

/// <summary>
/// SAREncoder：2D attention 编码器。
/// </summary>
internal sealed class SAREncoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _proj;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;

    public SAREncoder(int inChannels, int hiddenSize, int numLayers = 2) : base(nameof(SAREncoder))
    {
        _proj = Linear(inChannels, hiddenSize);
        _layers = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < numLayers; i++)
        {
            _layers.Add(new SAREncoderLayer(hiddenSize));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var x = _proj.call(input); // [B, W, hiddenSize]
        foreach (var layer in _layers)
        {
            x = layer.call(x);
        }

        return x;
    }
}

/// <summary>
/// SAREncoderLayer：带 2D attention 的编码层。
/// </summary>
internal sealed class SAREncoderLayer : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _attn;
    private readonly Module<Tensor, Tensor> _ffn;

    public SAREncoderLayer(int hiddenSize) : base(nameof(SAREncoderLayer))
    {
        _attn = new TwoDAttention(hiddenSize);
        _ffn = Sequential(
            Linear(hiddenSize, hiddenSize * 4),
            ReLU(),
            Linear(hiddenSize * 4, hiddenSize)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var residual = input;
        var x = _attn.call(input);
        x = x + residual;
        using var residual2 = x;
        x = _ffn.call(x);
        return x + residual2;
    }
}

/// <summary>
/// TwoDAttention：2D 自注意力机制。
/// </summary>
internal sealed class TwoDAttention : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _qProj;
    private readonly Module<Tensor, Tensor> _kProj;
    private readonly Module<Tensor, Tensor> _vProj;
    private readonly Module<Tensor, Tensor> _outProj;
    private readonly int _hiddenSize;

    public TwoDAttention(int hiddenSize) : base(nameof(TwoDAttention))
    {
        _hiddenSize = hiddenSize;
        _qProj = Linear(hiddenSize, hiddenSize);
        _kProj = Linear(hiddenSize, hiddenSize);
        _vProj = Linear(hiddenSize, hiddenSize);
        _outProj = Linear(hiddenSize, hiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, hiddenSize]
        var q = _qProj.call(input);
        var k = _kProj.call(input);
        var v = _vProj.call(input);

        var scale = Math.Sqrt(_hiddenSize);
        using var scores = torch.bmm(q, k.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.bmm(attn, v);
        return _outProj.call(output);
    }
}

/// <summary>
/// ParallelSARDecoder：并行 SAR 解码器。
/// </summary>
internal sealed class ParallelSARDecoder : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _embedding;
    private readonly Module<Tensor, Tensor> _decoder;
    private readonly Module<Tensor, Tensor> _outputProj;
    private readonly int _maxLen;

    public ParallelSARDecoder(int hiddenSize, int outChannels, int maxLen) : base(nameof(ParallelSARDecoder))
    {
        _maxLen = maxLen;
        _embedding = Embedding(outChannels, hiddenSize);
        _decoder = Sequential(
            Linear(hiddenSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, hiddenSize)
        );
        _outputProj = Linear(hiddenSize, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, hiddenSize]
        var b = input.shape[0];
        var w = input.shape[1];
        var device = input.device;

        // 创建 SOS token 序列
        var sosIds = zeros(new long[] { b, _maxLen }, ScalarType.Int64, device: device);
        var sosEmb = _embedding.call(sosIds); // [B, maxLen, hiddenSize]

        // 使用编码器输出作为上下文
        using var context = functional.adaptive_avg_pool1d(input.permute(0, 2, 1), _maxLen).permute(0, 2, 1); // [B, maxLen, hiddenSize]

        var x = sosEmb + context;
        x = _decoder.call(x);
        return _outputProj.call(x); // [B, maxLen, outChannels]
    }
}
