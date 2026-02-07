using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// RFLHead：RFL 的 head，支持文本解码和长度预测。
/// </summary>
public sealed class RFLHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _textHead;
    private readonly Module<Tensor, Tensor> _lengthHead;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public RFLHead(int inChannels, int outChannels, int maxLen = 25) : base(nameof(RFLHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;
        _textHead = Linear(inChannels, outChannels);
        _lengthHead = Linear(inChannels, maxLen + 1);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _textHead.call(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        return new Dictionary<string, Tensor>
        {
            ["predict"] = _textHead.call(input),
            ["length"] = _lengthHead.call(input)
        };
    }
}
