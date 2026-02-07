using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// VLHead：VisionLAN 的 head。
/// </summary>
public sealed class VLHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _lengthHead;
    private readonly Module<Tensor, Tensor> _charHead;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public VLHead(int inChannels, int outChannels, int maxLen = 25) : base(nameof(VLHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;
        _lengthHead = Linear(inChannels, maxLen + 1); // 长度预测
        _charHead = Linear(inChannels, outChannels); // 字符预测
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var lengthLogits = _lengthHead.call(input); // [B, W, maxLen+1]
        var charLogits = _charHead.call(input); // [B, W, outChannels]
        // 简化：返回字符 logits
        return charLogits;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor>
        {
            ["predict"] = logits,
            ["length"] = _lengthHead.call(input)
        };
    }
}
