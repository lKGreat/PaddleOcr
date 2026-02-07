using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// SPINAttentionHead：SPIN 的 attention head，类似 AttentionHead。
/// </summary>
public sealed class SPINAttentionHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly AttentionHead _attnHead;

    public SPINAttentionHead(int inChannels, int outChannels, int hiddenSize = 256, int maxLen = 25) : base(nameof(SPINAttentionHead))
    {
        _attnHead = new AttentionHead(inChannels, outChannels, hiddenSize, maxLen);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _attnHead.forward(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        return _attnHead.Forward(input, targets);
    }
}
