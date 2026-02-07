using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// PRENHead：PREN 的 head。
/// </summary>
public sealed class PRENHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _head;
    private readonly int _outChannels;

    public PRENHead(int inChannels, int outChannels) : base(nameof(PRENHead))
    {
        _outChannels = outChannels;
        _head = Sequential(
            Linear(inChannels, inChannels),
            ReLU(),
            Linear(inChannels, outChannels)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _head.call(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
