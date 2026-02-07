using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// CTC Head：Linear -> vocab_size，可选 2 层 FC。
/// </summary>
public sealed class CTCHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _fc;
    private readonly int _outChannels;

    public CTCHead(int inChannels, int outChannels, int midChannels = 0) : base(nameof(CTCHead))
    {
        _outChannels = outChannels;
        if (midChannels > 0)
        {
            _fc = Sequential(
                Linear(inChannels, midChannels),
                ReLU(),
                Linear(midChannels, outChannels)
            );
        }
        else
        {
            _fc = Linear(inChannels, outChannels);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C] -> [B, W, outChannels]
        var logits = _fc.call(input);
        if (!training)
        {
            logits = functional.softmax(logits, dim: -1);
        }

        return logits;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
