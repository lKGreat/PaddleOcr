using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// ParseQHead：ParseQ 的 head。
/// </summary>
public sealed class ParseQHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _head;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public ParseQHead(int inChannels, int outChannels, int maxLen = 25) : base(nameof(ParseQHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;
        _head = Sequential(
            Linear(inChannels, inChannels * 2),
            ReLU(),
            Linear(inChannels * 2, outChannels)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C] -> reshape to [B, maxLen, C] if needed
        var shape = input.shape;
        if (shape[1] != _maxLen)
        {
            input = functional.adaptive_avg_pool1d(input.permute(0, 2, 1), _maxLen).permute(0, 2, 1);
        }

        return _head.call(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
