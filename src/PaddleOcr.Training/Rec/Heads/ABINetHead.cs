using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// ABINetHead：ABINet 的 head，包含视觉和语言分支。
/// </summary>
public sealed class ABINetHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _visualHead;
    private readonly Module<Tensor, Tensor> _languageHead;
    private readonly Module<Tensor, Tensor> _fusionHead;
    private readonly int _outChannels;

    public ABINetHead(int inChannels, int outChannels, int hiddenSize = 256) : base(nameof(ABINetHead))
    {
        _outChannels = outChannels;
        _visualHead = Linear(inChannels, outChannels);
        _languageHead = Sequential(
            Linear(inChannels, hiddenSize),
            ReLU(),
            Linear(hiddenSize, outChannels)
        );
        _fusionHead = Sequential(
            Linear(outChannels * 2, hiddenSize),
            ReLU(),
            Linear(hiddenSize, outChannels)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var visualOut = _visualHead.call(input); // [B, W, outChannels]
        var langOut = _languageHead.call(input); // [B, W, outChannels]
        var fused = cat(new[] { visualOut, langOut }, dim: -1);
        return _fusionHead.call(fused); // [B, W, outChannels]
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
