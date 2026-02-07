using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// LaTeXOCRHead：LaTeXOCR 的 head，类似 NRTR。
/// </summary>
public sealed class LaTeXOCRHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly NRTRHead _nrtrHead;

    public LaTeXOCRHead(int inChannels, int outChannels, int hiddenSize = 512, int maxLen = 100) : base(nameof(LaTeXOCRHead))
    {
        _nrtrHead = new NRTRHead(inChannels, outChannels, hiddenSize, numHeads: 8, numLayers: 3, maxLen);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _nrtrHead.forward(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        return _nrtrHead.Forward(input, targets);
    }
}
