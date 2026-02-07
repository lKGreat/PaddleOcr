using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// PPFormulaNetHead：PP-FormulaNet 的 head，类似 NRTR。
/// </summary>
public sealed class PPFormulaNetHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly NRTRHead _nrtrHead;

    public PPFormulaNetHead(int inChannels, int outChannels, int hiddenSize = 512, int maxLen = 200) : base(nameof(PPFormulaNetHead))
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
