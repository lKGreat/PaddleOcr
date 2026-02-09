using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// CPPDHead：CPPD 的 head，类似 NRTR。
/// </summary>
public sealed class CPPDHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly NRTRHead _nrtrHead;

    public CPPDHead(int inChannels, int outChannels, int hiddenSize = 512, int maxLen = 25) : base(nameof(CPPDHead))
    {
        _nrtrHead = new NRTRHead(inChannels, outChannels, hiddenSize, numHeads: 8, numEncoderLayers: 3, numDecoderLayers: 3, maxLen: maxLen);
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
