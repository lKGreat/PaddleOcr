using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// UniMERNetHead：UniMERNet 的 head，类似 NRTR。
/// </summary>
public sealed class UniMERNetHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly NRTRHead _nrtrHead;

    public UniMERNetHead(int inChannels, int outChannels, int hiddenSize = 512, int maxLen = 100) : base(nameof(UniMERNetHead))
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
