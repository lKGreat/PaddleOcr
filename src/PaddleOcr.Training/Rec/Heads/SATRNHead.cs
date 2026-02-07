using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// SATRNHead：Spatial Attention Transformer，类似 SAR。
/// </summary>
public sealed class SATRNHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly SARHead _sarHead;

    public SATRNHead(int inChannels, int outChannels, int hiddenSize = 512, int maxLen = 25) : base(nameof(SATRNHead))
    {
        _sarHead = new SARHead(inChannels, outChannels, hiddenSize, maxLen);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _sarHead.forward(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        return _sarHead.Forward(input, targets);
    }
}
