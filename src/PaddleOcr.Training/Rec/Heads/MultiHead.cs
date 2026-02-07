using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// Multi-head rec head: CTC + optional GTC branch (NRTR/SAR/Attention).
/// </summary>
public sealed class MultiHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly CTCHead _ctcHead;
    private readonly Module<Tensor, Tensor>? _gtcHeadModule;
    private readonly IRecHead? _gtcHead;

    public MultiHead(
        int inChannels,
        int outChannelsCtc,
        int outChannelsGtc = 0,
        int hiddenSize = 48,
        int maxLen = 25,
        string? gtcHeadName = null)
        : base(nameof(MultiHead))
    {
        _ctcHead = new CTCHead(inChannels, outChannelsCtc);
        if (outChannelsGtc > 0)
        {
            _gtcHeadModule = BuildGtcHead(gtcHeadName, inChannels, outChannelsGtc, hiddenSize, maxLen);
            _gtcHead = _gtcHeadModule as IRecHead;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _ctcHead.forward(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var result = new Dictionary<string, Tensor>();
        var ctcOut = _ctcHead.Forward(input, targets);
        result["ctc"] = ctcOut["predict"];

        if (_gtcHead is not null)
        {
            var gtcOut = _gtcHead.Forward(input, targets);
            result["gtc"] = gtcOut["predict"];
        }

        result["predict"] = result["ctc"];
        return result;
    }

    private static Module<Tensor, Tensor> BuildGtcHead(
        string? gtcHeadName,
        int inChannels,
        int outChannels,
        int hiddenSize,
        int maxLen)
    {
        var normalized = (gtcHeadName ?? string.Empty).ToLowerInvariant();
        return normalized switch
        {
            "nrtr" or "nrtrhead" => new NRTRHead(inChannels, outChannels, hiddenSize, maxLen: maxLen),
            "sar" or "sarhead" => new SARHead(inChannels, outChannels, hiddenSize, maxLen),
            "attn" or "attention" or "attentionhead" => new AttentionHead(inChannels, outChannels, hiddenSize, maxLen),
            _ => new AttentionHead(inChannels, outChannels, hiddenSize, maxLen)
        };
    }
}
