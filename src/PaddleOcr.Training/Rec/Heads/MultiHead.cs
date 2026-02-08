using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// Multi-head rec head: CTC + optional GTC branch (NRTR/SAR/Attention).
/// GTC branch prefers backbone feature when provided in targets["backbone_feat"].
/// </summary>
public sealed class MultiHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly CTCHead _ctcHead;
    private readonly Module<Tensor, Tensor>? _gtcHeadModule;
    private readonly IRecHead? _gtcHead;
    private readonly int _gtcInChannels;

    public MultiHead(
        int inChannels,
        int outChannelsCtc,
        int outChannelsGtc = 0,
        int hiddenSize = 48,
        int maxLen = 25,
        string? gtcHeadName = null,
        int gtcInChannels = 0)
        : base(nameof(MultiHead))
    {
        _gtcInChannels = gtcInChannels > 0 ? gtcInChannels : inChannels;
        _ctcHead = new CTCHead(inChannels, outChannelsCtc);
        if (outChannelsGtc > 0)
        {
            _gtcHeadModule = BuildGtcHead(gtcHeadName, _gtcInChannels, outChannelsGtc, hiddenSize, maxLen);
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
            var gtcInput = PrepareGtcInput(input, targets, _gtcInChannels);
            var gtcOut = _gtcHead.Forward(gtcInput, targets);
            result["gtc"] = gtcOut["predict"];
        }

        result["predict"] = result["ctc"];
        return result;
    }

    private static Tensor PrepareGtcInput(Tensor headInput, Dictionary<string, Tensor>? targets, int expectedChannels)
    {
        var source = headInput;
        if (targets is not null && targets.TryGetValue("backbone_feat", out var backboneFeat))
        {
            source = backboneFeat;
        }

        if (source.shape.Length == 4)
        {
            // [B,C,H,W] -> [B,H*W,C]
            source = source.flatten(2).transpose(1, 2);
        }

        if (source.shape.Length != 3)
        {
            throw new InvalidOperationException($"MultiHead GTC branch expects rank-3/4 feature, but got rank-{source.shape.Length}");
        }

        if (source.shape[2] != expectedChannels)
        {
            // Keep strict to surface config mismatch early.
            throw new InvalidOperationException(
                $"MultiHead GTC feature channel mismatch: expected C={expectedChannels}, got C={source.shape[2]}. " +
                "Check backbone/head_list configuration.");
        }

        return source;
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
