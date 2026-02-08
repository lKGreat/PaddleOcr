using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using PaddleOcr.Training.Rec.Necks;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// Config for CTC branch internal neck encoder (SVTR or other).
/// Includes all SVTR-specific parameters.
/// </summary>
public record MultiHeadCtcNeckConfig(
    string EncoderType,
    int Dims,
    int Depth,
    int HiddenDims,
    bool UseGuide = false,
    int NumHeads = 8,
    bool QkvBias = true,
    float MlpRatio = 2.0f,
    float DropRate = 0.1f,
    float AttnDropRate = 0.1f,
    float DropPath = 0.0f,
    int[]? KernelSize = null);

/// <summary>
/// Multi-head rec head: CTC + optional GTC branch (NRTR/SAR/Attention).
/// CTC branch can optionally have an internal SequenceEncoder (e.g., SVTR).
/// GTC branch uses FCTranspose for channel projection, then head.
/// </summary>
public sealed class MultiHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly SequenceEncoder? _ctcEncoder;
    private readonly CTCHead _ctcHead;
    private readonly Module<Tensor, Tensor>? _beforeGtc;  // FCTranspose
    private readonly Module<Tensor, Tensor>? _gtcHeadModule;
    private readonly IRecHead? _gtcHead;
    private readonly int _gtcInChannels;
    private readonly int _nrtrDim;

    /// <summary>
    /// Legacy constructor (backward compatible, no internal CTC encoder).
    /// </summary>
    public MultiHead(
        int inChannels,
        int outChannelsCtc,
        int outChannelsGtc = 0,
        int hiddenSize = 48,
        int maxLen = 25,
        string? gtcHeadName = null,
        int gtcInChannels = 0)
        : this(inChannels, outChannelsCtc, outChannelsGtc, hiddenSize, maxLen, gtcHeadName, gtcInChannels, null, 0)
    {
    }

    /// <summary>
    /// Enhanced constructor with optional CTC internal encoder and GTC FCTranspose.
    /// </summary>
    public MultiHead(
        int inChannels,
        int outChannelsCtc,
        int outChannelsGtc = 0,
        int hiddenSize = 48,
        int maxLen = 25,
        string? gtcHeadName = null,
        int gtcInChannels = 0,
        MultiHeadCtcNeckConfig? ctcNeckConfig = null,
        int nrtrDim = 0)
        : base(nameof(MultiHead))
    {
        _gtcInChannels = gtcInChannels > 0 ? gtcInChannels : inChannels;
        _nrtrDim = nrtrDim;

        // CTC branch: optionally with internal encoder
        if (ctcNeckConfig is not null)
        {
            _ctcEncoder = new SequenceEncoder(
                inChannels,
                ctcNeckConfig.EncoderType,
                ctcNeckConfig.Dims,
                ctcNeckConfig.Depth,
                ctcNeckConfig.HiddenDims,
                ctcNeckConfig.UseGuide,
                ctcNeckConfig.NumHeads,
                ctcNeckConfig.QkvBias,
                ctcNeckConfig.MlpRatio,
                ctcNeckConfig.DropRate,
                ctcNeckConfig.AttnDropRate,
                ctcNeckConfig.DropPath,
                ctcNeckConfig.KernelSize);
            _ctcHead = new CTCHead(_ctcEncoder.OutChannels, outChannelsCtc);
        }
        else
        {
            _ctcHead = new CTCHead(inChannels, outChannelsCtc);
        }

        // GTC branch: with optional FCTranspose
        if (outChannelsGtc > 0)
        {
            if (nrtrDim > 0)
            {
                _beforeGtc = BuildFCTranspose(inChannels, nrtrDim);
            }

            _gtcHeadModule = BuildGtcHead(gtcHeadName, nrtrDim > 0 ? nrtrDim : _gtcInChannels, outChannelsGtc, hiddenSize, maxLen);
            _gtcHead = _gtcHeadModule as IRecHead;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var ctcInput = _ctcEncoder is not null ? _ctcEncoder.call(input) : input;
        return _ctcHead.forward(ctcInput);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var result = new Dictionary<string, Tensor>();

        // CTC branch: encoder (optional) -> head
        var ctcInput = _ctcEncoder is not null ? _ctcEncoder.call(input) : input;
        var ctcOut = _ctcHead.Forward(ctcInput, targets);
        result["ctc"] = ctcOut["predict"];

        // GTC branch: FCTranspose (optional) -> head
        if (_gtcHead is not null)
        {
            var gtcFeat = input;
            if (targets is not null && targets.TryGetValue("backbone_feat", out var backboneFeat))
            {
                gtcFeat = backboneFeat;
            }

            var gtcInput = gtcFeat;
            if (_beforeGtc is not null)
            {
                // Apply FCTranspose
                gtcInput = _beforeGtc.call(gtcFeat);
            }
            else
            {
                // Legacy: prepare as before
                gtcInput = PrepareGtcInput(gtcFeat, _gtcInChannels);
            }

            var gtcOut = _gtcHead.Forward(gtcInput, targets);
            result["gtc"] = gtcOut["predict"];
        }

        result["predict"] = result["ctc"];
        return result;
    }

    /// <summary>
    /// Build FCTranspose: Flatten(2) + Linear projection + Transpose.
    /// Input: [B, C, H, W] -> Output: [B, seq_len, nrtrDim]
    /// </summary>
    private static Module<Tensor, Tensor> BuildFCTranspose(int inChannels, int nrtrDim)
    {
        return new FCTranspose(inChannels, nrtrDim);
    }

    private static Tensor PrepareGtcInput(Tensor source, int expectedChannels)
    {
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

/// <summary>
/// FCTranspose: Flatten(2) + Linear projection + optional Transpose.
/// Used in MultiHead GTC branch to project backbone features to nrtrDim.
/// Input: [B, C, H, W] -> Output: [B, seq_len, nrtrDim]
/// </summary>
internal sealed class FCTranspose : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _linear;

    public FCTranspose(int inChannels, int outChannels) : base(nameof(FCTranspose))
    {
        _linear = Linear(inChannels, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Input: [B, C, H, W]
        if (input.shape.Length == 4)
        {
            var b = input.shape[0];
            var c = input.shape[1];
            var h = input.shape[2];
            var w = input.shape[3];

            // [B, C, H, W] -> [B, H*W, C]
            var flattened = input.flatten(2).transpose(1, 2);

            // [B, H*W, C] -> [B, H*W, out]
            return _linear.call(flattened);
        }

        if (input.shape.Length == 3)
        {
            // Already flattened: [B, seq_len, C] -> [B, seq_len, out]
            return _linear.call(input);
        }

        throw new InvalidOperationException($"FCTranspose expects rank-3/4 input, got rank-{input.shape.Length}");
    }
}
