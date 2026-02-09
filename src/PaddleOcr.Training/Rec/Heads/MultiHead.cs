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
/// 1:1 port of ppocr/modeling/heads/rec_multi_head.py MultiHead.
///
/// CTC branch: optional internal SequenceEncoder (e.g., SVTR) -> CTCHead.
/// GTC branch: FCTranspose (Flatten+transpose+Linear) -> optional AddPos -> head.
///
/// In eval mode, only the CTC branch is executed (matching Python behavior).
/// </summary>
public sealed class MultiHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly SequenceEncoder? _ctcEncoder;
    private readonly CTCHead _ctcHead;
    private readonly Module<Tensor, Tensor>? _beforeGtc;  // Sequential: Flatten(2) + FCTranspose + optional AddPos
    private readonly Module<Tensor, Tensor>? _gtcHeadModule;
    private readonly IRecHead? _gtcHead;
    private readonly int _gtcInChannels;
    private readonly int _nrtrDim;
    private readonly string _gtcType; // "sar" or "nrtr"

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
        int nrtrDim = 0,
        bool usePool = false,
        bool usePos = false,
        int numDecoderLayers = 4)
        : base(nameof(MultiHead))
    {
        _gtcInChannels = gtcInChannels > 0 ? gtcInChannels : inChannels;
        _nrtrDim = nrtrDim;
        _gtcType = "sar"; // default

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
            var normalizedGtcName = (gtcHeadName ?? string.Empty).ToLowerInvariant();

            if (normalizedGtcName.Contains("nrtr"))
            {
                _gtcType = "nrtr";
                var effectiveNrtrDim = nrtrDim > 0 ? nrtrDim : 256;

                // Python: self.before_gtc = Sequential(Flatten(2), FCTranspose(in_channels, nrtr_dim), [AddPos])
                if (usePos)
                {
                    _beforeGtc = Sequential(
                        new Flatten2D(),
                        new FCTranspose(inChannels, effectiveNrtrDim),
                        new AddPos(effectiveNrtrDim, 80));
                }
                else
                {
                    _beforeGtc = Sequential(
                        new Flatten2D(),
                        new FCTranspose(inChannels, effectiveNrtrDim));
                }

                // Python: self.gtc_head = Transformer(
                //     d_model=nrtr_dim, nhead=nrtr_dim//32,
                //     num_encoder_layers=-1, num_decoder_layers=num_decoder_layers,
                //     max_len=max_text_length, dim_feedforward=nrtr_dim*4,
                //     out_channels=out_channels_list["NRTRLabelDecode"])
                var nrtrHeads = effectiveNrtrDim / 32;
                if (nrtrHeads < 1) nrtrHeads = 1;

                _gtcHeadModule = new NRTRHead(
                    inChannels: effectiveNrtrDim,
                    outChannels: outChannelsGtc,
                    hiddenSize: effectiveNrtrDim,
                    numHeads: nrtrHeads,
                    numEncoderLayers: 0, // No encoder when used inside MultiHead
                    numDecoderLayers: numDecoderLayers,
                    maxLen: maxLen);
                _gtcHead = _gtcHeadModule as IRecHead;
            }
            else if (normalizedGtcName.Contains("sar"))
            {
                _gtcType = "sar";
                _gtcHeadModule = BuildGtcHead(gtcHeadName, _gtcInChannels, outChannelsGtc, hiddenSize, maxLen);
                _gtcHead = _gtcHeadModule as IRecHead;
            }
            else
            {
                // Fallback: attention head
                _gtcHeadModule = BuildGtcHead(gtcHeadName, nrtrDim > 0 ? nrtrDim : _gtcInChannels, outChannelsGtc, hiddenSize, maxLen);
                _gtcHead = _gtcHeadModule as IRecHead;
            }
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
        var ctcEncoder = _ctcEncoder is not null ? _ctcEncoder.call(input) : input;
        var ctcOut = _ctcHead.Forward(ctcEncoder, targets);
        result["ctc"] = ctcOut["predict"];
        result["ctc_neck"] = ctcEncoder; // Python: head_out["ctc_neck"] = ctc_encoder

        // Python: eval mode returns only CTC output
        if (!training)
        {
            result["predict"] = result["ctc"];
            return result;
        }

        // GTC branch: before_gtc -> head (training only)
        // Python: both CTC and GTC branches use x (head input) directly
        if (_gtcHead is not null)
        {
            Tensor gtcInput;

            if (_beforeGtc is not null)
            {
                // NRTR path: self.before_gtc(x) — transforms head input
                gtcInput = _beforeGtc.call(input);
            }
            else
            {
                // SAR/other path: use head input directly
                gtcInput = PrepareGtcInput(input, _gtcInChannels);
            }

            var gtcOut = _gtcHead.Forward(gtcInput, targets);
            var gtcKey = _gtcType == "sar" ? "sar" : "gtc";
            result[gtcKey] = gtcOut["predict"];
        }

        result["predict"] = result["ctc"];
        return result;
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
            "sar" or "sarhead" => new SARHead(inChannels, outChannels, hiddenSize, maxLen),
            "attn" or "attention" or "attentionhead" => new AttentionHead(inChannels, outChannels, hiddenSize, maxLen),
            _ => new AttentionHead(inChannels, outChannels, hiddenSize, maxLen)
        };
    }
}

/// <summary>
/// FCTranspose: Transpose + Linear (no bias).
/// Python: rec_multi_head.py FCTranspose
/// Input: [B, C, S] -> transpose -> [B, S, C] -> Linear -> [B, S, outChannels]
/// </summary>
internal sealed class FCTranspose : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _linear;

    public FCTranspose(int inChannels, int outChannels) : base(nameof(FCTranspose))
    {
        // Python: nn.Linear(in_channels, out_channels, bias_attr=False)
        _linear = Linear(inChannels, outChannels, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Input from Flatten(2): [B, C, S] -> transpose -> [B, S, C] -> Linear -> [B, S, out]
        return _linear.call(input.transpose(1, 2));
    }
}

/// <summary>
/// Flatten from dim=2: [B, C, H, W] -> [B, C, H*W]
/// Python: nn.Flatten(2)
/// </summary>
internal sealed class Flatten2D : Module<Tensor, Tensor>
{
    public Flatten2D() : base(nameof(Flatten2D))
    {
    }

    public override Tensor forward(Tensor input)
    {
        return input.flatten(2);
    }
}

/// <summary>
/// Learnable positional embedding added to input.
/// Python: AddPos in rec_multi_head.py
/// </summary>
internal sealed class AddPos : Module<Tensor, Tensor>
{
    private readonly Tensor _decPosEmbed;

    public AddPos(int dim, int w) : base(nameof(AddPos))
    {
        // Python: shape=[1, w, dim], initialized with trunc_normal_
        _decPosEmbed = Parameter(torch.randn(1, w, dim) * 0.02f);
        register_parameter("dec_pos_embed", (TorchSharp.Modules.Parameter)_decPosEmbed);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var seqLen = (int)input.shape[1];
        return input + _decPosEmbed[.., ..seqLen, ..];
    }
}

