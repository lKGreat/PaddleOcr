using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// SRNHead：PVAM + GSRM + VSFD。
/// 用于 SRN 算法。
/// </summary>
public sealed class SRNHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly PVAM _pvam;
    private readonly GSRM _gsrm;
    private readonly VSFD _vsfd;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public SRNHead(int inChannels, int outChannels, int hiddenSize = 256, int maxLen = 25) : base(nameof(SRNHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;
        _pvam = new PVAM(inChannels, hiddenSize, maxLen);
        _gsrm = new GSRM(hiddenSize, outChannels, maxLen);
        _vsfd = new VSFD(hiddenSize, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var pvamOut = _pvam.call(input); // [B, maxLen, hiddenSize]
        var gsrmOut = _gsrm.call(pvamOut); // [B, maxLen, outChannels]
        _vsfd.SetSemanticFeat(gsrmOut);
        var vsfdOut = _vsfd.call(pvamOut); // [B, maxLen, outChannels]
        return vsfdOut;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}

/// <summary>
/// PVAM：Positional Visual Attention Module。
/// </summary>
internal sealed class PVAM : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _queryProj;
    private readonly Module<Tensor, Tensor> _keyProj;
    private readonly Module<Tensor, Tensor> _valueProj;
    private readonly Module<Tensor, Tensor> _posEmbed;
    private readonly int _maxLen;

    public PVAM(int inChannels, int hiddenSize, int maxLen) : base(nameof(PVAM))
    {
        _maxLen = maxLen;
        _queryProj = Linear(hiddenSize, hiddenSize);
        _keyProj = Linear(inChannels, hiddenSize);
        _valueProj = Linear(inChannels, hiddenSize);
        _posEmbed = Embedding(maxLen, hiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var b = input.shape[0];
        var w = input.shape[1];
        var device = input.device;

        // 创建位置查询
        var posIds = arange(_maxLen, ScalarType.Int64, device: device).unsqueeze(0).expand(b, -1);
        var posQueries = _posEmbed.call(posIds); // [B, maxLen, hiddenSize]
        var queries = _queryProj.call(posQueries);

        var keys = _keyProj.call(input); // [B, W, hiddenSize]
        var values = _valueProj.call(input);

        // Attention
        var scale = Math.Sqrt(queries.shape[2]);
        using var scores = torch.bmm(queries, keys.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var output = torch.bmm(attn, values); // [B, maxLen, hiddenSize]

        return output;
    }
}

/// <summary>
/// GSRM：Global Semantic Reasoning Module。
/// </summary>
internal sealed class GSRM : Module<Tensor, Tensor>
{
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;
    private readonly Module<Tensor, Tensor> _outputProj;

    public GSRM(int hiddenSize, int outChannels, int maxLen) : base(nameof(GSRM))
    {
        _layers = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        for (var i = 0; i < 2; i++)
        {
            _layers.Add(new GSRMLayer(hiddenSize));
        }

        _outputProj = Linear(hiddenSize, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, maxLen, hiddenSize]
        var x = input;
        foreach (var layer in _layers)
        {
            x = layer.call(x);
        }

        return _outputProj.call(x); // [B, maxLen, outChannels]
    }
}

/// <summary>
/// GSRMLayer：GSRM 层。
/// </summary>
internal sealed class GSRMLayer : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _selfAttn;
    private readonly Module<Tensor, Tensor> _ffn;

    public GSRMLayer(int hiddenSize) : base(nameof(GSRMLayer))
    {
        _selfAttn = Sequential(
            Linear(hiddenSize, hiddenSize),
            ReLU(),
            Linear(hiddenSize, hiddenSize)
        );
        _ffn = Sequential(
            Linear(hiddenSize, hiddenSize * 4),
            ReLU(),
            Linear(hiddenSize * 4, hiddenSize)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var residual = input;
        var x = _selfAttn.call(input);
        x = x + residual;
        using var residual2 = x;
        x = _ffn.call(x);
        return x + residual2;
    }
}

/// <summary>
/// VSFD：Visual-Semantic Fusion Decoder。
/// </summary>
internal sealed class VSFD : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _fusion;
    private Tensor? _semanticFeat;

    public VSFD(int hiddenSize, int outChannels) : base(nameof(VSFD))
    {
        _fusion = Sequential(
            Linear(hiddenSize + outChannels, hiddenSize),
            ReLU(),
            Linear(hiddenSize, outChannels)
        );
        RegisterComponents();
    }

    public void SetSemanticFeat(Tensor semanticFeat)
    {
        _semanticFeat = semanticFeat;
    }

    public override Tensor forward(Tensor visualFeat)
    {
        // visualFeat: [B, maxLen, hiddenSize]
        // semanticFeat: [B, maxLen, outChannels]
        if (_semanticFeat is null)
        {
            throw new InvalidOperationException("Semantic features must be set before forward");
        }

        var fused = cat(new[] { visualFeat, _semanticFeat }, dim: -1);
        return _fusion.call(fused);
    }
}
