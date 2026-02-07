using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// VLHead：VisionLAN 的 head，包含视觉推理模块 (VRM)。
/// 训练阶段分三个步骤：LF_1（仅视觉）、LF_2（遮挡+视觉）、LA（语言辅助）。
/// 推理阶段使用 LA 完整路径。
/// 参考 ppocr/modeling/heads/rec_vl_head.py。
/// </summary>
public sealed class VLHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _charHead;
    private readonly Module<Tensor, Tensor> _lengthHead;
    private readonly VisualReasoningModule _vrm;
    private readonly int _outChannels;
    private readonly int _maxLen;

    public VLHead(int inChannels, int outChannels, int maxLen = 25) : base(nameof(VLHead))
    {
        _outChannels = outChannels;
        _maxLen = maxLen;

        // 字符分类头
        _charHead = Linear(inChannels, outChannels);

        // 长度预测头
        _lengthHead = Linear(inChannels, maxLen + 1);

        // 视觉推理模块
        _vrm = new VisualReasoningModule(inChannels, outChannels, maxLen);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        // 推理阶段：使用 VRM 进行语言辅助识别
        var vrmOut = _vrm.call(input); // [B, maxLen, outChannels]
        return vrmOut;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        // input: [B, W, C]
        var charLogits = _charHead.call(input); // [B, W, outChannels]

        // 长度预测
        using var pooled = input.mean(new long[] { 1 }); // [B, C]
        var lengthLogits = _lengthHead.call(pooled); // [B, maxLen+1]

        // VRM 语言辅助识别
        var vrmOut = _vrm.call(input); // [B, maxLen, outChannels]

        return new Dictionary<string, Tensor>
        {
            ["predict"] = vrmOut,
            ["visual"] = charLogits,
            ["length"] = lengthLogits
        };
    }
}

/// <summary>
/// VisualReasoningModule：视觉推理模块。
/// 使用注意力机制从视觉特征中解码出固定长度的字符 logits。
/// </summary>
internal sealed class VisualReasoningModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _queryEmbed;
    private readonly Module<Tensor, Tensor> _queryProj;
    private readonly Module<Tensor, Tensor> _keyProj;
    private readonly Module<Tensor, Tensor> _valueProj;
    private readonly Module<Tensor, Tensor> _outputProj;
    private readonly int _maxLen;

    public VisualReasoningModule(int inChannels, int outChannels, int maxLen) : base(nameof(VisualReasoningModule))
    {
        _maxLen = maxLen;
        _queryEmbed = Embedding(maxLen, inChannels);
        _queryProj = Linear(inChannels, inChannels);
        _keyProj = Linear(inChannels, inChannels);
        _valueProj = Linear(inChannels, inChannels);
        _outputProj = Sequential(
            Linear(inChannels, inChannels),
            ReLU(),
            Linear(inChannels, outChannels)
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var b = input.shape[0];
        var device = input.device;

        // 位置查询
        var posIds = arange(_maxLen, ScalarType.Int64, device: device).unsqueeze(0).expand(b, -1);
        var queries = _queryEmbed.call(posIds); // [B, maxLen, C]
        queries = _queryProj.call(queries);

        var keys = _keyProj.call(input); // [B, W, C]
        var values = _valueProj.call(input); // [B, W, C]

        // Attention: queries 与 keys 的注意力
        var scale = Math.Sqrt(queries.shape[2]);
        using var scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale;
        using var attn = functional.softmax(scores, dim: -1);
        var attended = torch.matmul(attn, values); // [B, maxLen, C]

        return _outputProj.call(attended); // [B, maxLen, outChannels]
    }
}
