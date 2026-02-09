namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// VisionLAN 标签编码器。
/// 使用 CTC 风格编码但额外生成遮挡标签 (label_res, label_sub, label_id)。
/// 字符 index 从 1 开始（+1 offset），0 为 padding。
/// 参考: ppocr/data/imaug/label_ops.py - VLLabelEncode
/// </summary>
public sealed class VLLabelEncode : BaseRecLabelEncoder
{
    public VLLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // VL 不添加特殊 token，但字符 index 从 1 开始
        return dictCharacter;
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count > MaxTextLen) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];
        var labelRes = new long[MaxTextLen]; // Full label
        var labelSub = new long[MaxTextLen]; // Sub label (occluded chars masked)
        var labelId = new long[MaxTextLen]; // Occlusion position indices

        for (var i = 0; i < length; i++)
        {
            label[i] = encoded[i] + 1; // +1 offset, 0 reserved for padding
            labelRes[i] = encoded[i] + 1;
            labelSub[i] = encoded[i] + 1;
        }
        // labelId stays all zeros (no occlusion by default; training data pipeline fills this)

        return new VLLabelEncodeResult(label, length, labelRes, labelSub, labelId);
    }
}

/// <summary>
/// VisionLAN 编码结果：包含遮挡标签。
/// </summary>
public sealed record VLLabelEncodeResult(
    long[] Label, int Length,
    long[] LabelRes, long[] LabelSub, long[] LabelId)
    : RecLabelEncodeResult(Label, Length);
