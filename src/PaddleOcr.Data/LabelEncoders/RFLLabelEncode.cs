namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// RFL 标签编码器。
/// 与 AttnLabelEncode 相同的特殊 token 但额外生成 cnt_label（字符计数标签）。
/// 参考: ppocr/data/imaug/label_ops.py - RFLLabelEncode
/// </summary>
public sealed class RFLLabelEncode : BaseRecLabelEncoder
{
    public RFLLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        return ["sos", .. dictCharacter, "eos"];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen) return null;

        var length = encoded.Count;
        var eosIdx = NumClasses - 1;
        var label = new long[MaxTextLen];
        label[0] = 0; // sos
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }
        label[length + 1] = eosIdx;

        return new RFLLabelEncodeResult(label, length, length);
    }
}

/// <summary>
/// RFL 编码结果：包含额外的 cnt_label（字符计数）。
/// </summary>
public sealed record RFLLabelEncodeResult(long[] Label, int Length, int CntLabel) : RecLabelEncodeResult(Label, Length);
