namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// CAN 标签编码器。
/// 输出: [char_ids...] + [EOS_idx]，无 padding。
/// 参考: ppocr/data/imaug/label_ops.py - CANLabelEncode
/// </summary>
public sealed class CANLabelEncode : BaseRecLabelEncoder
{
    public CANLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // CAN uses raw dict characters
        return dictCharacter;
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count > MaxTextLen) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];
        for (var i = 0; i < length; i++)
        {
            label[i] = encoded[i];
        }

        return new RecLabelEncodeResult(label, length);
    }
}
