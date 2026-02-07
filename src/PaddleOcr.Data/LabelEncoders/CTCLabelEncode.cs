namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// CTC 标签编码器。
/// 在字典前添加 "blank" token (index=0)，字符从 index=1 开始。
/// 输出: [char_ids...] + [0 padding]
/// 参考: ppocr/data/imaug/label_ops.py - CTCLabelEncode
/// </summary>
public sealed class CTCLabelEncode : BaseRecLabelEncoder
{
    public CTCLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // blank token 在 index 0
        return ["blank", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null)
        {
            return null;
        }

        var length = encoded.Count;
        var label = new long[MaxTextLen];
        for (var i = 0; i < Math.Min(length, MaxTextLen); i++)
        {
            label[i] = encoded[i];
        }
        // 剩余位置保持 0（blank）

        return new RecLabelEncodeResult(label, length);
    }
}
