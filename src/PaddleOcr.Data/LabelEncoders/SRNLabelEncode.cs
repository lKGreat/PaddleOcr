namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// SRN 标签编码器。
/// 在字典末尾添加 sos, eos。
/// 输出: [char_ids...] + [eos_idx padding]（用 eos 填充剩余位置）
/// 参考: ppocr/data/imaug/label_ops.py - SRNLabelEncode
/// </summary>
public sealed class SRNLabelEncode : BaseRecLabelEncoder
{
    public SRNLabelEncode(int maxTextLength = 25, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // sos, eos 在末尾
        dictCharacter.Add("sos");
        dictCharacter.Add("eos");
        return dictCharacter;
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null)
        {
            return null;
        }

        if (encoded.Count > MaxTextLen)
        {
            return null;
        }

        var length = encoded.Count;
        var eosIdx = NumClasses - 1; // eos 是最后一个字符

        // [char_ids...] + [eos_idx padding]
        var label = new long[MaxTextLen];
        for (var i = 0; i < MaxTextLen; i++)
        {
            label[i] = i < length ? encoded[i] : eosIdx;
        }

        return new RecLabelEncodeResult(label, length);
    }
}
