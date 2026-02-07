namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// Attention 标签编码器。
/// 在字典前添加 "sos"，末尾添加 "eos"。
/// 输出: [sos_idx(0)] + [char_ids...] + [eos_idx(N-1)] + [0 padding]
/// 参考: ppocr/data/imaug/label_ops.py - AttnLabelEncode
/// </summary>
public sealed class AttnLabelEncode : BaseRecLabelEncoder
{
    public AttnLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // sos 在 index 0, eos 在最后
        return ["sos", .. dictCharacter, "eos"];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null)
        {
            return null;
        }

        // AttnLabelEncode: text 长度必须 < maxTextLen（因为要加 sos + eos + padding）
        if (encoded.Count >= MaxTextLen)
        {
            return null;
        }

        var length = encoded.Count;
        var eosIdx = NumClasses - 1; // eos 是最后一个字符
        var label = new long[MaxTextLen];

        // [sos(0)] + [char_ids...] + [eos] + [0 padding]
        label[0] = 0; // sos
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }

        label[length + 1] = eosIdx;
        // 剩余位置保持 0

        return new RecLabelEncodeResult(label, length);
    }
}
