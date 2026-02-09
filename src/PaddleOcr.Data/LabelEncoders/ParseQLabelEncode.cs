namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// ParseQ 标签编码器。
/// 特殊 token: [E](EOS)=0, [B](BOS)=1, [P](PAD)=2，字符从 index 3 开始。
/// 输出: [BOS(1)] + [char_ids...] + [EOS(0)] + [PAD(2) padding]
/// 参考: ppocr/data/imaug/label_ops.py - ParseQLabelEncode
/// </summary>
public sealed class ParseQLabelEncode : BaseRecLabelEncoder
{
    public const int EosIdx = 0;
    public const int BosIdx = 1;
    public const int PadIdx = 2;

    public ParseQLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        return ["[E]", "[B]", "[P]", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen - 1) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];

        // [BOS] + chars + [EOS] + [PAD...]
        label[0] = BosIdx;
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }
        label[length + 1] = EosIdx;
        for (var i = length + 2; i < MaxTextLen; i++)
        {
            label[i] = PadIdx;
        }

        return new RecLabelEncodeResult(label, length);
    }
}
