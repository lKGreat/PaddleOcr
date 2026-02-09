namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// PREN 标签编码器。
/// 特殊 token: PAD=0, EOS=1, UNK=2，字符从 index 3 开始。
/// 输出: [char_ids...] + [EOS(1)] + [PAD(0) padding]
/// 参考: ppocr/data/imaug/label_ops.py - PRENLabelEncode
/// </summary>
public sealed class PRENLabelEncode : BaseRecLabelEncoder
{
    public const int PadIdx = 0;
    public const int EosIdx = 1;
    public const int UnkIdx = 2;

    public PRENLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        return ["<PAD>", "<EOS>", "<UNK>", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];
        for (var i = 0; i < length; i++)
        {
            label[i] = encoded[i];
        }
        label[length] = EosIdx;
        // Remaining positions are 0 (PAD)

        return new RecLabelEncodeResult(label, length);
    }
}
