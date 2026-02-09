namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// SPIN 标签编码器。
/// 特殊 token: sos=0, eos=1，字符从 index 2 开始。
/// 输出: [sos(0)] + [char_ids...] + [eos(1)] + [0 padding]
/// 参考: ppocr/data/imaug/label_ops.py - SPINLabelEncode
/// </summary>
public sealed class SPINLabelEncode : BaseRecLabelEncoder
{
    public const int SosIdx = 0;
    public const int EosIdx = 1;

    public SPINLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        return ["sos", "eos", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen - 1) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];
        label[0] = SosIdx;
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }
        label[length + 1] = EosIdx;

        return new RecLabelEncodeResult(label, length);
    }
}
