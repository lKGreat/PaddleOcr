namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// ViTSTR 标签编码器。
/// 特殊 token: &lt;s&gt;=0, &lt;/s&gt;=1，字符从 index 2 开始。
/// 输出: [ignore_index] + [char_ids...] + [&lt;/s&gt;(1)] + [ignore_index padding]
/// 参考: ppocr/data/imaug/label_ops.py - ViTSTRLabelEncode
/// </summary>
public sealed class ViTSTRLabelEncode : BaseRecLabelEncoder
{
    public const int BosIdx = 0;
    public const int EosIdx = 1;
    private readonly int _ignoreIndex;

    public ViTSTRLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
        _ignoreIndex = NumClasses;
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        return ["<s>", "</s>", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen - 1) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];

        // [ignore_index] + [char_ids...] + [</s>(1)] + [ignore_index padding]
        label[0] = _ignoreIndex;
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }
        label[length + 1] = EosIdx;
        for (var i = length + 2; i < MaxTextLen; i++)
        {
            label[i] = _ignoreIndex;
        }

        return new RecLabelEncodeResult(label, length);
    }
}
