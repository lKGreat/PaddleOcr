namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// ABINet 标签编码器。
/// 特殊 token: &lt;/s&gt;(EOS) 在 index 0，字符从 index 1 开始。
/// 输出: [char_ids...] + [EOS(0)] + [ignore_index padding]
/// 参考: ppocr/data/imaug/label_ops.py - ABINetLabelEncode
/// </summary>
public sealed class ABINetLabelEncode : BaseRecLabelEncoder
{
    public const int EosIdx = 0;
    private readonly int _ignoreIndex;

    public ABINetLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
        _ignoreIndex = NumClasses;
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        return ["</s>", .. dictCharacter];
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
        for (var i = length + 1; i < MaxTextLen; i++)
        {
            label[i] = _ignoreIndex;
        }

        return new RecLabelEncodeResult(label, length);
    }
}
