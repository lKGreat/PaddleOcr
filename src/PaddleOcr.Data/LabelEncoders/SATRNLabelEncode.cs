namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// SATRN 标签编码器。
/// 特殊 token: 字符后添加 [&lt;UKN&gt;, &lt;BOS/EOS&gt;, &lt;PAD&gt;]。
/// 输出: [start_idx] + [char_ids...] + [end_idx] + [pad_idx padding]
/// 参考: ppocr/data/imaug/label_ops.py - SATRNLabelEncode
/// </summary>
public sealed class SATRNLabelEncode : BaseRecLabelEncoder
{
    public int UnkIdx { get; private set; }
    public int BosEosIdx { get; private set; }
    public int PadIdx { get; private set; }

    public SATRNLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        var result = new List<string>(dictCharacter) { "<UKN>", "<BOS/EOS>", "<PAD>" };
        UnkIdx = result.Count - 3;
        BosEosIdx = result.Count - 2;
        PadIdx = result.Count - 1;
        return result;
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen - 1) return null;

        var length = encoded.Count;
        var label = new long[MaxTextLen];
        label[0] = BosEosIdx; // start
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }
        label[length + 1] = BosEosIdx; // end
        for (var i = length + 2; i < MaxTextLen; i++)
        {
            label[i] = PadIdx;
        }

        return new RecLabelEncodeResult(label, length);
    }
}
