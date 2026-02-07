namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// NRTR 标签编码器。
/// 在字典前添加 [blank, &lt;unk&gt;, &lt;s&gt;, &lt;/s&gt;]，字符从 index=4 开始。
/// 输出: [2(&lt;s&gt;)] + [char_ids...] + [3(&lt;/s&gt;)] + [0 padding]
/// 参考: ppocr/data/imaug/label_ops.py - NRTRLabelEncode
/// </summary>
public sealed class NRTRLabelEncode : BaseRecLabelEncoder
{
    /// <summary>blank token 索引。</summary>
    public const int BlankIdx = 0;
    /// <summary>&lt;unk&gt; token 索引。</summary>
    public const int UnkIdx = 1;
    /// <summary>&lt;s&gt; (BOS) token 索引。</summary>
    public const int BosIdx = 2;
    /// <summary>&lt;/s&gt; (EOS) token 索引。</summary>
    public const int EosIdx = 3;

    public NRTRLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // [blank, <unk>, <s>, </s>] 在前面
        return ["blank", "<unk>", "<s>", "</s>", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null)
        {
            return null;
        }

        // NRTR: 文本长度必须 < maxTextLen - 1（因为要加 <s> + </s>）
        if (encoded.Count >= MaxTextLen - 1)
        {
            return null;
        }

        var length = encoded.Count;

        // [<s>(2)] + [char_ids...] + [</s>(3)] + [0 padding]
        var label = new long[MaxTextLen];
        label[0] = BosIdx; // <s>
        for (var i = 0; i < length; i++)
        {
            label[i + 1] = encoded[i];
        }

        label[length + 1] = EosIdx; // </s>
        // 剩余位置保持 0（blank）

        return new RecLabelEncodeResult(label, length);
    }
}
