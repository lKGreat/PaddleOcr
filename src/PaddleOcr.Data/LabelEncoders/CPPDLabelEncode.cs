namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// CPPD 标签编码器。
/// 特殊 token: [&lt;/s&gt;] 在 index 0 (EOS)，字符从 index 1 开始。
/// 额外生成 label_node 和 label_order。
/// 参考: ppocr/data/imaug/label_ops.py - CPPDLabelEncode
/// </summary>
public sealed class CPPDLabelEncode : BaseRecLabelEncoder
{
    private readonly int _ignoreIndex;

    public CPPDLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
        _ignoreIndex = NumClasses;
    }

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // </s> (EOS) at index 0
        return ["</s>", .. dictCharacter];
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null) return null;
        if (encoded.Count >= MaxTextLen) return null;

        var length = encoded.Count;

        // Label: [char_ids...] + [eos(0)] + [ignore_index padding]
        var label = new long[MaxTextLen];
        for (var i = 0; i < length; i++)
        {
            label[i] = encoded[i];
        }
        label[length] = 0; // EOS
        for (var i = length + 1; i < MaxTextLen; i++)
        {
            label[i] = _ignoreIndex;
        }

        // label_node: one-hot positions for characters present
        var labelNode = new long[NumClasses];
        foreach (var idx in encoded)
        {
            if (idx >= 0 && idx < NumClasses)
            {
                labelNode[idx] = 1;
            }
        }
        labelNode[0] = 1; // EOS is always present

        // label_order: positional ordering of characters
        var labelOrder = new long[MaxTextLen];
        for (var i = 0; i < length; i++)
        {
            labelOrder[i] = encoded[i];
        }

        return new CPPDLabelEncodeResult(label, length, labelNode, labelOrder);
    }
}

/// <summary>
/// CPPD 编码结果：包含 label_node 和 label_order。
/// </summary>
public sealed record CPPDLabelEncodeResult(
    long[] Label, int Length,
    long[] LabelNode, long[] LabelOrder)
    : RecLabelEncodeResult(Label, Length);
