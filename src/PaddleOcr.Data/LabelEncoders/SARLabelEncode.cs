namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// SAR 标签编码器。
/// 在字典末尾依次添加 &lt;UKN&gt;, &lt;BOS/EOS&gt;, &lt;PAD&gt;。
/// 输出: [BOS_idx] + [char_ids...] + [EOS_idx] + [PAD_idx padding]
/// 参考: ppocr/data/imaug/label_ops.py - SARLabelEncode
/// </summary>
public sealed class SARLabelEncode : BaseRecLabelEncoder
{
    private int _unknownIdx;
    private int _startIdx;
    private int _endIdx;
    private int _paddingIdx;

    public SARLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false)
        : base(maxTextLength, characterDictPath, useSpaceChar)
    {
    }

    /// <summary>PAD token 的索引。</summary>
    public int PaddingIdx => _paddingIdx;

    /// <summary>BOS/EOS token 的索引。</summary>
    public int StartIdx => _startIdx;

    protected override List<string> AddSpecialChar(List<string> dictCharacter)
    {
        // 按照官方顺序：字符 + <UKN> + <BOS/EOS> + <PAD>
        dictCharacter.Add("<UKN>");
        _unknownIdx = dictCharacter.Count - 1;
        dictCharacter.Add("<BOS/EOS>");
        _startIdx = dictCharacter.Count - 1;
        _endIdx = _startIdx;
        dictCharacter.Add("<PAD>");
        _paddingIdx = dictCharacter.Count - 1;
        return dictCharacter;
    }

    public override RecLabelEncodeResult? Encode(string text)
    {
        var encoded = EncodeText(text);
        if (encoded is null)
        {
            return null;
        }

        // SAR: 文本长度必须 < maxTextLen - 1（因为要加 BOS + EOS）
        if (encoded.Count >= MaxTextLen - 1)
        {
            return null;
        }

        var length = encoded.Count;

        // target = [BOS] + text + [EOS]
        var target = new List<int>(length + 2) { _startIdx };
        target.AddRange(encoded);
        target.Add(_endIdx);

        // 用 PAD 填充到 maxTextLen
        var label = new long[MaxTextLen];
        for (var i = 0; i < MaxTextLen; i++)
        {
            label[i] = i < target.Count ? target[i] : _paddingIdx;
        }

        return new RecLabelEncodeResult(label, length);
    }
}
