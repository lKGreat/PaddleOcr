namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// Multi-label 编码器：组合 CTC + SAR/NRTR 双编码。
/// PP-OCRv4 使用此编码器，MultiHead 同时输出 CTC 和 Attention 预测。
/// 参考: ppocr/data/imaug/label_ops.py - MultiLabelEncode
/// </summary>
public sealed class MultiLabelEncode : IRecLabelEncoder
{
    private readonly CTCLabelEncode _ctcEncode;
    private readonly IRecLabelEncoder _gtcEncode;
    private readonly string _gtcKey;

    /// <param name="maxTextLength">最大文本长度</param>
    /// <param name="characterDictPath">字典文件路径</param>
    /// <param name="useSpaceChar">是否使用空格</param>
    /// <param name="gtcEncodeType">GTC 编码类型：null/空 = SAR, "NRTRLabelEncode" = NRTR</param>
    public MultiLabelEncode(int maxTextLength, string? characterDictPath = null, bool useSpaceChar = false, string? gtcEncodeType = null)
    {
        _ctcEncode = new CTCLabelEncode(maxTextLength, characterDictPath, useSpaceChar);

        if (string.IsNullOrEmpty(gtcEncodeType) || gtcEncodeType == "SARLabelEncode")
        {
            _gtcEncode = new SARLabelEncode(maxTextLength, characterDictPath, useSpaceChar);
            _gtcKey = "label_sar";
        }
        else if (gtcEncodeType == "NRTRLabelEncode")
        {
            _gtcEncode = new NRTRLabelEncode(maxTextLength, characterDictPath, useSpaceChar);
            _gtcKey = "label_gtc";
        }
        else
        {
            // 其他类型按名称尝试，默认用 SAR
            _gtcEncode = new SARLabelEncode(maxTextLength, characterDictPath, useSpaceChar);
            _gtcKey = "label_sar";
        }
    }

    public int NumClasses => _ctcEncode.NumClasses;
    public IReadOnlyList<string> Characters => _ctcEncode.Characters;

    /// <summary>GTC 编码器的类别数。</summary>
    public int GtcNumClasses => _gtcEncode.NumClasses;

    /// <summary>GTC 标签的 key 名称（"label_sar" 或 "label_gtc"）。</summary>
    public string GtcKey => _gtcKey;

    public RecLabelEncodeResult? Encode(string text)
    {
        return _ctcEncode.Encode(text);
    }

    /// <summary>
    /// 同时编码 CTC 和 GTC 标签。
    /// </summary>
    public MultiLabelEncodeResult? EncodeBoth(string text)
    {
        var ctcResult = _ctcEncode.Encode(text);
        var gtcResult = _gtcEncode.Encode(text);
        if (ctcResult is null || gtcResult is null)
        {
            return null;
        }

        return new MultiLabelEncodeResult(ctcResult.Label, gtcResult.Label, ctcResult.Length, _gtcKey);
    }
}
