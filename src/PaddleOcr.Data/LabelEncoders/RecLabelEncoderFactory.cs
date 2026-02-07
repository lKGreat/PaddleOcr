namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// 标签编码器工厂：根据算法/头名称创建对应编码器。
/// </summary>
public static class RecLabelEncoderFactory
{
    /// <summary>
    /// 根据头名称创建对应的标签编码器。
    /// </summary>
    public static IRecLabelEncoder Create(string headName, int maxTextLength, string? dictPath, bool useSpaceChar, string? gtcEncodeType = null)
    {
        return headName.ToLowerInvariant() switch
        {
            "ctc" or "ctchead" => new CTCLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "attn" or "attentionhead" or "attnhead" => new AttnLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "sar" or "sarhead" => new SARLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "nrtr" or "nrtrhead" => new NRTRLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "srn" or "srnhead" => new SRNLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "multi" or "multihead" => new MultiLabelEncode(maxTextLength, dictPath, useSpaceChar, gtcEncodeType),
            "satrn" or "satrnhead" => new SARLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "spin" or "spinattentionhead" => new AttnLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "robustscanner" or "robustscannerhead" => new SARLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "abinet" or "abinethead" => new NRTRLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "vl" or "vlhead" or "visionlan" => new CTCLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "parseq" or "parseqhead" => new NRTRLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "cppd" or "cppdhead" => new NRTRLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "can" or "canhead" => new CTCLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "pren" or "prenhead" => new AttnLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "latexocr" or "latexocrhead" => new NRTRLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "unimernet" or "unimernethead" => new NRTRLabelEncode(maxTextLength, dictPath, useSpaceChar),
            "rfl" or "rflhead" => new AttnLabelEncode(maxTextLength, dictPath, useSpaceChar),
            _ => new CTCLabelEncode(maxTextLength, dictPath, useSpaceChar)
        };
    }
}
