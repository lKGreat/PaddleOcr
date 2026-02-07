namespace PaddleOcr.Models;

/// <summary>
/// PaddleOCR 支持的所有 rec 识别算法枚举。
/// </summary>
public enum RecAlgorithm
{
    CRNN,
    STARNet,
    RARE,
    SRN,
    NRTR,
    SAR,
    SVTR,
    SVTR_LCNet,
    SVTR_HGNet,
    ViTSTR,
    ABINet,
    SPIN,
    VisionLAN,
    RobustScanner,
    RFL,
    SATRN,
    ParseQ,
    CPPD,
    CPPDPadding,
    CAN,
    LaTeXOCR,
    UniMERNet,
    PPFormulaNet_S,
    PPFormulaNet_L,
    PPFormulaNet_Plus_S,
    PPFormulaNet_Plus_M,
    PPFormulaNet_Plus_L,
    PREN
}

/// <summary>
/// RecAlgorithm 相关的扩展与工具方法。
/// </summary>
public static class RecAlgorithmExtensions
{
    /// <summary>
    /// 从字符串解析 RecAlgorithm，支持大小写不敏感和常见别名。
    /// </summary>
    public static RecAlgorithm Parse(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return RecAlgorithm.SVTR_LCNet;
        }

        var normalized = name.Trim();

        // 处理常见别名
        if (normalized.Equals("SVTR_LCNet", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.SVTR_LCNet;
        }

        if (normalized.Equals("SVTR_HGNet", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.SVTR_HGNet;
        }

        if (normalized.Equals("CPPD_Padding", StringComparison.OrdinalIgnoreCase) ||
            normalized.Equals("CPPDPadding", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.CPPDPadding;
        }

        if (normalized.Equals("PP-FormulaNet-S", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.PPFormulaNet_S;
        }

        if (normalized.Equals("PP-FormulaNet-L", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.PPFormulaNet_L;
        }

        if (normalized.Equals("PP-FormulaNet_plus-S", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.PPFormulaNet_Plus_S;
        }

        if (normalized.Equals("PP-FormulaNet_plus-M", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.PPFormulaNet_Plus_M;
        }

        if (normalized.Equals("PP-FormulaNet_plus-L", StringComparison.OrdinalIgnoreCase))
        {
            return RecAlgorithm.PPFormulaNet_Plus_L;
        }

        if (Enum.TryParse<RecAlgorithm>(normalized, ignoreCase: true, out var result))
        {
            return result;
        }

        return RecAlgorithm.SVTR_LCNet;
    }

    /// <summary>
    /// 获取算法对应的默认后处理解码器名称。
    /// </summary>
    public static string GetDefaultPostprocessor(this RecAlgorithm algorithm)
    {
        return algorithm switch
        {
            RecAlgorithm.CRNN or RecAlgorithm.STARNet or RecAlgorithm.SVTR
                or RecAlgorithm.SVTR_LCNet or RecAlgorithm.SVTR_HGNet => "ctc",
            RecAlgorithm.RARE or RecAlgorithm.SPIN => "attn",
            RecAlgorithm.SRN => "srn",
            RecAlgorithm.NRTR => "nrtr",
            RecAlgorithm.SAR => "sar",
            RecAlgorithm.ViTSTR => "vitstr",
            RecAlgorithm.ABINet => "abinet",
            RecAlgorithm.CAN => "can",
            RecAlgorithm.LaTeXOCR => "latexocr",
            RecAlgorithm.ParseQ => "parseq",
            RecAlgorithm.CPPD or RecAlgorithm.CPPDPadding => "cppd",
            RecAlgorithm.PREN => "pren",
            RecAlgorithm.UniMERNet => "unimernet",
            RecAlgorithm.VisionLAN => "visionlan",
            RecAlgorithm.RFL => "rfl",
            RecAlgorithm.SATRN => "satrn",
            RecAlgorithm.RobustScanner => "sar",
            RecAlgorithm.PPFormulaNet_S or RecAlgorithm.PPFormulaNet_L
                or RecAlgorithm.PPFormulaNet_Plus_S or RecAlgorithm.PPFormulaNet_Plus_M
                or RecAlgorithm.PPFormulaNet_Plus_L => "latexocr",
            _ => "ctc"
        };
    }

    /// <summary>
    /// 获取算法对应的默认图像形状 (C, H, W)。
    /// </summary>
    public static (int C, int H, int W) GetDefaultImageShape(this RecAlgorithm algorithm)
    {
        return algorithm switch
        {
            RecAlgorithm.NRTR or RecAlgorithm.ViTSTR => (1, 32, 100),
            RecAlgorithm.SAR or RecAlgorithm.RobustScanner or RecAlgorithm.SATRN => (3, 48, 48),
            RecAlgorithm.SRN => (1, 64, 256),
            RecAlgorithm.CAN => (1, 256, 256),
            RecAlgorithm.LaTeXOCR or RecAlgorithm.PPFormulaNet_S or RecAlgorithm.PPFormulaNet_L
                or RecAlgorithm.PPFormulaNet_Plus_S or RecAlgorithm.PPFormulaNet_Plus_M
                or RecAlgorithm.PPFormulaNet_Plus_L => (3, 192, 672),
            RecAlgorithm.UniMERNet => (3, 192, 672),
            RecAlgorithm.ABINet => (3, 32, 128),
            RecAlgorithm.VisionLAN => (3, 64, 256),
            RecAlgorithm.SPIN => (3, 32, 100),
            RecAlgorithm.RFL => (1, 32, 100),
            RecAlgorithm.ParseQ => (3, 32, 128),
            RecAlgorithm.CPPD or RecAlgorithm.CPPDPadding => (3, 32, 128),
            RecAlgorithm.PREN => (3, 64, 256),
            _ => (3, 48, 320)
        };
    }
}
