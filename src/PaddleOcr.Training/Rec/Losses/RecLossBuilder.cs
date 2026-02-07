using PaddleOcr.Training.Rec.Losses;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// RecLossBuilder：从配置构建损失函数。
/// </summary>
public static class RecLossBuilder
{
    /// <summary>
    /// 构建损失函数。
    /// </summary>
    public static IRecLoss Build(string name, Dictionary<string, object>? config = null)
    {
        config ??= new Dictionary<string, object>();
        return name.ToLowerInvariant() switch
        {
            "ctc" or "ctcloss" => new CTCLoss(
                blank: GetInt(config, "blank", 0),
                reduction: GetBool(config, "reduction", true)
            ),
            "attn" or "attention" or "attentionloss" => new AttentionLoss(
                ignoreIndex: GetInt(config, "ignore_index", 0)
            ),
            "sar" or "sarloss" => new SARLoss(
                ignoreIndex: GetInt(config, "ignore_index", 0)
            ),
            "nrtr" or "nrtrloss" => new NRTRLoss(
                labelSmoothing: GetFloat(config, "label_smoothing", 0.0f),
                paddingIdx: GetInt(config, "padding_idx", 0)
            ),
            "srn" or "srnloss" => new SRNLoss(
                weight: GetFloat(config, "weight", 1.0f)
            ),
            "multi" or "multiloss" => new MultiLoss(
                ctcWeight: GetFloat(config, "ctc_weight", 1.0f),
                attnWeight: GetFloat(config, "attn_weight", 1.0f),
                ignoreIndex: GetInt(config, "ignore_index", 0)
            ),
            "aster" or "asterloss" => new AsterLoss(),
            "pren" or "prenloss" => new PRENLoss(),
            "vl" or "vlloss" => new VLLoss(),
            "spin" or "spinattentionloss" => new SPINAttentionLoss(),
            "rfl" or "rflloss" => new RFLLoss(),
            "can" or "canloss" => new CANLoss(),
            "satrn" or "satrnloss" => new SATRNLoss(),
            "parseq" or "parseqloss" => new ParseQLoss(),
            "cppd" or "cppdloss" => new CPPDLoss(),
            "latexocr" or "latexocrloss" => new LaTeXOCRLoss(),
            "unimernet" or "unimernetloss" => new UniMERNetLoss(),
            "ppformulanet" or "ppformulanetloss" => new PPFormulaNetLoss(),
            "enhancedctc" or "enhancedctcloss" => new EnhancedCTCLoss(),
            _ => new CTCLoss()
        };
    }

    private static int GetInt(Dictionary<string, object> config, string key, int defaultValue)
    {
        return config.ContainsKey(key) && config[key] is int value ? value : defaultValue;
    }

    private static float GetFloat(Dictionary<string, object> config, string key, float defaultValue)
    {
        return config.ContainsKey(key) && config[key] is float value ? value : defaultValue;
    }

    private static bool GetBool(Dictionary<string, object> config, string key, bool defaultValue)
    {
        return config.ContainsKey(key) && config[key] is bool value ? value : defaultValue;
    }
}
