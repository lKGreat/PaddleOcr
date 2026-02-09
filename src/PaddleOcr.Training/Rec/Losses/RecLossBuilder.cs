using System.Globalization;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// Build rec losses from YAML-like config dictionaries.
/// </summary>
public static class RecLossBuilder
{
    public static IRecLoss Build(string name, Dictionary<string, object>? config = null)
    {
        config ??= new Dictionary<string, object>();
        return name.ToLowerInvariant() switch
        {
            "ctc" or "ctcloss" => new CTCLoss(
                blank: GetInt(config, "blank", 0),
                reduction: GetBool(config, "reduction", true)),
            "attn" or "attention" or "attentionloss" => new AttentionLoss(
                ignoreIndex: GetInt(config, "ignore_index", 0)),
            "sar" or "sarloss" => new SARLoss(
                ignoreIndex: GetInt(config, "ignore_index", 0)),
            "nrtr" or "nrtrloss" => new NRTRLoss(
                labelSmoothing: GetFloat(config, "label_smoothing", 0.0f),
                paddingIdx: GetInt(config, "padding_idx", 0)),
            "srn" or "srnloss" => new SRNLoss(
                weight: GetFloat(config, "weight", 1.0f)),
            "multi" or "multiloss" => BuildMultiLoss(config),
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
            "ce" or "celoss" => new CELoss(
                labelSmoothing: GetFloat(config, "label_smoothing", 0.0f),
                ignoreIndex: GetInt(config, "ignore_index", 0)),
            _ => new CTCLoss()
        };
    }

    private static IRecLoss BuildMultiLoss(Dictionary<string, object> config)
    {
        var ctcWeight = GetFloatAny(config, ["weight_1", "ctc_weight"], 1.0f);
        var gtcWeight = GetFloatAny(config, ["weight_2", "gtc_weight", "attn_weight"], 1.0f);
        var ignoreIndex = GetInt(config, "ignore_index", 0);

        IRecLoss ctcLoss = new CTCLoss();
        IRecLoss? gtcLoss = null;

        if (TryGetList(config, "loss_config_list", out var lossConfigList))
        {
            foreach (var item in lossConfigList)
            {
                if (item is not Dictionary<string, object?> node || node.Count == 0)
                {
                    continue;
                }

                var kv = node.First();
                var subConfig = ToDictionary(kv.Value);
                var built = Build(kv.Key, subConfig);
                if (IsCtcLoss(kv.Key))
                {
                    ctcLoss = built;
                }
                else
                {
                    gtcLoss = built;
                }
            }
        }

        if (gtcLoss is null && gtcWeight > 0f)
        {
            gtcLoss = new AttentionLoss(ignoreIndex);
        }

        return new MultiLoss(
            ctcLoss: ctcLoss,
            gtcLoss: gtcLoss,
            ctcWeight: ctcWeight,
            gtcWeight: gtcWeight,
            ctcLabelKey: "label_ctc",
            gtcLabelKey: "label_gtc");
    }

    private static bool IsCtcLoss(string lossName)
    {
        return lossName.Contains("ctc", StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryGetList(Dictionary<string, object> config, string key, out IList<object?> list)
    {
        list = [];
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return false;
        }

        if (raw is IList<object?> objectList)
        {
            list = objectList;
            return true;
        }

        return false;
    }

    private static Dictionary<string, object> ToDictionary(object? raw)
    {
        if (raw is null)
        {
            return new Dictionary<string, object>();
        }

        if (raw is Dictionary<string, object> dict)
        {
            return dict;
        }

        if (raw is Dictionary<string, object?> nullableDict)
        {
            return nullableDict.ToDictionary(kv => kv.Key, kv => kv.Value ?? (object)string.Empty);
        }

        if (raw is IReadOnlyDictionary<string, object?> roDict)
        {
            return roDict.ToDictionary(kv => kv.Key, kv => kv.Value ?? (object)string.Empty);
        }

        return new Dictionary<string, object>();
    }

    private static int GetInt(Dictionary<string, object> config, string key, int defaultValue)
    {
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return defaultValue;
        }

        return raw switch
        {
            int i => i,
            long l => (int)l,
            float f => (int)f,
            double d => (int)d,
            decimal m => (int)m,
            _ => int.TryParse(raw.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) ? parsed : defaultValue
        };
    }

    private static float GetFloat(Dictionary<string, object> config, string key, float defaultValue)
    {
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return defaultValue;
        }

        return raw switch
        {
            float f => f,
            double d => (float)d,
            decimal m => (float)m,
            int i => i,
            long l => l,
            _ => float.TryParse(raw.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : defaultValue
        };
    }

    private static float GetFloatAny(Dictionary<string, object> config, IReadOnlyList<string> keys, float defaultValue)
    {
        foreach (var key in keys)
        {
            if (!config.ContainsKey(key))
            {
                continue;
            }

            return GetFloat(config, key, defaultValue);
        }

        return defaultValue;
    }

    private static bool GetBool(Dictionary<string, object> config, string key, bool defaultValue)
    {
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return defaultValue;
        }

        return raw switch
        {
            bool b => b,
            _ => bool.TryParse(raw.ToString(), out var parsed) ? parsed : defaultValue
        };
    }
}
