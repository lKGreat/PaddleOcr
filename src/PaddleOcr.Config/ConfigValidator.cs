namespace PaddleOcr.Config;

public static class ConfigValidator
{
    public static bool ValidateBasic(IReadOnlyDictionary<string, object?> cfg, out string message)
    {
        if (!cfg.ContainsKey("Global"))
        {
            message = "missing top-level section: Global";
            return false;
        }

        if (!cfg.ContainsKey("Architecture"))
        {
            message = "missing top-level section: Architecture";
            return false;
        }

        message = "ok";
        return true;
    }

    public static IReadOnlyList<string> Diff(IReadOnlyDictionary<string, object?> baseCfg, IReadOnlyDictionary<string, object?> targetCfg)
    {
        var left = Flatten(baseCfg);
        var right = Flatten(targetCfg);
        var keys = left.Keys.Concat(right.Keys).Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
        var diffs = new List<string>();
        foreach (var key in keys)
        {
            left.TryGetValue(key, out var lv);
            right.TryGetValue(key, out var rv);
            if (string.Equals(lv, rv, StringComparison.Ordinal))
            {
                continue;
            }

            diffs.Add($"{key}: '{lv ?? "<null>"}' -> '{rv ?? "<null>"}'");
        }

        return diffs;
    }

    private static Dictionary<string, string?> Flatten(IReadOnlyDictionary<string, object?> cfg)
    {
        var output = new Dictionary<string, string?>(StringComparer.Ordinal);
        Walk(cfg, string.Empty, output);
        return output;
    }

    private static void Walk(object? node, string path, IDictionary<string, string?> output)
    {
        if (node is IReadOnlyDictionary<string, object?> rd)
        {
            foreach (var (k, v) in rd)
            {
                Walk(v, Append(path, k), output);
            }
            return;
        }

        if (node is Dictionary<string, object?> d)
        {
            foreach (var (k, v) in d)
            {
                Walk(v, Append(path, k), output);
            }
            return;
        }

        if (node is List<object?> list)
        {
            for (var i = 0; i < list.Count; i++)
            {
                Walk(list[i], $"{path}[{i}]", output);
            }
            return;
        }

        output[path] = node?.ToString();
    }

    private static string Append(string path, string key)
    {
        return string.IsNullOrEmpty(path) ? key : $"{path}.{key}";
    }
}

