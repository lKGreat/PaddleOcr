using PaddleOcr.Core.Errors;

namespace PaddleOcr.Config;

public static class ConfigMerger
{
    public static void MergeInPlace(IDictionary<string, object?> config, IReadOnlyDictionary<string, object?> overrides)
    {
        foreach (var pair in overrides)
        {
            ApplyPath(config, pair.Key, pair.Value);
        }
    }

    private static void ApplyPath(IDictionary<string, object?> root, string path, object? value)
    {
        var parts = path.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
        {
            throw new PocrException($"Invalid override path: {path}");
        }

        IDictionary<string, object?> current = root;
        for (var i = 0; i < parts.Length - 1; i++)
        {
            var key = parts[i];
            if (!current.TryGetValue(key, out var next) || next is not IDictionary<string, object?> dict)
            {
                dict = new Dictionary<string, object?>(StringComparer.Ordinal);
                current[key] = dict;
            }

            current = dict;
        }

        current[parts[^1]] = value;
    }
}

