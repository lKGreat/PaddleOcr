using PaddleOcr.Core.Errors;
using System.Collections;
using System.Text.RegularExpressions;

namespace PaddleOcr.Config;

public static class ConfigMerger
{
    private static readonly Regex PathTokenRegex = new(
        @"^(?<key>[^\[\]]+)(\[(?<index>\d+)\])?$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    public static void MergeInPlace(IDictionary<string, object?> config, IReadOnlyDictionary<string, object?> overrides)
    {
        foreach (var pair in overrides)
        {
            ApplyPath(config, pair.Key, pair.Value);
        }
    }

    private static void ApplyPath(IDictionary<string, object?> root, string path, object? value)
    {
        var rawParts = path.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (rawParts.Length == 0)
        {
            throw new PocrException($"Invalid override path: {path}");
        }

        var parts = rawParts.Select(ParseToken).ToArray();
        IDictionary<string, object?> current = root;
        for (var i = 0; i < parts.Length - 1; i++)
        {
            var token = parts[i];
            if (token.Index is null)
            {
                if (!current.TryGetValue(token.Key, out var next) || next is not IDictionary<string, object?> dict)
                {
                    dict = new Dictionary<string, object?>(StringComparer.Ordinal);
                    current[token.Key] = dict;
                }

                current = dict;
                continue;
            }

            if (!current.TryGetValue(token.Key, out var listObj) || listObj is not IList list)
            {
                list = new List<object?>();
                current[token.Key] = list;
            }

            var idx = token.Index.Value;
            EnsureListSize(list, idx + 1);
            if (list[idx] is not IDictionary<string, object?> childDict)
            {
                childDict = new Dictionary<string, object?>(StringComparer.Ordinal);
                list[idx] = childDict;
            }

            current = childDict;
        }

        var last = parts[^1];
        if (last.Index is null)
        {
            current[last.Key] = value;
            return;
        }

        if (!current.TryGetValue(last.Key, out var lastListObj) || lastListObj is not IList lastList)
        {
            lastList = new List<object?>();
            current[last.Key] = lastList;
        }

        var lastIdx = last.Index.Value;
        EnsureListSize(lastList, lastIdx + 1);
        lastList[lastIdx] = value;
    }

    private static PathToken ParseToken(string raw)
    {
        var m = PathTokenRegex.Match(raw);
        if (!m.Success)
        {
            throw new PocrException($"Invalid override path segment: {raw}");
        }

        var key = m.Groups["key"].Value;
        if (string.IsNullOrWhiteSpace(key))
        {
            throw new PocrException($"Invalid override path segment: {raw}");
        }

        if (m.Groups["index"].Success &&
            int.TryParse(m.Groups["index"].Value, out var index) &&
            index >= 0)
        {
            return new PathToken(key, index);
        }

        return new PathToken(key, null);
    }

    private static void EnsureListSize(IList list, int size)
    {
        while (list.Count < size)
        {
            list.Add(null);
        }
    }

    private readonly record struct PathToken(string Key, int? Index);
}
