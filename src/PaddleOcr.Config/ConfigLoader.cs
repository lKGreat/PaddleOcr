using PaddleOcr.Core.Errors;
using YamlDotNet.Serialization;

namespace PaddleOcr.Config;

public sealed class ConfigLoader
{
    private readonly IDeserializer _deserializer;

    public ConfigLoader()
    {
        _deserializer = new DeserializerBuilder().Build();
    }

    public IReadOnlyDictionary<string, object?> Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new PocrException($"Config file not found: {path}");
        }

        var yaml = File.ReadAllText(path);
        var raw = _deserializer.Deserialize<object>(yaml);
        var normalized = Normalize(raw) as Dictionary<string, object?>;
        return normalized ?? new Dictionary<string, object?>(StringComparer.Ordinal);
    }

    private static object? Normalize(object? node)
    {
        if (node is null)
        {
            return null;
        }

        if (node is Dictionary<object, object> objDict)
        {
            var result = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var pair in objDict)
            {
                result[pair.Key.ToString() ?? string.Empty] = Normalize(pair.Value);
            }

            return result;
        }

        if (node is Dictionary<string, object> strDict)
        {
            var result = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var pair in strDict)
            {
                result[pair.Key] = Normalize(pair.Value);
            }

            return result;
        }

        if (node is IList<object> list)
        {
            return list.Select(Normalize).ToList();
        }

        return node;
    }
}

