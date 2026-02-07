using PaddleOcr.Core.Errors;
using YamlDotNet.Serialization;

namespace PaddleOcr.Config;

public static class OverrideParser
{
    private static readonly IDeserializer Deserializer = new DeserializerBuilder().Build();

    public static IReadOnlyDictionary<string, object?> Parse(IReadOnlyList<string> rawOverrides)
    {
        var result = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var raw in rawOverrides)
        {
            var idx = raw.IndexOf('=');
            if (idx <= 0 || idx == raw.Length - 1)
            {
                throw new PocrException($"Invalid override: {raw}. Expect key=value");
            }

            var key = raw[..idx].Trim();
            var valueText = raw[(idx + 1)..].Trim();
            var value = DeserializeScalarOrComplex(valueText);
            result[key] = value;
        }

        return result;
    }

    private static object? DeserializeScalarOrComplex(string text)
    {
        if (text.Length >= 2 &&
            ((text.StartsWith('\'') && text.EndsWith('\'')) ||
             (text.StartsWith('"') && text.EndsWith('"'))))
        {
            return text[1..^1];
        }

        if (string.Equals(text, "null", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (bool.TryParse(text, out var b))
        {
            return b;
        }

        if (int.TryParse(text, out var i))
        {
            return i;
        }

        if (double.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d))
        {
            return d;
        }

        if (text.StartsWith('[') || text.StartsWith('{'))
        {
            try
            {
                return Deserializer.Deserialize<object>(text);
            }
            catch
            {
                return text;
            }
        }

        try
        {
            return Deserializer.Deserialize<object>(text);
        }
        catch
        {
            return text;
        }
    }
}
