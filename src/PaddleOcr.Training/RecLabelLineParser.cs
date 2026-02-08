namespace PaddleOcr.Training;

public static class RecLabelLineParser
{
    public static bool TryParse(string line, out string imageRelPath, out string text)
    {
        return TryParse(line, delimiter: null, out imageRelPath, out text);
    }

    public static bool TryParse(string line, string? delimiter, out string imageRelPath, out string text)
    {
        imageRelPath = string.Empty;
        text = string.Empty;
        if (string.IsNullOrWhiteSpace(line))
        {
            return false;
        }

        if (TryParseByDelimiter(line, delimiter, out imageRelPath, out text))
        {
            return true;
        }

        if (TryParseByTab(line, out imageRelPath, out text))
        {
            return true;
        }

        return TryParseByWhitespace(line, out imageRelPath, out text);
    }

    private static bool TryParseByDelimiter(string line, string? delimiter, out string imageRelPath, out string text)
    {
        imageRelPath = string.Empty;
        text = string.Empty;
        if (string.IsNullOrWhiteSpace(delimiter))
        {
            return false;
        }

        var normalized = NormalizeDelimiter(delimiter);
        if (string.IsNullOrEmpty(normalized))
        {
            return false;
        }

        if (normalized.Any(char.IsWhiteSpace))
        {
            return TryParseByWhitespace(line, out imageRelPath, out text);
        }

        var idx = line.IndexOf(normalized, StringComparison.Ordinal);
        if (idx <= 0)
        {
            return false;
        }

        var left = line[..idx].Trim();
        var right = line[(idx + normalized.Length)..].Trim();
        if (left.Length == 0 || right.Length == 0)
        {
            return false;
        }

        imageRelPath = left;
        text = right;
        return true;
    }

    private static bool TryParseByTab(string line, out string imageRelPath, out string text)
    {
        imageRelPath = string.Empty;
        text = string.Empty;
        var tabIdx = line.IndexOf('\t');
        if (tabIdx < 0)
        {
            return false;
        }

        var left = line[..tabIdx].Trim();
        var right = line[(tabIdx + 1)..].Trim();
        if (left.Length == 0 || right.Length == 0)
        {
            return false;
        }

        imageRelPath = left;
        text = right;
        return true;
    }

    private static bool TryParseByWhitespace(string line, out string imageRelPath, out string text)
    {
        imageRelPath = string.Empty;
        text = string.Empty;
        var span = line.AsSpan();
        var sepIdx = -1;
        for (var i = 0; i < span.Length; i++)
        {
            if (char.IsWhiteSpace(span[i]))
            {
                sepIdx = i;
                break;
            }
        }
        if (sepIdx <= 0)
        {
            return false;
        }

        var j = sepIdx;
        while (j < span.Length && char.IsWhiteSpace(span[j]))
        {
            j++;
        }
        if (j >= span.Length)
        {
            return false;
        }

        var img = line[..sepIdx].Trim();
        var txt = line[j..].Trim();
        if (img.Length == 0 || txt.Length == 0)
        {
            return false;
        }

        imageRelPath = img;
        text = txt;
        return true;
    }

    private static string NormalizeDelimiter(string delimiter)
    {
        return delimiter
            .Replace("\\t", "\t", StringComparison.Ordinal)
            .Replace("\\n", "\n", StringComparison.Ordinal)
            .Replace("\\r", "\r", StringComparison.Ordinal);
    }
}
