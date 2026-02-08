namespace PaddleOcr.Training;

public static class RecLabelLineParser
{
    public static bool TryParse(string line, out string imageRelPath, out string text)
    {
        imageRelPath = string.Empty;
        text = string.Empty;
        if (string.IsNullOrWhiteSpace(line))
        {
            return false;
        }

        var span = line.AsSpan();
        var tabIdx = line.IndexOf('\t');
        if (tabIdx >= 0)
        {
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
}
