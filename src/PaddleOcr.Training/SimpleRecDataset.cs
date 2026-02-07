using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Training;

internal sealed class SimpleRecDataset
{
    private readonly List<(string ImagePath, string Text)> _samples;
    private readonly int _height;
    private readonly int _width;
    private readonly int _maxTextLength;
    private readonly IReadOnlyDictionary<char, int> _charToId;

    public SimpleRecDataset(
        string labelFile,
        string dataDir,
        int height,
        int width,
        int maxTextLength,
        IReadOnlyDictionary<char, int> charToId)
    {
        _height = height;
        _width = width;
        _maxTextLength = maxTextLength;
        _charToId = charToId;
        _samples = LoadSamples(labelFile, dataDir);
    }

    public int Count => _samples.Count;

    public IEnumerable<(float[] Images, long[] Labels, int Batch)> GetBatches(int batchSize, bool shuffle, Random rng)
    {
        var indices = Enumerable.Range(0, _samples.Count).ToList();
        if (shuffle)
        {
            for (var i = indices.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        for (var offset = 0; offset < indices.Count; offset += batchSize)
        {
            var take = Math.Min(batchSize, indices.Count - offset);
            var images = new float[take * 3 * _height * _width];
            var labels = new long[take * _maxTextLength];
            for (var bi = 0; bi < take; bi++)
            {
                var sample = _samples[indices[offset + bi]];
                var chw = LoadImageChw(sample.ImagePath);
                Array.Copy(chw, 0, images, bi * chw.Length, chw.Length);

                var seq = Encode(sample.Text, _maxTextLength, _charToId);
                Array.Copy(seq, 0, labels, bi * _maxTextLength, _maxTextLength);
            }

            yield return (images, labels, take);
        }
    }

    public static (IReadOnlyDictionary<char, int> Map, IReadOnlyList<char> Vocab) LoadDictionary(string? dictPath, bool useSpaceChar)
    {
        var chars = new List<char>();
        if (!string.IsNullOrWhiteSpace(dictPath) && File.Exists(dictPath))
        {
            foreach (var line in File.ReadLines(dictPath, Encoding.UTF8))
            {
                var token = line.TrimEnd('\r', '\n');
                if (string.IsNullOrEmpty(token))
                {
                    continue;
                }

                chars.Add(token[0]);
            }
        }
        else
        {
            chars.AddRange("0123456789abcdefghijklmnopqrstuvwxyz");
        }

        if (useSpaceChar && !chars.Contains(' '))
        {
            chars.Add(' ');
        }

        var map = new Dictionary<char, int>();
        var dedup = new List<char>();
        var idx = 1; // 0 is PAD
        foreach (var c in chars)
        {
            if (!map.TryAdd(c, idx))
            {
                continue;
            }

            dedup.Add(c);
            idx++;
        }

        return (map, dedup);
    }

    public static long[] Encode(string text, int maxTextLength, IReadOnlyDictionary<char, int> charToId)
    {
        var result = new long[maxTextLength];
        for (var i = 0; i < maxTextLength && i < text.Length; i++)
        {
            if (charToId.TryGetValue(text[i], out var id))
            {
                result[i] = id;
            }
        }

        return result;
    }

    public static string Decode(long[] ids, IReadOnlyList<char> vocab)
    {
        var sb = new StringBuilder(ids.Length);
        foreach (var id in ids)
        {
            if (id <= 0 || id > vocab.Count)
            {
                continue;
            }

            sb.Append(vocab[(int)id - 1]);
        }

        return sb.ToString();
    }

    private float[] LoadImageChw(string imagePath)
    {
        using var img = Image.Load<Rgb24>(imagePath);
        img.Mutate(x => x.Resize(_width, _height));

        var data = new float[3 * _height * _width];
        var hw = _height * _width;
        for (var y = 0; y < _height; y++)
        {
            for (var x = 0; x < _width; x++)
            {
                var p = img[x, y];
                var idx = y * _width + x;
                data[idx] = p.R / 255f;
                data[hw + idx] = p.G / 255f;
                data[2 * hw + idx] = p.B / 255f;
            }
        }

        return data;
    }

    private static List<(string ImagePath, string Text)> LoadSamples(string labelFile, string dataDir)
    {
        if (!File.Exists(labelFile))
        {
            throw new FileNotFoundException($"Label file not found: {labelFile}");
        }

        var result = new List<(string, string)>();
        foreach (var line in File.ReadLines(labelFile))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var split = line.Split('\t', 2);
            if (split.Length != 2)
            {
                continue;
            }

            var img = split[0].Trim();
            var fullPath = Path.IsPathRooted(img) ? img : Path.GetFullPath(Path.Combine(dataDir, img));
            if (!File.Exists(fullPath))
            {
                continue;
            }

            var text = split[1].Trim();
            if (text.Length == 0)
            {
                continue;
            }

            result.Add((fullPath, text));
        }

        if (result.Count == 0)
        {
            throw new InvalidOperationException("No valid rec samples found.");
        }

        return result;
    }
}
