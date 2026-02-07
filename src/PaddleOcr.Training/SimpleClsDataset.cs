using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Training;

internal sealed class SimpleClsDataset
{
    private readonly List<(string ImagePath, long Label)> _samples;
    private readonly int _height;
    private readonly int _width;

    public SimpleClsDataset(string labelFile, string dataDir, int height, int width)
    {
        _height = height;
        _width = width;
        _samples = LoadSamples(labelFile, dataDir);
    }

    public int Count => _samples.Count;

    public IEnumerable<(float[] Images, long[] Labels, int Batch)> GetBatches(int batchSize, bool shuffle, Random random)
    {
        var indices = Enumerable.Range(0, _samples.Count).ToList();
        if (shuffle)
        {
            for (var i = indices.Count - 1; i > 0; i--)
            {
                var j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        for (var offset = 0; offset < indices.Count; offset += batchSize)
        {
            var take = Math.Min(batchSize, indices.Count - offset);
            var images = new float[take * 3 * _height * _width];
            var labels = new long[take];
            for (var bi = 0; bi < take; bi++)
            {
                var sample = _samples[indices[offset + bi]];
                var chw = LoadImageChw(sample.ImagePath);
                Array.Copy(chw, 0, images, bi * chw.Length, chw.Length);
                labels[bi] = sample.Label;
            }

            yield return (images, labels, take);
        }
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

    private static List<(string ImagePath, long Label)> LoadSamples(string labelFile, string dataDir)
    {
        if (!File.Exists(labelFile))
        {
            throw new FileNotFoundException($"Label file not found: {labelFile}");
        }

        var result = new List<(string, long)>();
        foreach (var line in File.ReadLines(labelFile))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var parts = line.Split('\t', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2)
            {
                continue;
            }

            var img = parts[0].Trim();
            var fullPath = Path.IsPathRooted(img) ? img : Path.GetFullPath(Path.Combine(dataDir, img));
            if (!File.Exists(fullPath))
            {
                continue;
            }

            if (!long.TryParse(parts[1].Trim(), out var label))
            {
                continue;
            }

            result.Add((fullPath, label));
        }

        if (result.Count == 0)
        {
            throw new InvalidOperationException("No valid samples found in label file.");
        }

        return result;
    }
}

