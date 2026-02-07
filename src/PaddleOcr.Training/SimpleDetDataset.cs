using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Training;

internal sealed class SimpleDetDataset
{
    private readonly List<DetSample> _samples;
    private readonly int _size;

    public SimpleDetDataset(string labelFile, string dataDir, int inputSize)
    {
        _size = inputSize;
        _samples = LoadSamples(labelFile, dataDir);
    }

    public int Count => _samples.Count;

    public IEnumerable<(float[] Images, float[] Masks, int Batch)> GetBatches(int batchSize, bool shuffle, Random rng)
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
            var images = new float[take * 3 * _size * _size];
            var masks = new float[take * _size * _size];
            for (var bi = 0; bi < take; bi++)
            {
                var sample = _samples[indices[offset + bi]];
                var (img, mask) = LoadItem(sample);
                Array.Copy(img, 0, images, bi * img.Length, img.Length);
                Array.Copy(mask, 0, masks, bi * mask.Length, mask.Length);
            }

            yield return (images, masks, take);
        }
    }

    private (float[] Image, float[] Mask) LoadItem(DetSample sample)
    {
        using var img = Image.Load<Rgb24>(sample.ImagePath);
        var srcW = img.Width;
        var srcH = img.Height;
        img.Mutate(x => x.Resize(_size, _size));

        var imageData = new float[3 * _size * _size];
        var hw = _size * _size;
        for (var y = 0; y < _size; y++)
        {
            for (var x = 0; x < _size; x++)
            {
                var p = img[x, y];
                var idx = y * _size + x;
                imageData[idx] = p.R / 255f;
                imageData[hw + idx] = p.G / 255f;
                imageData[2 * hw + idx] = p.B / 255f;
            }
        }

        var mask = new float[_size * _size];
        foreach (var poly in sample.Polygons)
        {
            var scaled = poly
                .Select(p => new[]
                {
                    Math.Clamp(p[0] * _size / Math.Max(1, srcW), 0, _size - 1),
                    Math.Clamp(p[1] * _size / Math.Max(1, srcH), 0, _size - 1)
                })
                .ToArray();
            var xs = scaled.Select(p => p[0]).ToArray();
            var ys = scaled.Select(p => p[1]).ToArray();
            var x1 = Math.Clamp(xs.Min(), 0, _size - 1);
            var y1 = Math.Clamp(ys.Min(), 0, _size - 1);
            var x2 = Math.Clamp(xs.Max(), 0, _size - 1);
            var y2 = Math.Clamp(ys.Max(), 0, _size - 1);
            for (var y = y1; y <= y2; y++)
            {
                for (var x = x1; x <= x2; x++)
                {
                    if (PointInPolygon(x + 0.5f, y + 0.5f, scaled))
                    {
                        mask[y * _size + x] = 1f;
                    }
                }
            }
        }

        return (imageData, mask);
    }

    private static bool PointInPolygon(float x, float y, int[][] poly)
    {
        var inside = false;
        for (var i = 0; i < poly.Length; i++)
        {
            var j = (i + poly.Length - 1) % poly.Length;
            var xi = poly[i][0];
            var yi = poly[i][1];
            var xj = poly[j][0];
            var yj = poly[j][1];
            var intersect = ((yi > y) != (yj > y))
                            && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6f) + xi);
            if (intersect)
            {
                inside = !inside;
            }
        }

        return inside;
    }

    private static List<DetSample> LoadSamples(string labelFile, string dataDir)
    {
        if (!File.Exists(labelFile))
        {
            throw new FileNotFoundException($"Label file not found: {labelFile}");
        }

        var result = new List<DetSample>();
        foreach (var line in File.ReadLines(labelFile))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var tab = line.IndexOf('\t');
            if (tab <= 0 || tab >= line.Length - 1)
            {
                continue;
            }

            var imageRel = line[..tab];
            var json = line[(tab + 1)..];
            var fullPath = Path.IsPathRooted(imageRel) ? imageRel : Path.GetFullPath(Path.Combine(dataDir, imageRel));
            if (!File.Exists(fullPath))
            {
                continue;
            }

            var polys = ParsePolygons(json);
            result.Add(new DetSample(fullPath, polys));
        }

        if (result.Count == 0)
        {
            throw new InvalidOperationException("No valid det samples found.");
        }

        return result;
    }

    private static List<int[][]> ParsePolygons(string json)
    {
        var list = new List<int[][]>();
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
                return list;
            }

            foreach (var item in doc.RootElement.EnumerateArray())
            {
                if (!item.TryGetProperty("points", out var points) || points.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                var poly = new List<int[]>();
                foreach (var p in points.EnumerateArray())
                {
                    if (p.ValueKind == JsonValueKind.Array && p.GetArrayLength() >= 2)
                    {
                        poly.Add([p[0].GetInt32(), p[1].GetInt32()]);
                    }
                }

                if (poly.Count >= 4)
                {
                    list.Add(poly.ToArray());
                }
            }
        }
        catch
        {
            return list;
        }

        return list;
    }
}

internal sealed record DetSample(string ImagePath, List<int[][]> Polygons);
