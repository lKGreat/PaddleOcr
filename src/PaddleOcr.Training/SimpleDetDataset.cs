using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Training;

internal sealed class SimpleDetDataset
{
    private readonly List<DetSample> _samples;
    private readonly int _size;

    public SimpleDetDataset(
        string labelFile,
        string dataDir,
        int inputSize,
        string invalidSamplePolicy = "skip",
        int minValidSamples = 1)
    {
        _size = inputSize;
        var loaded = LoadSamples(labelFile, dataDir, invalidSamplePolicy, minValidSamples);
        _samples = loaded.Samples;
        Audit = loaded.Audit;
    }

    public int Count => _samples.Count;
    public DetDataAudit Audit { get; }

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

    private static (List<DetSample> Samples, DetDataAudit Audit) LoadSamples(
        string labelFile,
        string dataDir,
        string invalidSamplePolicy,
        int minValidSamples)
    {
        if (!File.Exists(labelFile))
        {
            throw new FileNotFoundException($"Label file not found: {labelFile}");
        }

        var policy = invalidSamplePolicy.Equals("fail", StringComparison.OrdinalIgnoreCase) ? "fail" : "skip";
        var minSamples = Math.Max(1, minValidSamples);
        var audit = new DetDataAudit
        {
            LabelFile = labelFile,
            DataDir = dataDir,
            InvalidSamplePolicy = policy
        };
        var result = new List<DetSample>();
        var lineNumber = 0;
        foreach (var line in File.ReadLines(labelFile))
        {
            lineNumber++;
            audit.TotalLines++;
            if (string.IsNullOrWhiteSpace(line))
            {
                AddSkipped(audit, "empty_line", lineNumber, line);
                continue;
            }

            var tab = line.IndexOf('\t');
            if (tab <= 0 || tab >= line.Length - 1)
            {
                if (OnInvalid(policy, "missing_tab_separator", lineNumber, line, audit, out var ex))
                {
                    throw ex;
                }

                continue;
            }

            var imageRel = line[..tab];
            var json = line[(tab + 1)..];
            var fullPath = Path.IsPathRooted(imageRel) ? imageRel : Path.GetFullPath(Path.Combine(dataDir, imageRel));
            if (!File.Exists(fullPath))
            {
                if (OnInvalid(policy, "missing_image_file", lineNumber, imageRel, audit, out var ex))
                {
                    throw ex;
                }

                continue;
            }

            if (!TryParsePolygons(json, out var polys, out var reason))
            {
                if (OnInvalid(policy, reason, lineNumber, imageRel, audit, out var ex))
                {
                    throw ex;
                }

                continue;
            }

            result.Add(new DetSample(fullPath, polys));
        }

        audit.ValidSamples = result.Count;
        if (result.Count < minSamples)
        {
            throw new InvalidOperationException(
                $"det dataset has insufficient valid samples: {result.Count} < {minSamples}. " +
                $"label_file={labelFile}");
        }

        return (result, audit);
    }

    private static bool OnInvalid(
        string policy,
        string reason,
        int lineNumber,
        string detail,
        DetDataAudit audit,
        out InvalidOperationException exception)
    {
        exception = new InvalidOperationException(
            $"invalid det sample at line {lineNumber}: reason={reason}, detail={detail}");
        if (policy == "fail")
        {
            return true;
        }

        AddSkipped(audit, reason, lineNumber, detail);
        return false;
    }

    private static void AddSkipped(DetDataAudit audit, string reason, int lineNumber, string detail)
    {
        audit.SkippedSamples++;
        audit.SkippedByReason[reason] = audit.SkippedByReason.GetValueOrDefault(reason) + 1;
        if (audit.Examples.Count < 10)
        {
            audit.Examples.Add($"line={lineNumber}, reason={reason}, detail={detail}");
        }
    }

    private static bool TryParsePolygons(string json, out List<int[][]> polygons, out string reason)
    {
        polygons = [];
        reason = "invalid_json";
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
                reason = "invalid_json_root";
                return false;
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
                        if (!p[0].TryGetInt32(out var x) || !p[1].TryGetInt32(out var y))
                        {
                            continue;
                        }

                        poly.Add([x, y]);
                    }
                }

                if (poly.Count >= 4)
                {
                    polygons.Add(poly.ToArray());
                }
            }

            if (polygons.Count == 0)
            {
                reason = "empty_or_invalid_polygon";
                return false;
            }

            reason = string.Empty;
            return true;
        }
        catch
        {
            reason = "invalid_json";
            return false;
        }
    }
}

internal sealed record DetSample(string ImagePath, List<int[][]> Polygons);

internal sealed class DetDataAudit
{
    public string LabelFile { get; init; } = string.Empty;
    public string DataDir { get; init; } = string.Empty;
    public string InvalidSamplePolicy { get; init; } = "skip";
    public int TotalLines { get; set; }
    public int ValidSamples { get; set; }
    public int SkippedSamples { get; set; }
    public Dictionary<string, int> SkippedByReason { get; } = new(StringComparer.OrdinalIgnoreCase);
    public List<string> Examples { get; } = [];
}
