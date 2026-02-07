using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Fonts;

namespace PaddleOcr.Inference.Onnx;

public sealed class DetOnnxRunner
{
    public void Run(DetOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        using var det = new InferenceSession(options.DetModelPath);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var output = OnnxRuntimeUtils.RunSession(det, img, 640, 640).FirstOrDefault();
            var boxes = output is null
                ? [PostprocessUtils.FullImageBox(img.Width, img.Height)]
                : PostprocessUtils.DetectBoxes(output.Data, output.Dims, img.Width, img.Height, options.DetThresh);
            var sorted = PostprocessUtils.SortBoxes(boxes);
            var payload = sorted.Select(b => new { transcription = "", points = b.Points }).ToList();
            lines.Add($"{Path.GetFileName(file)}\t{JsonSerializer.Serialize(payload)}");
            OnnxRuntimeUtils.SaveVisualization(file, sorted, options.OutputDir);
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "det_results.txt"), lines);
    }
}

public sealed class RecOnnxRunner
{
    public void Run(RecOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        var charset = CharsetLoader.Load(options.RecCharDictPath, options.UseSpaceChar);
        using var rec = new InferenceSession(options.RecModelPath);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var output = OnnxRuntimeUtils.RunSession(rec, img, 48, 320).FirstOrDefault();
            var recRes = output is null
                ? new RecResult(string.Empty, 0f)
                : PostprocessUtils.DecodeRecCtc(output.Data, output.Dims, charset);
            var payload = recRes.Score >= options.DropScore
                ? new[] { new { text = recRes.Text, score = recRes.Score } }
                : Array.Empty<object>();
            lines.Add($"{Path.GetFileName(file)}\t{JsonSerializer.Serialize(payload)}");
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "rec_results.txt"), lines);
    }
}

public sealed class ClsOnnxRunner
{
    public void Run(ClsOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        using var cls = new InferenceSession(options.ClsModelPath);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var output = OnnxRuntimeUtils.RunSession(cls, img, 48, 192).FirstOrDefault();
            var clsRes = output is null
                ? new ClsResult(options.LabelList.FirstOrDefault() ?? "0", 0f)
                : PostprocessUtils.DecodeCls(output.Data, options.LabelList);
            var payload = new[]
            {
                new { label = clsRes.Label, score = clsRes.Score }
            };
            lines.Add($"{Path.GetFileName(file)}\t{JsonSerializer.Serialize(payload)}");
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "cls_results.txt"), lines);
    }
}

public sealed class SystemOnnxRunner
{
    public void Run(SystemOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        var outFile = Path.Combine(options.OutputDir, "system_results.txt");

        using var det = options.DetModelPath is null ? null : new InferenceSession(options.DetModelPath);
        using var rec = new InferenceSession(options.RecModelPath);
        using var cls = options.ClsModelPath is null ? null : new InferenceSession(options.ClsModelPath);
        var charset = CharsetLoader.Load(options.RecCharDictPath, options.UseSpaceChar);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var original = Image.Load<Rgb24>(file);
            var boxes = GetBoxes(file, original, det, options.DetThresh);
            var sorted = PostprocessUtils.SortBoxes(boxes);

            var items = new List<OcrItem>(sorted.Count);
            foreach (var box in sorted)
            {
                using var crop = OnnxRuntimeUtils.CropBox(original, box);
                if (cls is not null)
                {
                    var clsOut = OnnxRuntimeUtils.RunSession(cls, crop, 48, 192).FirstOrDefault();
                    if (clsOut is not null)
                    {
                        var clsRes = PostprocessUtils.DecodeCls(clsOut.Data, options.LabelList);
                        if (clsRes.Label == "180" && clsRes.Score >= options.ClsThresh)
                        {
                            crop.Mutate(x => x.Rotate(RotateMode.Rotate180));
                        }
                    }
                }

                var recOut = OnnxRuntimeUtils.RunSession(rec, crop, 48, 320).FirstOrDefault();
                var recRes = recOut is null
                    ? new RecResult(string.Empty, 0f)
                    : PostprocessUtils.DecodeRecCtc(recOut.Data, recOut.Dims, charset);

                if (!string.IsNullOrWhiteSpace(recRes.Text) && recRes.Score >= options.DropScore)
                {
                    items.Add(new OcrItem(recRes.Text, box.Points, recRes.Score));
                }
            }

            lines.Add($"{Path.GetFileName(file)}\t{JsonSerializer.Serialize(items)}");
            OnnxRuntimeUtils.SaveVisualization(
                file,
                items.Select(x => new OcrBox(x.Points)).ToList(),
                options.OutputDir,
                items.Select(x => x.Transcription).ToList(),
                items.Select(x => x.Score).ToList());
        }

        File.WriteAllLines(outFile, lines);
    }

    private static List<OcrBox> GetBoxes(string file, Image<Rgb24> image, InferenceSession? det, float thresh)
    {
        if (det is null)
        {
            return [PostprocessUtils.FullImageBox(image.Width, image.Height)];
        }

        var detOut = OnnxRuntimeUtils.RunSession(det, file, 640, 640).FirstOrDefault();
        if (detOut is null)
        {
            return [PostprocessUtils.FullImageBox(image.Width, image.Height)];
        }

        var boxes = PostprocessUtils.DetectBoxes(detOut.Data, detOut.Dims, image.Width, image.Height, thresh);
        return boxes.Count == 0 ? [PostprocessUtils.FullImageBox(image.Width, image.Height)] : boxes;
    }
}

public sealed record TensorOutput(float[] Data, int[] Dims);

public sealed record DetOnnxOptions(string ImageDir, string DetModelPath, string OutputDir, float DetThresh);

public sealed record RecOnnxOptions(
    string ImageDir,
    string RecModelPath,
    string OutputDir,
    string? RecCharDictPath,
    bool UseSpaceChar,
    float DropScore);

public sealed record ClsOnnxOptions(
    string ImageDir,
    string ClsModelPath,
    string OutputDir,
    IReadOnlyList<string> LabelList,
    float ClsThresh);

public sealed record SystemOnnxOptions(
    string ImageDir,
    string RecModelPath,
    string? DetModelPath,
    string? ClsModelPath,
    string OutputDir,
    string? RecCharDictPath,
    bool UseSpaceChar,
    IReadOnlyList<string> LabelList,
    float DropScore,
    float ClsThresh,
    float DetThresh);

public sealed record OcrItem(string Transcription, int[][] Points, float Score);
public sealed record OcrBox(int[][] Points);
public sealed record ClsResult(string Label, float Score);
public sealed record RecResult(string Text, float Score);

public static class CharsetLoader
{
    public static IReadOnlyList<string> Load(string? dictPath, bool useSpaceChar)
    {
        var chars = new List<string> { "" };
        if (!string.IsNullOrWhiteSpace(dictPath) && File.Exists(dictPath))
        {
            chars.AddRange(File.ReadLines(dictPath).Select(x => x.TrimEnd()).Where(x => x.Length > 0));
        }
        else
        {
            chars.AddRange("0123456789abcdefghijklmnopqrstuvwxyz".Select(x => x.ToString()));
        }

        if (useSpaceChar && !chars.Contains(" "))
        {
            chars.Add(" ");
        }

        return chars;
    }
}

public static class OnnxRuntimeUtils
{
    public static List<TensorOutput> RunSession(InferenceSession session, string imageFile, int defaultH, int defaultW)
    {
        using var img = Image.Load<Rgb24>(imageFile);
        return RunSession(session, img, defaultH, defaultW);
    }

    public static List<TensorOutput> RunSession(InferenceSession session, Image<Rgb24> image, int defaultH, int defaultW)
    {
        var input = session.InputMetadata.First();
        var tensor = BuildTensor(image, input.Value.Dimensions, defaultH, defaultW);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(input.Key, tensor) };
        using var outputs = session.Run(inputs);
        var result = new List<TensorOutput>(outputs.Count);
        foreach (var outValue in outputs)
        {
            var t = outValue.AsTensor<float>();
            result.Add(new TensorOutput(t.ToArray(), t.Dimensions.ToArray()));
        }

        return result;
    }

    public static Image<Rgb24> CropBox(Image<Rgb24> source, OcrBox box)
    {
        var xs = box.Points.Select(p => p[0]).ToArray();
        var ys = box.Points.Select(p => p[1]).ToArray();
        var x1 = Math.Max(0, xs.Min());
        var y1 = Math.Max(0, ys.Min());
        var x2 = Math.Min(source.Width - 1, xs.Max());
        var y2 = Math.Min(source.Height - 1, ys.Max());
        var w = Math.Max(1, x2 - x1 + 1);
        var h = Math.Max(1, y2 - y1 + 1);
        return source.Clone(x => x.Crop(new Rectangle(x1, y1, w, h)));
    }

    public static void SaveVisualization(
        string sourceImagePath,
        IReadOnlyList<OcrBox> boxes,
        string outputDir,
        IReadOnlyList<string>? texts = null,
        IReadOnlyList<float>? scores = null)
    {
        Directory.CreateDirectory(outputDir);
        using var img = Image.Load<Rgb24>(sourceImagePath);
        var font = SystemFonts.CreateFont("Arial", 14);
        for (var i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];
            DrawPolygon(img, box.Points, new Rgb24(255, 255, 0));
            if (texts is not null && i < texts.Count)
            {
                var x = Math.Max(0, box.Points[0][0]);
                var y = Math.Max(0, box.Points[0][1] - 18);
                var score = scores is not null && i < scores.Count ? $" {scores[i]:F2}" : string.Empty;
                img.Mutate(ctx => ctx.DrawText($"{texts[i]}{score}", font, Color.Lime, new PointF(x, y)));
            }
        }

        var saveFile = Path.Combine(outputDir, Path.GetFileName(sourceImagePath));
        img.Save(saveFile);
    }

    public static IEnumerable<string> EnumerateImages(string path)
    {
        if (File.Exists(path))
        {
            yield return path;
            yield break;
        }

        if (!Directory.Exists(path))
        {
            yield break;
        }

        var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        };

        foreach (var file in Directory.EnumerateFiles(path, "*.*", SearchOption.TopDirectoryOnly))
        {
            if (exts.Contains(Path.GetExtension(file)))
            {
                yield return file;
            }
        }
    }

    private static DenseTensor<float> BuildTensor(Image<Rgb24> src, IReadOnlyList<int> dims, int defaultH, int defaultW)
    {
        var n = dims.Count > 0 && dims[0] > 0 ? dims[0] : 1;
        var c = dims.Count > 1 && dims[1] > 0 ? dims[1] : 3;
        var h = dims.Count > 2 && dims[2] > 0 ? dims[2] : defaultH;
        var w = dims.Count > 3 && dims[3] > 0 ? dims[3] : defaultW;

        using var img = src.Clone();
        img.Mutate(x => x.Resize(w, h));

        var tensor = new DenseTensor<float>([n, c, h, w]);
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var p = img[x, y];
                tensor[0, 0, y, x] = p.R / 255f;
                tensor[0, 1, y, x] = p.G / 255f;
                tensor[0, 2, y, x] = p.B / 255f;
            }
        }

        return tensor;
    }

    private static void DrawPolygon(Image<Rgb24> img, int[][] points, Rgb24 color)
    {
        if (points.Length < 2)
        {
            return;
        }

        for (var i = 0; i < points.Length; i++)
        {
            var a = points[i];
            var b = points[(i + 1) % points.Length];
            DrawLine(img, a[0], a[1], b[0], b[1], color);
        }
    }

    private static void DrawLine(Image<Rgb24> img, int x0, int y0, int x1, int y1, Rgb24 color)
    {
        var dx = Math.Abs(x1 - x0);
        var sx = x0 < x1 ? 1 : -1;
        var dy = -Math.Abs(y1 - y0);
        var sy = y0 < y1 ? 1 : -1;
        var err = dx + dy;

        while (true)
        {
            if (x0 >= 0 && x0 < img.Width && y0 >= 0 && y0 < img.Height)
            {
                img[x0, y0] = color;
            }

            if (x0 == x1 && y0 == y1)
            {
                break;
            }

            var e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }

            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }
}

public static class PostprocessUtils
{
    public static OcrBox FullImageBox(int w, int h)
    {
        return new OcrBox(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ]);
    }

    public static List<OcrBox> DetectBoxes(float[] data, int[] dims, int imgW, int imgH, float thresh)
    {
        var (map, h, w) = ExtractMap(data, dims);
        if (map.Length == 0 || h <= 0 || w <= 0)
        {
            return [];
        }

        var visited = new bool[h * w];
        var boxes = new List<OcrBox>();
        var minArea = Math.Max(3, (h * w) / 2000);

        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var idx = y * w + x;
                if (visited[idx] || map[idx] < thresh)
                {
                    continue;
                }

                var queue = new Queue<(int X, int Y)>();
                queue.Enqueue((x, y));
                visited[idx] = true;

                var minX = x;
                var minY = y;
                var maxX = x;
                var maxY = y;
                var count = 0;

                while (queue.Count > 0)
                {
                    var (cx, cy) = queue.Dequeue();
                    count++;
                    minX = Math.Min(minX, cx);
                    minY = Math.Min(minY, cy);
                    maxX = Math.Max(maxX, cx);
                    maxY = Math.Max(maxY, cy);

                    foreach (var (nx, ny) in Neighbors(cx, cy, w, h))
                    {
                        var nidx = ny * w + nx;
                        if (visited[nidx] || map[nidx] < thresh)
                        {
                            continue;
                        }

                        visited[nidx] = true;
                        queue.Enqueue((nx, ny));
                    }
                }

                if (count < minArea)
                {
                    continue;
                }

                var x1 = minX * imgW / w;
                var y1 = minY * imgH / h;
                var x2 = Math.Max(x1 + 1, maxX * imgW / w);
                var y2 = Math.Max(y1 + 1, maxY * imgH / h);
                boxes.Add(new OcrBox(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ]));
            }
        }

        return boxes;
    }

    public static List<OcrBox> SortBoxes(List<OcrBox> boxes)
    {
        var sorted = boxes
            .OrderBy(b => b.Points[0][1])
            .ThenBy(b => b.Points[0][0])
            .ToList();

        for (var i = 0; i < sorted.Count - 1; i++)
        {
            for (var j = i; j >= 0; j--)
            {
                var dy = Math.Abs(sorted[j + 1].Points[0][1] - sorted[j].Points[0][1]);
                if (dy < 10 && sorted[j + 1].Points[0][0] < sorted[j].Points[0][0])
                {
                    (sorted[j], sorted[j + 1]) = (sorted[j + 1], sorted[j]);
                }
            }
        }

        return sorted;
    }

    public static ClsResult DecodeCls(float[] logits, IReadOnlyList<string> labels)
    {
        if (logits.Length == 0)
        {
            return new ClsResult(labels.FirstOrDefault() ?? "0", 0f);
        }

        var probs = Softmax(logits);
        var best = 0;
        for (var i = 1; i < probs.Length; i++)
        {
            if (probs[i] > probs[best])
            {
                best = i;
            }
        }

        var label = best < labels.Count ? labels[best] : best.ToString();
        return new ClsResult(label, probs[best]);
    }

    public static RecResult DecodeRecCtc(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 1)
        {
            return new RecResult(string.Empty, 0f);
        }

        var classes = dims.Length > 0 ? dims[^1] : charset.Count;
        if (classes <= 1 || classes > logits.Length)
        {
            classes = charset.Count;
        }

        var time = Math.Max(1, logits.Length / classes);
        if (time * classes != logits.Length)
        {
            return new RecResult(string.Empty, 0f);
        }

        var tokens = new List<int>(time);
        var scores = new List<float>(time);
        for (var t = 0; t < time; t++)
        {
            var slice = new float[classes];
            Array.Copy(logits, t * classes, slice, 0, classes);
            var probs = Softmax(slice);
            var best = 0;
            for (var c = 1; c < classes; c++)
            {
                if (probs[c] > probs[best])
                {
                    best = c;
                }
            }

            tokens.Add(best);
            scores.Add(probs[best]);
        }

        var textChars = new List<string>();
        var keptScores = new List<float>();
        var prev = -1;
        for (var i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            if (token == 0 || token == prev)
            {
                prev = token;
                continue;
            }

            if (token < charset.Count)
            {
                textChars.Add(charset[token]);
                keptScores.Add(scores[i]);
            }

            prev = token;
        }

        var text = string.Concat(textChars);
        var score = keptScores.Count == 0 ? 0f : keptScores.Average();
        return new RecResult(text, score);
    }

    private static (float[] Map, int H, int W) ExtractMap(float[] data, int[] dims)
    {
        if (dims.Length >= 4)
        {
            var c = dims[^3];
            var h = dims[^2];
            var w = dims[^1];
            if (h > 0 && w > 0)
            {
                var map = new float[h * w];
                for (var y = 0; y < h; y++)
                {
                    for (var x = 0; x < w; x++)
                    {
                        var idx = ((0 * c + 0) * h + y) * w + x;
                        if (idx < data.Length)
                        {
                            map[y * w + x] = data[idx];
                        }
                    }
                }

                return (map, h, w);
            }
        }

        if (dims.Length >= 2)
        {
            var h = dims[^2];
            var w = dims[^1];
            if (h > 0 && w > 0 && h * w <= data.Length)
            {
                return (data.Take(h * w).ToArray(), h, w);
            }
        }

        var sq = (int)Math.Sqrt(data.Length);
        return sq * sq == data.Length ? (data, sq, sq) : ([], 0, 0);
    }

    private static IEnumerable<(int X, int Y)> Neighbors(int x, int y, int w, int h)
    {
        if (x > 0) yield return (x - 1, y);
        if (x + 1 < w) yield return (x + 1, y);
        if (y > 0) yield return (x, y - 1);
        if (y + 1 < h) yield return (x, y + 1);
    }

    private static float[] Softmax(float[] x)
    {
        if (x.Length == 0)
        {
            return x;
        }

        var max = x.Max();
        var exps = new float[x.Length];
        var sum = 0f;
        for (var i = 0; i < x.Length; i++)
        {
            exps[i] = (float)Math.Exp(x[i] - max);
            sum += exps[i];
        }

        if (sum <= 0f)
        {
            return Enumerable.Repeat(1f / x.Length, x.Length).ToArray();
        }

        for (var i = 0; i < exps.Length; i++)
        {
            exps[i] /= sum;
        }

        return exps;
    }
}
