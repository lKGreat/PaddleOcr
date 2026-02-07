using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Onnx;

public sealed class SystemOnnxRunner
{
    public void Run(SystemOnnxOptions options)
    {
        var imageFiles = EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        var outFile = Path.Combine(options.OutputDir, "system_results.txt");

        using var det = options.DetModelPath is null ? null : new InferenceSession(options.DetModelPath);
        using var rec = new InferenceSession(options.RecModelPath);
        using var cls = options.ClsModelPath is null ? null : new InferenceSession(options.ClsModelPath);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            var result = RunSingle(file, det, cls, rec);
            lines.Add($"{Path.GetFileName(file)}\t{JsonSerializer.Serialize(result)}");
        }

        File.WriteAllLines(outFile, lines);
    }

    private static object RunSingle(string imageFile, InferenceSession? det, InferenceSession? cls, InferenceSession rec)
    {
        var detShapes = det is null ? Array.Empty<int[]>() : RunSession(det, imageFile, 640, 640);
        var clsShapes = cls is null ? Array.Empty<int[]>() : RunSession(cls, imageFile, 48, 192);
        var recShapes = RunSession(rec, imageFile, 48, 320);

        return new
        {
            image = imageFile,
            det_outputs = detShapes,
            cls_outputs = clsShapes,
            rec_outputs = recShapes
        };
    }

    private static int[][] RunSession(InferenceSession session, string imageFile, int defaultH, int defaultW)
    {
        var input = session.InputMetadata.First();
        var tensor = BuildTensor(imageFile, input.Value.Dimensions, defaultH, defaultW);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(input.Key, tensor) };
        using var outputs = session.Run(inputs);
        return outputs.Select(o => o.AsTensor<float>().Dimensions.ToArray()).ToArray();
    }

    private static DenseTensor<float> BuildTensor(string file, IReadOnlyList<int> dims, int defaultH, int defaultW)
    {
        var n = dims.Count > 0 && dims[0] > 0 ? dims[0] : 1;
        var c = dims.Count > 1 && dims[1] > 0 ? dims[1] : 3;
        var h = dims.Count > 2 && dims[2] > 0 ? dims[2] : defaultH;
        var w = dims.Count > 3 && dims[3] > 0 ? dims[3] : defaultW;

        using var img = Image.Load<Rgb24>(file);
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

    private static IEnumerable<string> EnumerateImages(string path)
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
}

public sealed record SystemOnnxOptions(
    string ImageDir,
    string RecModelPath,
    string? DetModelPath,
    string? ClsModelPath,
    string OutputDir);
