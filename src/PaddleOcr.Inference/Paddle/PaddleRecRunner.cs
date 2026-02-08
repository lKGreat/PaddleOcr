using System.Diagnostics;
using System.Text.Json;
using PaddleOcr.Inference.Onnx;
using PaddleOcr.Inference.Rec.Preprocessors;
using PaddleOcr.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Inference.Paddle;

public sealed record RecPaddleOptions(
    string ImageDir,
    string RecModelDirOrFile,
    string OutputDir,
    string? RecCharDictPath,
    bool UseSpaceChar,
    float DropScore,
    RecAlgorithm RecAlgorithm,
    string RecImageShape,
    int MaxTextLength,
    bool RecImageInverse,
    bool RecLogDetail,
    string? PaddleLibDir);

public sealed class RecPaddleRunner
{
    public void Run(RecPaddleOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        var charset = CharsetLoader.Load(options.RecCharDictPath, options.UseSpaceChar);
        var recPost = InferenceComponentRegistry.GetRecPostprocessor(options.RecAlgorithm);
        var preprocessor = RecPreprocessorFactory.Create(options.RecAlgorithm, options.RecImageInverse);
        var (targetC, targetH, targetW) = ParseImageShape(options.RecImageShape, options.RecAlgorithm);

        using var native = PaddleNative.Create(options.PaddleLibDir);
        using var predictor = native.CreatePredictor(options.RecModelDirOrFile);

        var lines = new List<string>(imageFiles.Count);
        var traces = new List<RecPaddleTraceItem>(imageFiles.Count);
        var totalWatch = Stopwatch.StartNew();
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var preprocessWatch = Stopwatch.StartNew();
            var preResult = preprocessor.Process(img, targetC, targetH, targetW);
            preprocessWatch.Stop();

            var inferWatch = Stopwatch.StartNew();
            var (data, dims) = predictor.Run(preResult.Data, preResult.Dims, preResult.ValidRatio);
            inferWatch.Stop();

            var recRes = recPost(data, dims, charset);
            AppendResult(lines, file, recRes, options.DropScore);
            traces.Add(new RecPaddleTraceItem(
                Path.GetFileName(file),
                preprocessWatch.Elapsed.TotalMilliseconds,
                inferWatch.Elapsed.TotalMilliseconds,
                recRes.Score,
                recRes.Text.Length));
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "rec_results.txt"), lines);
        if (options.RecLogDetail)
        {
            WriteRecProfile(options.OutputDir, imageFiles.Count, traces, totalWatch.Elapsed.TotalMilliseconds);
        }
    }

    private static void AppendResult(List<string> lines, string filePath, RecResult recRes, float dropScore)
    {
        var payload = recRes.Score >= dropScore
            ? new[] { new { text = recRes.Text, score = recRes.Score } }
            : Array.Empty<object>();
        lines.Add($"{Path.GetFileName(filePath)}\t{JsonSerializer.Serialize(payload)}");
    }

    private static (int C, int H, int W) ParseImageShape(string shape, RecAlgorithm algorithm)
    {
        var parts = shape.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length == 3 &&
            int.TryParse(parts[0], out var c) &&
            int.TryParse(parts[1], out var h) &&
            int.TryParse(parts[2], out var w))
        {
            return (c, h, w);
        }

        return algorithm.GetDefaultImageShape();
    }

    private static void WriteRecProfile(
        string outputDir,
        int imageCount,
        IReadOnlyList<RecPaddleTraceItem> traces,
        double totalMs)
    {
        var profile = new
        {
            image_count = imageCount,
            traced_count = traces.Count,
            total_ms = totalMs,
            avg_preprocess_ms = traces.Count == 0 ? 0d : traces.Average(x => x.PreprocessMs),
            avg_inference_ms = traces.Count == 0 ? 0d : traces.Average(x => x.InferenceMs),
            avg_score = traces.Count == 0 ? 0d : traces.Average(x => x.Score)
        };

        File.WriteAllText(
            Path.Combine(outputDir, "rec_runtime_profile.json"),
            JsonSerializer.Serialize(profile, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllLines(
            Path.Combine(outputDir, "rec_trace.jsonl"),
            traces.Select(x => JsonSerializer.Serialize(x)));
    }
}

internal sealed record RecPaddleTraceItem(
    string File,
    double PreprocessMs,
    double InferenceMs,
    float Score,
    int TextLength);
