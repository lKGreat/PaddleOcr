using System.Text.Json;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PaddleOcr.Models;
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
        var inputBuilderName = DetInferenceExtensions.ResolveInputBuilder(options.DetAlgorithm);
        var inputDims = det.InputMetadata.First().Value.Dimensions.ToArray();

        var lines = new List<string>(imageFiles.Count);
        var predictionByImage = new Dictionary<string, List<OcrBox>>(StringComparer.OrdinalIgnoreCase);
        var runtimeByImage = new Dictionary<string, DetRuntimeProfile>(StringComparer.OrdinalIgnoreCase);
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var (sorted, profile) = options.UseSlice
                ? PredictWithSlice(options, det, img, inputDims, inputBuilderName)
                : PredictSingle(options, det, img, inputDims, inputBuilderName);
            var imageName = Path.GetFileName(file);
            runtimeByImage[imageName] = profile;
            var payload = sorted.Select(b => new { transcription = "", points = b.Points }).ToList();
            lines.Add($"{imageName}\t{JsonSerializer.Serialize(payload)}");
            predictionByImage[imageName] = sorted;
            OnnxRuntimeUtils.SaveVisualization(file, sorted, options.OutputDir);
        }

        var resultPath = string.IsNullOrWhiteSpace(options.SaveResPath)
            ? Path.Combine(options.OutputDir, "det_results.txt")
            : options.SaveResPath;
        var resultDir = Path.GetDirectoryName(Path.GetFullPath(resultPath));
        if (!string.IsNullOrWhiteSpace(resultDir))
        {
            Directory.CreateDirectory(resultDir);
        }

        File.WriteAllLines(resultPath, lines);
        DetInferenceExtensions.WriteDetMetrics(options, options.OutputDir, predictionByImage, runtimeByImage);
    }

    private static (List<OcrBox> Boxes, DetRuntimeProfile Profile) PredictSingle(
        DetOnnxOptions options,
        InferenceSession det,
        Image<Rgb24> image,
        IReadOnlyList<int> inputDims,
        string inputBuilderName)
    {
        var (inputWidth, inputHeight) = DetInferenceExtensions.ResolveDetInputSize(options, image.Width, image.Height, inputDims);
        var totalWatch = Stopwatch.StartNew();
        var profiled = OnnxRuntimeUtils.RunSessionProfiled(det, image, inputHeight, inputWidth, inputBuilderName);
        var outputs = profiled.Outputs;
        var postWatch = Stopwatch.StartNew();
        var boxes = outputs.Count == 0
            ? [PostprocessUtils.FullImageBox(image.Width, image.Height)]
            : DetInferenceExtensions.DecodeBoxes(options, outputs, image.Width, image.Height);
        var sorted = PostprocessUtils.SortBoxes(boxes.Count == 0 ? [PostprocessUtils.FullImageBox(image.Width, image.Height)] : boxes);
        postWatch.Stop();
        totalWatch.Stop();
        return (
            sorted,
            new DetRuntimeProfile(
                profiled.PreprocessMs,
                profiled.InferenceMs,
                postWatch.Elapsed.TotalMilliseconds,
                totalWatch.Elapsed.TotalMilliseconds,
                image.Width,
                image.Height,
                inputWidth,
                inputHeight));
    }

    private static (List<OcrBox> Boxes, DetRuntimeProfile Profile) PredictWithSlice(
        DetOnnxOptions options,
        InferenceSession det,
        Image<Rgb24> image,
        IReadOnlyList<int> inputDims,
        string inputBuilderName)
    {
        var limit = Math.Max(32, options.DetLimitSideLen);
        var ratioH = image.Width == 0 ? 0f : image.Height / (float)image.Width;
        var ratioW = image.Height == 0 ? 0f : image.Width / (float)image.Height;
        if (!(ratioH > 2f && image.Height > limit) && !(ratioW > 3f && image.Width > limit * 3))
        {
            return PredictSingle(options, det, image, inputDims, inputBuilderName);
        }

        var minBoundDistance = Math.Max(1, options.SliceMinBoundDistance);
        var merged = new List<OcrBox>();
        var preprocessMs = 0d;
        var inferenceMs = 0d;
        var postprocessMs = 0d;
        var totalMs = 0d;
        var (baseInputW, baseInputH) = DetInferenceExtensions.ResolveDetInputSize(options, image.Width, image.Height, inputDims);

        if (ratioH > 2f && image.Height > limit)
        {
            var startH = 0;
            var endH = 0;
            while (endH <= image.Height)
            {
                endH = startH + image.Width * 3 / 4;
                var cropHeight = Math.Min(endH, image.Height) - startH;
                if (cropHeight <= 0)
                {
                    break;
                }

                using var sub = image.Clone(x => x.Crop(new Rectangle(0, startH, image.Width, cropHeight)));
                var (subBoxes, subProfile) = PredictSingle(options, det, sub, inputDims, inputBuilderName);
                preprocessMs += subProfile.PreprocessMs;
                inferenceMs += subProfile.InferenceMs;
                postprocessMs += subProfile.PostprocessMs;
                totalMs += subProfile.TotalMs;
                var offset = startH;

                if (subBoxes.Count == 0 || image.Width - subBoxes.Max(MaxPointY) > minBoundDistance)
                {
                    startH = endH;
                }
                else
                {
                    subBoxes = subBoxes.OrderBy(GetBoxPoint2Y).ToList();
                    var bottomLine = subBoxes.Count <= 1 ? 0 : subBoxes.Take(subBoxes.Count - 1).Max(GetBoxPoint2Y);
                    if (bottomLine > 0)
                    {
                        startH += bottomLine;
                        subBoxes = subBoxes.Where(x => GetBoxPoint2Y(x) <= bottomLine).ToList();
                    }
                    else
                    {
                        startH = endH;
                    }
                }

                merged.AddRange(subBoxes.Select(x => OffsetBox(x, 0, offset, image.Width, image.Height)));
            }
        }
        else
        {
            var startW = 0;
            var endW = 0;
            while (endW <= image.Width)
            {
                endW = startW + image.Height * 3 / 4;
                var cropWidth = Math.Min(endW, image.Width) - startW;
                if (cropWidth <= 0)
                {
                    break;
                }

                using var sub = image.Clone(x => x.Crop(new Rectangle(startW, 0, cropWidth, image.Height)));
                var (subBoxes, subProfile) = PredictSingle(options, det, sub, inputDims, inputBuilderName);
                preprocessMs += subProfile.PreprocessMs;
                inferenceMs += subProfile.InferenceMs;
                postprocessMs += subProfile.PostprocessMs;
                totalMs += subProfile.TotalMs;
                var offset = startW;

                if (subBoxes.Count == 0 || image.Height - subBoxes.Max(MaxPointX) > minBoundDistance)
                {
                    startW = endW;
                }
                else
                {
                    subBoxes = subBoxes.OrderBy(GetBoxPoint2X).ToList();
                    var rightLine = subBoxes.Count <= 1 ? 0 : subBoxes.Take(subBoxes.Count - 1).Max(GetBoxPoint2X);
                    if (rightLine > 0)
                    {
                        startW += rightLine;
                        subBoxes = subBoxes.Where(x => GetBoxPoint2X(x) <= rightLine).ToList();
                    }
                    else
                    {
                        startW = endW;
                    }
                }

                merged.AddRange(subBoxes.Select(x => OffsetBox(x, offset, 0, image.Width, image.Height)));
            }
        }

        if (merged.Count == 0)
        {
            merged.Add(PostprocessUtils.FullImageBox(image.Width, image.Height));
        }

        merged = SuppressByIou(merged, options.SliceMergeIou);
        var sorted = PostprocessUtils.SortBoxes(merged);
        var profile = new DetRuntimeProfile(
            preprocessMs,
            inferenceMs,
            postprocessMs,
            totalMs,
            image.Width,
            image.Height,
            baseInputW,
            baseInputH);
        return (sorted, profile);
    }

    private static OcrBox OffsetBox(OcrBox box, int offsetX, int offsetY, int width, int height)
    {
        var points = box.Points
            .Select(p => new[]
            {
                Math.Clamp(p[0] + offsetX, 0, Math.Max(0, width - 1)),
                Math.Clamp(p[1] + offsetY, 0, Math.Max(0, height - 1))
            })
            .ToArray();
        return new OcrBox(points);
    }

    private static List<OcrBox> SuppressByIou(IReadOnlyList<OcrBox> boxes, float iouThreshold)
    {
        var threshold = Math.Clamp(iouThreshold, 0.01f, 0.99f);
        var ordered = boxes.OrderByDescending(BoxArea).ToList();
        var kept = new List<OcrBox>(ordered.Count);
        foreach (var box in ordered)
        {
            var suppress = kept.Any(x => ComputeIou(x, box) >= threshold);
            if (!suppress)
            {
                kept.Add(box);
            }
        }

        return kept;
    }

    private static float BoxArea(OcrBox box)
    {
        var (x1, y1, x2, y2) = ToRect(box);
        return Math.Max(1f, (x2 - x1 + 1f) * (y2 - y1 + 1f));
    }

    private static float ComputeIou(OcrBox a, OcrBox b)
    {
        var (ax1, ay1, ax2, ay2) = ToRect(a);
        var (bx1, by1, bx2, by2) = ToRect(b);
        var ix1 = Math.Max(ax1, bx1);
        var iy1 = Math.Max(ay1, by1);
        var ix2 = Math.Min(ax2, bx2);
        var iy2 = Math.Min(ay2, by2);
        if (ix2 <= ix1 || iy2 <= iy1)
        {
            return 0f;
        }

        var inter = (ix2 - ix1) * (iy2 - iy1);
        var areaA = Math.Max(1f, (ax2 - ax1 + 1f) * (ay2 - ay1 + 1f));
        var areaB = Math.Max(1f, (bx2 - bx1 + 1f) * (by2 - by1 + 1f));
        var union = areaA + areaB - inter;
        return union <= 0f ? 0f : inter / union;
    }

    private static (float X1, float Y1, float X2, float Y2) ToRect(OcrBox box)
    {
        var xs = box.Points.Select(p => (float)p[0]).ToArray();
        var ys = box.Points.Select(p => (float)p[1]).ToArray();
        return (xs.Min(), ys.Min(), xs.Max(), ys.Max());
    }

    private static int GetBoxPoint2Y(OcrBox box)
    {
        var idx = box.Points.Length <= 2 ? box.Points.Length - 1 : 2;
        return idx < 0 ? 0 : box.Points[idx][1];
    }

    private static int GetBoxPoint2X(OcrBox box)
    {
        var idx = box.Points.Length <= 2 ? box.Points.Length - 1 : 2;
        return idx < 0 ? 0 : box.Points[idx][0];
    }

    private static int MaxPointY(OcrBox box)
    {
        return box.Points.Length == 0 ? 0 : box.Points.Max(p => p[1]);
    }

    private static int MaxPointX(OcrBox box)
    {
        return box.Points.Length == 0 ? 0 : box.Points.Max(p => p[0]);
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
        var recPost = InferenceComponentRegistry.GetRecPostprocessor(options.RecAlgorithm);
        var preprocessor = Rec.Preprocessors.RecPreprocessorFactory.Create(options.RecAlgorithm, options.RecImageInverse);
        using var rec = new InferenceSession(options.RecModelPath);

        // 解析图像形状
        var (targetC, targetH, targetW) = ParseImageShape(options.RecImageShape, options.RecAlgorithm);
        var batchSize = Math.Max(1, options.RecBatchNum);
        var inputName = rec.InputMetadata.First().Key;

        var lines = new List<string>(imageFiles.Count);
        var recTrace = new List<RecTraceItem>(imageFiles.Count);
        var fallbackCount = 0;
        var totalWatch = Stopwatch.StartNew();

        // 按 batch 分批处理
        for (var batchStart = 0; batchStart < imageFiles.Count; batchStart += batchSize)
        {
            var batchEnd = Math.Min(batchStart + batchSize, imageFiles.Count);
            var currentBatchSize = batchEnd - batchStart;

            // 预处理批次中的所有图像
            var preprocessWatch = Stopwatch.StartNew();
            var batchPreResults = new Rec.RecPreprocessResult[currentBatchSize];
            for (var i = 0; i < currentBatchSize; i++)
            {
                using var img = Image.Load<Rgb24>(imageFiles[batchStart + i]);
                batchPreResults[i] = preprocessor.Process(img, targetC, targetH, targetW);
            }
            preprocessWatch.Stop();
            var preprocessPerImageMs = preprocessWatch.Elapsed.TotalMilliseconds / Math.Max(1, currentBatchSize);

            // 检查模型是否接受 valid_ratio 额外输入
            var hasValidRatioInput = rec.InputMetadata.ContainsKey("valid_ratio");

            if (currentBatchSize == 1)
            {
                // 单图像：直接推理
                var preResult = batchPreResults[0];
                var tensor = new DenseTensor<float>(preResult.Data, preResult.Dims);
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

                if (hasValidRatioInput)
                {
                    var vrData = new float[] { preResult.ValidRatio };
                    var vrTensor = new DenseTensor<float>(vrData, new[] { 1 });
                    inputs.Add(NamedOnnxValue.CreateFromTensor("valid_ratio", vrTensor));
                }

                var inferWatch = Stopwatch.StartNew();
                using var outputs = rec.Run(inputs);
                inferWatch.Stop();
                var outTensor = outputs.First().AsTensor<float>();
                var recRes = recPost(outTensor.ToArray(), outTensor.Dimensions.ToArray(), charset);
                AppendResult(lines, imageFiles[batchStart], recRes, options.DropScore);
                recTrace.Add(new RecTraceItem(
                    Path.GetFileName(imageFiles[batchStart]),
                    preprocessPerImageMs,
                    inferWatch.Elapsed.TotalMilliseconds,
                    0d,
                    false,
                    recRes.Score,
                    recRes.Text.Length));
            }
            else
            {
                // 多图像：合并为 batch tensor 一次推理
                // 所有预处理结果的 channel 和 height 相同，width 也相同（已 pad 到 targetW）
                var channels = batchPreResults[0].Dims.Length >= 4 ? batchPreResults[0].Dims[1] : targetC;
                var singleSize = channels * targetH * targetW;
                var batchData = new float[currentBatchSize * singleSize];

                for (var i = 0; i < currentBatchSize; i++)
                {
                    var src = batchPreResults[i].Data;
                    Array.Copy(src, 0, batchData, i * singleSize, Math.Min(src.Length, singleSize));
                }

                var batchDims = new[] { currentBatchSize, channels, targetH, targetW };
                var batchTensor = new DenseTensor<float>(batchData, batchDims);
                var batchInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, batchTensor) };

                // 如果模型支持动态 batch，一次推理整个 batch
                try
                {
                    var inferWatch = Stopwatch.StartNew();
                    using var batchOutputs = rec.Run(batchInputs);
                    inferWatch.Stop();
                    var batchOutTensor = batchOutputs.First().AsTensor<float>();
                    var batchOutDims = batchOutTensor.Dimensions.ToArray();
                    var batchOutData = batchOutTensor.ToArray();
                    var inferPerImageMs = inferWatch.Elapsed.TotalMilliseconds / Math.Max(1, currentBatchSize);

                    // 按样本拆分输出
                    if (batchOutDims.Length >= 2 && batchOutDims[0] == currentBatchSize)
                    {
                        var perSampleSize = batchOutData.Length / currentBatchSize;
                        var singleOutDims = new int[batchOutDims.Length];
                        Array.Copy(batchOutDims, singleOutDims, batchOutDims.Length);
                        singleOutDims[0] = 1;

                        for (var i = 0; i < currentBatchSize; i++)
                        {
                            var sampleData = new float[perSampleSize];
                            Array.Copy(batchOutData, i * perSampleSize, sampleData, 0, perSampleSize);
                            var recRes = recPost(sampleData, singleOutDims, charset);
                            AppendResult(lines, imageFiles[batchStart + i], recRes, options.DropScore);
                            recTrace.Add(new RecTraceItem(
                                Path.GetFileName(imageFiles[batchStart + i]),
                                preprocessPerImageMs,
                                inferPerImageMs,
                                0d,
                                false,
                                recRes.Score,
                                recRes.Text.Length));
                        }
                    }
                    else
                    {
                        // 如果输出维度不匹配，回退到逐个解码
                        fallbackCount += currentBatchSize;
                        FallbackSingleInference(rec, inputName, batchPreResults, imageFiles, batchStart, currentBatchSize, recPost, charset, options.DropScore, lines, recTrace, preprocessPerImageMs);
                    }
                }
                catch
                {
                    // 如果模型不支持动态 batch，回退到逐个推理
                    fallbackCount += currentBatchSize;
                    FallbackSingleInference(rec, inputName, batchPreResults, imageFiles, batchStart, currentBatchSize, recPost, charset, options.DropScore, lines, recTrace, preprocessPerImageMs);
                }
            }
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "rec_results.txt"), lines);
        if (options.RecLogDetail)
        {
            WriteRecProfile(options.OutputDir, imageFiles.Count, recTrace, fallbackCount, totalWatch.Elapsed.TotalMilliseconds);
        }
    }

    private static void FallbackSingleInference(
        InferenceSession rec,
        string inputName,
        Rec.RecPreprocessResult[] preResults,
        IReadOnlyList<string> imageFiles,
        int batchStart,
        int count,
        Func<float[], int[], IReadOnlyList<string>, Models.RecResult> recPost,
        IReadOnlyList<string> charset,
        float dropScore,
        List<string> lines,
        List<RecTraceItem> traces,
        double preprocessPerImageMs)
    {
        // 检查模型是否接受 valid_ratio 额外输入（SAR/RobustScanner/SATRN 等算法需要）
        var hasValidRatioInput = rec.InputMetadata.ContainsKey("valid_ratio");

        for (var i = 0; i < count; i++)
        {
            var preResult = preResults[i];
            var tensor = new DenseTensor<float>(preResult.Data, preResult.Dims);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            // 如果模型有 valid_ratio 输入，传递预处理结果中的 ValidRatio
            if (hasValidRatioInput)
            {
                var vrData = new float[] { preResult.ValidRatio };
                var vrTensor = new DenseTensor<float>(vrData, new[] { 1 });
                inputs.Add(NamedOnnxValue.CreateFromTensor("valid_ratio", vrTensor));
            }

            var inferWatch = Stopwatch.StartNew();
            using var outputs = rec.Run(inputs);
            inferWatch.Stop();
            var outTensor = outputs.First().AsTensor<float>();
            var recRes = recPost(outTensor.ToArray(), outTensor.Dimensions.ToArray(), charset);
            AppendResult(lines, imageFiles[batchStart + i], recRes, dropScore);
            traces.Add(new RecTraceItem(
                Path.GetFileName(imageFiles[batchStart + i]),
                preprocessPerImageMs,
                inferWatch.Elapsed.TotalMilliseconds,
                0d,
                true,
                recRes.Score,
                recRes.Text.Length));
        }
    }

    private static void AppendResult(List<string> lines, string filePath, Models.RecResult recRes, float dropScore)
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
        IReadOnlyList<RecTraceItem> traces,
        int fallbackCount,
        double totalMs)
    {
        var profile = new
        {
            image_count = imageCount,
            traced_count = traces.Count,
            fallback_single_count = fallbackCount,
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
        var clsPost = InferenceComponentRegistry.GetClsPostprocessor();

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var output = OnnxRuntimeUtils.RunSession(cls, img, 48, 192).FirstOrDefault();
            var clsRes = output is null
                ? new ClsResult(options.LabelList.FirstOrDefault() ?? "0", 0f)
                : clsPost(output.Data, options.LabelList);
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
        var recPost = InferenceComponentRegistry.GetRecPostprocessor();

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
                    : recPost(recOut.Data, recOut.Dims, charset);

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

        var detPost = InferenceComponentRegistry.GetDetPostprocessor();
        var boxes = detPost(detOut.Data, detOut.Dims, image.Width, image.Height, thresh);
        return boxes.Count == 0 ? [PostprocessUtils.FullImageBox(image.Width, image.Height)] : boxes;
    }
}

public sealed class SrOnnxRunner
{
    public void Run(SrOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        using var sr = new InferenceSession(options.SrModelPath);
        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var img = Image.Load<Rgb24>(file);
            var output = OnnxRuntimeUtils.RunSession(sr, img, img.Height, img.Width).FirstOrDefault();
            if (output is null || output.Dims.Length < 4)
            {
                continue;
            }

            using var outImg = TensorToImage(output.Data, output.Dims);
            var saveFile = Path.Combine(options.OutputDir, Path.GetFileName(file));
            outImg.Save(saveFile);
            lines.Add($"{Path.GetFileName(file)}\t{saveFile}");
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "sr_results.txt"), lines);
    }

    private static Image<Rgb24> TensorToImage(float[] data, int[] dims)
    {
        var c = dims[1];
        var h = dims[2];
        var w = dims[3];
        if (c < 3)
        {
            throw new InvalidOperationException("SR output channel count should be >=3.");
        }

        var img = new Image<Rgb24>(w, h);
        var hw = h * w;
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var idx = y * w + x;
                var r = Clamp255(data[idx]);
                var g = Clamp255(data[hw + idx]);
                var b = Clamp255(data[2 * hw + idx]);
                img[x, y] = new Rgb24((byte)r, (byte)g, (byte)b);
            }
        }

        return img;
    }

    private static int Clamp255(float v)
    {
        var value = v;
        if (value <= 1f)
        {
            value *= 255f;
        }

        return Math.Clamp((int)MathF.Round(value), 0, 255);
    }
}

public sealed class TableOnnxRunner
{
    public void Run(TableOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        using var table = new InferenceSession(options.TableModelPath);
        using var det = options.DetModelPath is null ? null : new InferenceSession(options.DetModelPath);
        using var rec = options.RecModelPath is null ? null : new InferenceSession(options.RecModelPath);
        var charset = CharsetLoader.Load(options.RecCharDictPath, options.UseSpaceChar);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var image = Image.Load<Rgb24>(file);
            var tableOutputs = OnnxRuntimeUtils.RunSession(table, image, image.Height, image.Width);
            var ocr = OcrPipelineHelper.ExtractOcrItems(image, file, det, rec, charset, options.DropScore, options.DetThresh);
            lines.Add(TableResultSerializer.BuildLine(file, tableOutputs, ocr));
            if (ocr.Count > 0)
            {
                OnnxRuntimeUtils.SaveVisualization(
                    file,
                    ocr.Select(x => new OcrBox(x.Points)).ToList(),
                    options.OutputDir,
                    ocr.Select(x => x.Transcription).ToList(),
                    ocr.Select(x => x.Score).ToList());
            }
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, "table_results.txt"), lines);
    }
}

public sealed class KieOnnxRunner
{
    public void Run(KieOnnxOptions options)
    {
        var imageFiles = OnnxRuntimeUtils.EnumerateImages(options.ImageDir).ToList();
        if (imageFiles.Count == 0)
        {
            throw new InvalidOperationException($"No image found in: {options.ImageDir}");
        }

        Directory.CreateDirectory(options.OutputDir);
        using var kie = new InferenceSession(options.KieModelPath);
        using var det = options.DetModelPath is null ? null : new InferenceSession(options.DetModelPath);
        using var rec = options.RecModelPath is null ? null : new InferenceSession(options.RecModelPath);
        var charset = CharsetLoader.Load(options.RecCharDictPath, options.UseSpaceChar);

        var lines = new List<string>(imageFiles.Count);
        foreach (var file in imageFiles)
        {
            using var image = Image.Load<Rgb24>(file);
            var tensors = OnnxRuntimeUtils.RunSession(kie, image, image.Height, image.Width);
            var ocr = OcrPipelineHelper.ExtractOcrItems(image, file, det, rec, charset, options.DropScore, options.DetThresh);
            var payload = new
            {
                task = options.TaskName,
                tensors = tensors.Select((x, i) => new { index = i, dims = x.Dims, size = x.Data.Length }).ToList(),
                ocr = ocr.Select(x => new { text = x.Transcription, score = x.Score, points = x.Points }).ToList()
            };
            lines.Add($"{Path.GetFileName(file)}\t{JsonSerializer.Serialize(payload)}");
        }

        File.WriteAllLines(Path.Combine(options.OutputDir, $"{options.TaskName}_results.txt"), lines);
    }
}

internal static class OcrPipelineHelper
{
    public static List<OcrItem> ExtractOcrItems(
        Image<Rgb24> original,
        string imagePath,
        InferenceSession? det,
        InferenceSession? rec,
        IReadOnlyList<string> charset,
        float dropScore,
        float detThresh)
    {
        if (rec is null)
        {
            return [];
        }
        var recPost = InferenceComponentRegistry.GetRecPostprocessor();

        var boxes = det is null
            ? [PostprocessUtils.FullImageBox(original.Width, original.Height)]
            : GetBoxes(det, imagePath, original.Width, original.Height, detThresh);
        var sorted = PostprocessUtils.SortBoxes(boxes);
        var result = new List<OcrItem>(sorted.Count);
        foreach (var box in sorted)
        {
            using var crop = OnnxRuntimeUtils.CropBox(original, box);
            var recOut = OnnxRuntimeUtils.RunSession(rec, crop, 48, 320).FirstOrDefault();
            var recRes = recOut is null
                ? new RecResult(string.Empty, 0f)
                : recPost(recOut.Data, recOut.Dims, charset);
            if (!string.IsNullOrWhiteSpace(recRes.Text) && recRes.Score >= dropScore)
            {
                result.Add(new OcrItem(recRes.Text, box.Points, recRes.Score));
            }
        }

        return result;
    }

    private static List<OcrBox> GetBoxes(InferenceSession det, string imagePath, int width, int height, float thresh)
    {
        var detOut = OnnxRuntimeUtils.RunSession(det, imagePath, 640, 640).FirstOrDefault();
        if (detOut is null)
        {
            return [PostprocessUtils.FullImageBox(width, height)];
        }

        var detPost = InferenceComponentRegistry.GetDetPostprocessor();
        var boxes = detPost(detOut.Data, detOut.Dims, width, height, thresh);
        return boxes.Count == 0 ? [PostprocessUtils.FullImageBox(width, height)] : boxes;
    }
}

public static class TableResultSerializer
{
    public static string BuildLine(string imageFile, IReadOnlyList<TensorOutput> tableOutputs, IReadOnlyList<OcrItem> ocr)
    {
        var payload = BuildPayload(tableOutputs, ocr);
        return $"{Path.GetFileName(imageFile)}\t{JsonSerializer.Serialize(payload)}";
    }

    public static object BuildPayload(IReadOnlyList<TensorOutput> tableOutputs, IReadOnlyList<OcrItem> ocr)
    {
        return new
        {
            table_tensors = tableOutputs.Select((x, i) => new { index = i, dims = x.Dims, size = x.Data.Length }).ToList(),
            ocr = ocr.Select(x => new { text = x.Transcription, score = x.Score, points = x.Points }).ToList()
        };
    }
}

public sealed record TensorOutput(float[] Data, int[] Dims);
public sealed record SessionRunProfile(List<TensorOutput> Outputs, double PreprocessMs, double InferenceMs);
internal sealed record RecTraceItem(
    string ImageName,
    double PreprocessMs,
    double InferenceMs,
    double PostprocessMs,
    bool FallbackSingle,
    float Score,
    int TextLength);

public sealed record DetOnnxOptions(
    string ImageDir,
    string DetModelPath,
    string OutputDir,
    string DetAlgorithm,
    float DetThresh,
    float DetBoxThresh,
    float DetUnclipRatio,
    bool UseDilation,
    string BoxType,
    int DetLimitSideLen,
    string DetLimitType,
    string? SaveResPath,
    float DetEastScoreThresh,
    float DetEastCoverThresh,
    float DetEastNmsThresh,
    float DetSastScoreThresh,
    float DetSastNmsThresh,
    float DetPseThresh,
    float DetPseBoxThresh,
    float DetPseMinArea,
    float DetPseScale,
    IReadOnlyList<int> FceScales,
    float FceAlpha,
    float FceBeta,
    int FceFourierDegree,
    string? DetGtLabelPath,
    float DetEvalIouThresh,
    string? DetMetricsPath,
    string DetDbScoreMode = "fast",
    int DetMaxCandidates = 1000,
    bool UseSlice = false,
    float SliceMergeIou = 0.3f,
    int SliceMinBoundDistance = 50);

public sealed record RecOnnxOptions(
    string ImageDir,
    string RecModelPath,
    string OutputDir,
    string? RecCharDictPath,
    bool UseSpaceChar,
    float DropScore,
    RecAlgorithm RecAlgorithm = RecAlgorithm.SVTR_LCNet,
    string RecImageShape = "3,48,320",
    int RecBatchNum = 6,
    int MaxTextLength = 25,
    bool RecImageInverse = false,
    bool ReturnWordBox = false,
    bool RecLogDetail = false);

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

public sealed record SrOnnxOptions(string ImageDir, string SrModelPath, string OutputDir);
public sealed record TableOnnxOptions(
    string ImageDir,
    string TableModelPath,
    string OutputDir,
    string? DetModelPath,
    string? RecModelPath,
    string? RecCharDictPath,
    bool UseSpaceChar,
    float DropScore,
    float DetThresh);
public sealed record KieOnnxOptions(
    string TaskName,
    string ImageDir,
    string KieModelPath,
    string OutputDir,
    string? DetModelPath,
    string? RecModelPath,
    string? RecCharDictPath,
    bool UseSpaceChar,
    float DropScore,
    float DetThresh);

public sealed record OcrItem(string Transcription, int[][] Points, float Score);
public sealed record OcrBox(int[][] Points);
public sealed record ClsResult(string Label, float Score);
// RecResult 已迁移到 PaddleOcr.Models.RecResult

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
    public static SessionRunProfile RunSessionProfiled(
        InferenceSession session,
        string imageFile,
        int defaultH,
        int defaultW,
        string? inputBuilderName = null)
    {
        using var img = Image.Load<Rgb24>(imageFile);
        return RunSessionProfiled(session, img, defaultH, defaultW, inputBuilderName);
    }

    public static SessionRunProfile RunSessionProfiled(
        InferenceSession session,
        Image<Rgb24> image,
        int defaultH,
        int defaultW,
        string? inputBuilderName = null)
    {
        var input = session.InputMetadata.First();
        var builder = string.IsNullOrWhiteSpace(inputBuilderName)
            ? InferencePreprocessRegistry.GetInputBuilder()
            : InferencePreprocessRegistry.GetInputBuilder(inputBuilderName);

        var preprocessWatch = Stopwatch.StartNew();
        var tensor = builder(image, input.Value.Dimensions, defaultH, defaultW);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(input.Key, tensor) };
        preprocessWatch.Stop();

        var inferenceWatch = Stopwatch.StartNew();
        using var outputs = session.Run(inputs);
        var tensors = ConvertTensorOutputs(outputs);
        inferenceWatch.Stop();

        return new SessionRunProfile(
            tensors,
            preprocessWatch.Elapsed.TotalMilliseconds,
            inferenceWatch.Elapsed.TotalMilliseconds);
    }

    public static List<TensorOutput> RunSession(
        InferenceSession session,
        string imageFile,
        int defaultH,
        int defaultW,
        string? inputBuilderName = null)
    {
        return RunSessionProfiled(session, imageFile, defaultH, defaultW, inputBuilderName).Outputs;
    }

    public static List<TensorOutput> RunSession(
        InferenceSession session,
        Image<Rgb24> image,
        int defaultH,
        int defaultW,
        string? inputBuilderName = null)
    {
        return RunSessionProfiled(session, image, defaultH, defaultW, inputBuilderName).Outputs;
    }

    private static List<TensorOutput> ConvertTensorOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
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

    public static List<OcrBox> DetectBoxes(
        float[] data,
        int[] dims,
        int imgW,
        int imgH,
        float thresh,
        float boxThresh = 0.6f,
        float unclipRatio = 1.5f,
        bool useDilation = false,
        string boxType = "quad",
        int maxCandidates = 1000,
        string scoreMode = "fast")
    {
        var (rawMap, h, w) = ExtractMap(data, dims);
        var map = useDilation ? DilateMap(rawMap, h, w, thresh) : rawMap;
        if (map.Length == 0 || h <= 0 || w <= 0)
        {
            return [];
        }

        var visited = new bool[h * w];
        var boxes = new List<OcrBox>();
        var minArea = Math.Max(3, (h * w) / 2000);
        var maxCount = Math.Max(1, maxCandidates);
        var slowScoreMode = scoreMode.Equals("slow", StringComparison.OrdinalIgnoreCase);
        var reachedLimit = false;

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
                var scoreSum = 0f;

                while (queue.Count > 0)
                {
                    var (cx, cy) = queue.Dequeue();
                    count++;
                    scoreSum += map[cy * w + cx];
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

                var mapX1 = Math.Max(0, minX);
                var mapY1 = Math.Max(0, minY);
                var mapX2 = Math.Min(w - 1, maxX);
                var mapY2 = Math.Min(h - 1, maxY);
                var compScore = slowScoreMode
                    ? ComputeRectAverage(rawMap, h, w, mapX1, mapY1, mapX2, mapY2)
                    : scoreSum / Math.Max(1, count);
                if (compScore < boxThresh)
                {
                    continue;
                }

                var x1 = mapX1 * imgW / w;
                var y1 = mapY1 * imgH / h;
                var x2 = Math.Max(x1 + 1, mapX2 * imgW / w);
                var y2 = Math.Max(y1 + 1, mapY2 * imgH / h);

                var expanded = ExpandRect(x1, y1, x2, y2, imgW, imgH, unclipRatio);
                boxes.Add(ToBox(expanded.X1, expanded.Y1, expanded.X2, expanded.Y2, boxType));
                if (boxes.Count >= maxCount)
                {
                    reachedLimit = true;
                    break;
                }
            }

            if (reachedLimit)
            {
                break;
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

    private static float[] DilateMap(float[] map, int h, int w, float thresh)
    {
        var result = (float[])map.Clone();
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var idx = y * w + x;
                if (map[idx] < thresh)
                {
                    continue;
                }

                for (var ny = Math.Max(0, y - 1); ny <= Math.Min(h - 1, y + 1); ny++)
                {
                    for (var nx = Math.Max(0, x - 1); nx <= Math.Min(w - 1, x + 1); nx++)
                    {
                        var nidx = ny * w + nx;
                        if (result[nidx] < map[idx])
                        {
                            result[nidx] = map[idx];
                        }
                    }
                }
            }
        }

        return result;
    }

    private static float ComputeRectAverage(float[] map, int h, int w, int x1, int y1, int x2, int y2)
    {
        var sx = Math.Clamp(Math.Min(x1, x2), 0, Math.Max(0, w - 1));
        var ex = Math.Clamp(Math.Max(x1, x2), 0, Math.Max(0, w - 1));
        var sy = Math.Clamp(Math.Min(y1, y2), 0, Math.Max(0, h - 1));
        var ey = Math.Clamp(Math.Max(y1, y2), 0, Math.Max(0, h - 1));
        var sum = 0f;
        var count = 0;
        for (var y = sy; y <= ey; y++)
        {
            for (var x = sx; x <= ex; x++)
            {
                sum += map[y * w + x];
                count++;
            }
        }

        return count == 0 ? 0f : sum / count;
    }

    private static (int X1, int Y1, int X2, int Y2) ExpandRect(
        int x1,
        int y1,
        int x2,
        int y2,
        int imgW,
        int imgH,
        float unclipRatio)
    {
        var ratio = Math.Max(1f, unclipRatio);
        var cx = (x1 + x2) / 2f;
        var cy = (y1 + y2) / 2f;
        var halfW = (x2 - x1 + 1) * 0.5f * ratio;
        var halfH = (y2 - y1 + 1) * 0.5f * ratio;
        var nx1 = Math.Clamp((int)MathF.Floor(cx - halfW), 0, Math.Max(0, imgW - 1));
        var ny1 = Math.Clamp((int)MathF.Floor(cy - halfH), 0, Math.Max(0, imgH - 1));
        var nx2 = Math.Clamp((int)MathF.Ceiling(cx + halfW), 0, Math.Max(0, imgW - 1));
        var ny2 = Math.Clamp((int)MathF.Ceiling(cy + halfH), 0, Math.Max(0, imgH - 1));
        if (nx2 <= nx1)
        {
            nx2 = Math.Min(imgW - 1, nx1 + 1);
        }

        if (ny2 <= ny1)
        {
            ny2 = Math.Min(imgH - 1, ny1 + 1);
        }

        return (nx1, ny1, nx2, ny2);
    }

    private static OcrBox ToBox(int x1, int y1, int x2, int y2, string boxType)
    {
        var points = boxType.Equals("poly", StringComparison.OrdinalIgnoreCase)
            ? new[]
            {
                new[] { x1, y1 },
                new[] { x2, y1 },
                new[] { x2, y2 },
                new[] { x1, y2 }
            }
            : new[]
            {
                new[] { x1, y1 },
                new[] { x2, y1 },
                new[] { x2, y2 },
                new[] { x1, y2 }
            };
        return new OcrBox(points);
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
