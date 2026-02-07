using System.Text.Json;

namespace PaddleOcr.Inference.Onnx;

public static class DetInferenceExtensions
{
    private static readonly IReadOnlyDictionary<string, IDetPostprocessorStrategy> Strategies =
        new Dictionary<string, IDetPostprocessorStrategy>(StringComparer.OrdinalIgnoreCase)
        {
            ["DB"] = new DbLikePostprocessor(),
            ["DB++"] = new DbLikePostprocessor(),
            ["EAST"] = new EastPostprocessor(),
            ["SAST"] = new SastPostprocessor(),
            ["PSE"] = new PsePostprocessor(),
            ["FCE"] = new FcePostprocessor(),
            ["CT"] = new CtPostprocessor()
        };

    public static string ResolveInputBuilder(string algorithm)
    {
        if (algorithm.Equals("DB++", StringComparison.OrdinalIgnoreCase))
        {
            return "det-dbpp-chw-01";
        }

        if (algorithm.Equals("DB", StringComparison.OrdinalIgnoreCase))
        {
            return "det-db-chw-01";
        }

        return "rgb-chw-01";
    }

    public static (int Width, int Height) ResolveDetInputSize(
        DetOnnxOptions options,
        int imageWidth,
        int imageHeight,
        IReadOnlyList<int> modelInputDims)
    {
        var staticHeight = modelInputDims.Count > 2 ? modelInputDims[2] : -1;
        var staticWidth = modelInputDims.Count > 3 ? modelInputDims[3] : -1;
        if (staticHeight > 0 && staticWidth > 0)
        {
            return (staticWidth, staticHeight);
        }

        var srcW = Math.Max(1, imageWidth);
        var srcH = Math.Max(1, imageHeight);
        var limitSideLen = Math.Max(32, options.DetLimitSideLen);
        var limitType = string.IsNullOrWhiteSpace(options.DetLimitType)
            ? "max"
            : options.DetLimitType.Trim().ToLowerInvariant();

        var ratio = 1f;
        if (limitType == "min")
        {
            var minSide = Math.Min(srcW, srcH);
            if (minSide < limitSideLen)
            {
                ratio = limitSideLen / (float)minSide;
            }
        }
        else
        {
            var maxSide = Math.Max(srcW, srcH);
            if (maxSide > limitSideLen)
            {
                ratio = limitSideLen / (float)maxSide;
            }
        }

        var shrinkOnly = ratio < 1f;
        var resizedW = AlignToStride(srcW * ratio, 32, shrinkOnly);
        var resizedH = AlignToStride(srcH * ratio, 32, shrinkOnly);
        return (resizedW, resizedH);
    }

    public static List<OcrBox> DecodeBoxes(
        DetOnnxOptions options,
        IReadOnlyList<TensorOutput> outputs,
        int imageWidth,
        int imageHeight)
    {
        if (outputs.Count == 0)
        {
            return [];
        }

        var strategy = Strategies.TryGetValue(options.DetAlgorithm, out var found)
            ? found
            : new DbLikePostprocessor();
        var boxes = strategy.Decode(options, outputs, imageWidth, imageHeight);
        return NormalizeBoxes(boxes, imageWidth, imageHeight, options.BoxType);
    }

    public static void WriteDetMetrics(
        DetOnnxOptions options,
        string outputDir,
        IReadOnlyDictionary<string, List<OcrBox>> predictions)
    {
        WriteDetMetrics(options, outputDir, predictions, new Dictionary<string, DetRuntimeProfile>());
    }

    public static void WriteDetMetrics(
        DetOnnxOptions options,
        string outputDir,
        IReadOnlyDictionary<string, List<OcrBox>> predictions,
        IReadOnlyDictionary<string, DetRuntimeProfile> runtimeProfiles)
    {
        var payload = BuildMetricsPayload(options, predictions, runtimeProfiles);
        var metricsPath = string.IsNullOrWhiteSpace(options.DetMetricsPath)
            ? Path.Combine(outputDir, "det_metrics.json")
            : options.DetMetricsPath;
        var metricsDir = Path.GetDirectoryName(Path.GetFullPath(metricsPath));
        if (!string.IsNullOrWhiteSpace(metricsDir))
        {
            Directory.CreateDirectory(metricsDir);
        }

        File.WriteAllText(metricsPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static object BuildMetricsPayload(
        DetOnnxOptions options,
        IReadOnlyDictionary<string, List<OcrBox>> predictions,
        IReadOnlyDictionary<string, DetRuntimeProfile> runtimeProfiles)
    {
        var totalBoxes = predictions.Sum(x => x.Value.Count);
        var avgBoxes = predictions.Count == 0 ? 0f : totalBoxes / (float)predictions.Count;
        var quality = EvaluateQuality(predictions, options.DetGtLabelPath, options.DetEvalIouThresh);
        var qualityPayload = BuildQualityPayload(quality);
        var runtime = BuildRuntimeSummary(runtimeProfiles);
        return new
        {
            schema_version = "1.1",
            algorithm = options.DetAlgorithm,
            image_count = predictions.Count,
            total_pred_boxes = totalBoxes,
            avg_pred_boxes_per_image = avgBoxes,
            thresholds = new
            {
                det_db_thresh = options.DetThresh,
                det_db_box_thresh = options.DetBoxThresh,
                det_db_unclip_ratio = options.DetUnclipRatio,
                det_eval_iou_thresh = options.DetEvalIouThresh
            },
            quality = qualityPayload,
            per_image = BuildPerImageRows(predictions, quality?.Rows, runtimeProfiles),
            algorithm_runtime_profile = runtime,
            runtime_per_image = BuildRuntimePerImageRows(runtimeProfiles),
            generated_at_utc = DateTime.UtcNow
        };
    }

    private static object? BuildQualityPayload(QualitySummary? quality)
    {
        if (quality is null)
        {
            return null;
        }

        return new
        {
            precision = quality.Precision,
            recall = quality.Recall,
            hmean = quality.Hmean,
            true_positive = quality.TruePositive,
            false_positive = quality.FalsePositive,
            false_negative = quality.FalseNegative,
            iou_thresh = quality.IouThreshold
        };
    }

    private static object BuildRuntimeSummary(IReadOnlyDictionary<string, DetRuntimeProfile> runtimeProfiles)
    {
        if (runtimeProfiles.Count == 0)
        {
            return new
            {
                image_count = 0,
                avg_preprocess_ms = 0.0,
                avg_inference_ms = 0.0,
                avg_postprocess_ms = 0.0,
                avg_total_ms = 0.0
            };
        }

        return new
        {
            image_count = runtimeProfiles.Count,
            avg_preprocess_ms = runtimeProfiles.Values.Average(x => x.PreprocessMs),
            avg_inference_ms = runtimeProfiles.Values.Average(x => x.InferenceMs),
            avg_postprocess_ms = runtimeProfiles.Values.Average(x => x.PostprocessMs),
            avg_total_ms = runtimeProfiles.Values.Average(x => x.TotalMs)
        };
    }

    private static IEnumerable<object> BuildPerImageRows(
        IReadOnlyDictionary<string, List<OcrBox>> predictions,
        IReadOnlyDictionary<string, QualityRow>? qualityRows,
        IReadOnlyDictionary<string, DetRuntimeProfile> runtimeProfiles)
    {
        foreach (var item in predictions.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
        {
            var hasRuntime = runtimeProfiles.TryGetValue(item.Key, out var runtime);
            if (qualityRows is not null && qualityRows.TryGetValue(item.Key, out var row))
            {
                if (hasRuntime && runtime is not null)
                {
                    yield return new
                    {
                        image = item.Key,
                        pred_count = item.Value.Count,
                        gt_count = row.GtCount,
                        tp = row.TruePositive,
                        fp = row.FalsePositive,
                        fn = row.FalseNegative,
                        precision = row.Precision,
                        recall = row.Recall,
                        hmean = row.Hmean,
                        preprocess_ms = runtime.PreprocessMs,
                        inference_ms = runtime.InferenceMs,
                        postprocess_ms = runtime.PostprocessMs,
                        total_ms = runtime.TotalMs,
                        original_width = runtime.OriginalWidth,
                        original_height = runtime.OriginalHeight,
                        input_width = runtime.InputWidth,
                        input_height = runtime.InputHeight
                    };
                }
                else
                {
                    yield return new
                    {
                        image = item.Key,
                        pred_count = item.Value.Count,
                        gt_count = row.GtCount,
                        tp = row.TruePositive,
                        fp = row.FalsePositive,
                        fn = row.FalseNegative,
                        precision = row.Precision,
                        recall = row.Recall,
                        hmean = row.Hmean
                    };
                }
            }
            else
            {
                if (hasRuntime && runtime is not null)
                {
                    yield return new
                    {
                        image = item.Key,
                        pred_count = item.Value.Count,
                        preprocess_ms = runtime.PreprocessMs,
                        inference_ms = runtime.InferenceMs,
                        postprocess_ms = runtime.PostprocessMs,
                        total_ms = runtime.TotalMs,
                        original_width = runtime.OriginalWidth,
                        original_height = runtime.OriginalHeight,
                        input_width = runtime.InputWidth,
                        input_height = runtime.InputHeight
                    };
                }
                else
                {
                    yield return new
                    {
                        image = item.Key,
                        pred_count = item.Value.Count
                    };
                }
            }
        }
    }

    private static IEnumerable<object> BuildRuntimePerImageRows(IReadOnlyDictionary<string, DetRuntimeProfile> runtimeProfiles)
    {
        foreach (var item in runtimeProfiles.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
        {
            yield return new
            {
                image = item.Key,
                preprocess_ms = item.Value.PreprocessMs,
                inference_ms = item.Value.InferenceMs,
                postprocess_ms = item.Value.PostprocessMs,
                total_ms = item.Value.TotalMs,
                original_width = item.Value.OriginalWidth,
                original_height = item.Value.OriginalHeight,
                input_width = item.Value.InputWidth,
                input_height = item.Value.InputHeight
            };
        }
    }

    private static QualitySummary? EvaluateQuality(
        IReadOnlyDictionary<string, List<OcrBox>> predictions,
        string? labelFilePath,
        float iouThresh)
    {
        if (string.IsNullOrWhiteSpace(labelFilePath) || !File.Exists(labelFilePath))
        {
            return null;
        }

        var gt = LoadGroundTruth(labelFilePath);
        var imageKeys = new HashSet<string>(predictions.Keys, StringComparer.OrdinalIgnoreCase);
        foreach (var key in gt.Keys)
        {
            imageKeys.Add(key);
        }

        var tp = 0;
        var fp = 0;
        var fn = 0;
        var rows = new Dictionary<string, QualityRow>(StringComparer.OrdinalIgnoreCase);
        foreach (var key in imageKeys)
        {
            var predBoxes = predictions.TryGetValue(key, out var pred) ? pred : [];
            var gtBoxes = gt.TryGetValue(key, out var g) ? g : [];
            var match = MatchByIou(predBoxes, gtBoxes, iouThresh);
            tp += match.TruePositive;
            fp += match.FalsePositive;
            fn += match.FalseNegative;
            rows[key] = new QualityRow(
                GtCount: gtBoxes.Count,
                TruePositive: match.TruePositive,
                FalsePositive: match.FalsePositive,
                FalseNegative: match.FalseNegative,
                Precision: match.Precision,
                Recall: match.Recall,
                Hmean: match.Hmean);
        }

        var precision = tp + fp == 0 ? 0f : tp / (float)(tp + fp);
        var recall = tp + fn == 0 ? 0f : tp / (float)(tp + fn);
        var hmean = precision + recall <= 0f ? 0f : 2f * precision * recall / (precision + recall);
        return new QualitySummary(
            Precision: precision,
            Recall: recall,
            Hmean: hmean,
            TruePositive: tp,
            FalsePositive: fp,
            FalseNegative: fn,
            IouThreshold: iouThresh,
            Rows: rows);
    }

    private static MatchSummary MatchByIou(IReadOnlyList<OcrBox> predBoxes, IReadOnlyList<OcrBox> gtBoxes, float iouThresh)
    {
        var gtMatched = new bool[gtBoxes.Count];
        var tp = 0;
        var fp = 0;
        foreach (var predBox in predBoxes)
        {
            var best = -1;
            var bestIou = 0f;
            for (var i = 0; i < gtBoxes.Count; i++)
            {
                if (gtMatched[i])
                {
                    continue;
                }

                var iou = ComputeIou(predBox, gtBoxes[i]);
                if (iou > bestIou)
                {
                    bestIou = iou;
                    best = i;
                }
            }

            if (best >= 0 && bestIou >= iouThresh)
            {
                gtMatched[best] = true;
                tp++;
            }
            else
            {
                fp++;
            }
        }

        var fn = gtMatched.Count(x => !x);
        var precision = tp + fp == 0 ? 0f : tp / (float)(tp + fp);
        var recall = tp + fn == 0 ? 0f : tp / (float)(tp + fn);
        var hmean = precision + recall <= 0f ? 0f : 2f * precision * recall / (precision + recall);
        return new MatchSummary(tp, fp, fn, precision, recall, hmean);
    }

    private static Dictionary<string, List<OcrBox>> LoadGroundTruth(string labelFilePath)
    {
        var result = new Dictionary<string, List<OcrBox>>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in File.ReadLines(labelFilePath))
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

            var key = Path.GetFileName(line[..tab]);
            if (string.IsNullOrWhiteSpace(key))
            {
                continue;
            }

            var json = line[(tab + 1)..];
            if (!TryParseOcrBoxes(json, out var boxes))
            {
                continue;
            }

            result[key] = boxes;
        }

        return result;
    }

    private static bool TryParseOcrBoxes(string json, out List<OcrBox> boxes)
    {
        boxes = [];
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
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
                    if (p.ValueKind == JsonValueKind.Array &&
                        p.GetArrayLength() >= 2 &&
                        p[0].TryGetInt32(out var x) &&
                        p[1].TryGetInt32(out var y))
                    {
                        poly.Add([x, y]);
                    }
                }

                if (poly.Count >= 4)
                {
                    boxes.Add(new OcrBox(poly.ToArray()));
                }
            }
        }
        catch
        {
            return false;
        }

        return true;
    }

    private static List<OcrBox> NormalizeBoxes(IReadOnlyList<OcrBox> boxes, int width, int height, string boxType)
    {
        var normalized = new List<OcrBox>(boxes.Count);
        foreach (var box in boxes)
        {
            if (box.Points.Length < 4)
            {
                continue;
            }

            var points = box.Points
                .Select(p => new[]
                {
                    Math.Clamp(p[0], 0, Math.Max(0, width - 1)),
                    Math.Clamp(p[1], 0, Math.Max(0, height - 1))
                })
                .ToArray();

            if (boxType.Equals("quad", StringComparison.OrdinalIgnoreCase))
            {
                points = EnsureQuadOrder(points);
            }

            var xs = points.Select(x => x[0]).ToArray();
            var ys = points.Select(x => x[1]).ToArray();
            var w = Math.Max(1, xs.Max() - xs.Min() + 1);
            var h = Math.Max(1, ys.Max() - ys.Min() + 1);
            if (w < 2 || h < 2)
            {
                continue;
            }

            normalized.Add(new OcrBox(points));
        }

        return normalized;
    }

    private static int[][] EnsureQuadOrder(int[][] points)
    {
        if (points.Length < 4)
        {
            return points;
        }

        var rect = new int[4][];
        var sums = points.Select(p => p[0] + p[1]).ToArray();
        var diffs = points.Select(p => p[1] - p[0]).ToArray();
        rect[0] = points[Array.IndexOf(sums, sums.Min())];
        rect[2] = points[Array.IndexOf(sums, sums.Max())];
        rect[1] = points[Array.IndexOf(diffs, diffs.Min())];
        rect[3] = points[Array.IndexOf(diffs, diffs.Max())];
        return rect;
    }

    private static TensorOutput PickPrimary(IReadOnlyList<TensorOutput> outputs)
    {
        return outputs
            .OrderByDescending(x => ResolveMapArea(x.Dims))
            .ThenByDescending(x => x.Data.Length)
            .First();
    }

    private static TensorOutput PickScoreLike(IReadOnlyList<TensorOutput> outputs)
    {
        var scoreLike = outputs
            .Where(x => ResolveChannel(x.Dims) <= 1 && ResolveMapArea(x.Dims) > 0)
            .OrderByDescending(x => ResolveMapArea(x.Dims))
            .ToList();
        if (scoreLike.Count > 0)
        {
            return scoreLike[0];
        }

        return PickPrimary(outputs);
    }

    private static int ResolveChannel(IReadOnlyList<int> dims)
    {
        return dims.Count >= 3 ? Math.Max(1, dims[^3]) : 1;
    }

    private static int ResolveMapArea(IReadOnlyList<int> dims)
    {
        return dims.Count >= 2 ? Math.Max(1, dims[^2]) * Math.Max(1, dims[^1]) : 0;
    }

    private static int AlignToStride(float value, int stride, bool shrinkOnly)
    {
        if (shrinkOnly)
        {
            var aligned = (int)Math.Floor(value / stride) * stride;
            return Math.Max(stride, aligned);
        }

        var rounded = (int)Math.Round(value / stride, MidpointRounding.AwayFromZero) * stride;
        return Math.Max(stride, rounded);
    }

    private static float EstimateBoxShortSide(OcrBox box)
    {
        var xs = box.Points.Select(x => x[0]).ToArray();
        var ys = box.Points.Select(x => x[1]).ToArray();
        var w = Math.Max(1, xs.Max() - xs.Min() + 1);
        var h = Math.Max(1, ys.Max() - ys.Min() + 1);
        return Math.Min(w, h);
    }

    private static List<OcrBox> NonMaxSuppression(IReadOnlyList<OcrBox> boxes, float iouThreshold)
    {
        if (boxes.Count <= 1)
        {
            return boxes.ToList();
        }

        var ordered = boxes
            .OrderByDescending(BoxArea)
            .ToList();
        var kept = new List<OcrBox>(ordered.Count);
        foreach (var box in ordered)
        {
            var suppressed = kept.Any(existing => ComputeIou(existing, box) >= iouThreshold);
            if (!suppressed)
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
        var areaA = Math.Max(1, (ax2 - ax1) * (ay2 - ay1));
        var areaB = Math.Max(1, (bx2 - bx1) * (by2 - by1));
        var union = areaA + areaB - inter;
        return union <= 0 ? 0f : inter / union;
    }

    private static (float X1, float Y1, float X2, float Y2) ToRect(OcrBox box)
    {
        var xs = box.Points.Select(p => (float)p[0]).ToArray();
        var ys = box.Points.Select(p => (float)p[1]).ToArray();
        return (xs.Min(), ys.Min(), xs.Max(), ys.Max());
    }

    private interface IDetPostprocessorStrategy
    {
        List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight);
    }

    private sealed class DbLikePostprocessor : IDetPostprocessorStrategy
    {
        public List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight)
        {
            var map = PickPrimary(outputs);
            var boxes = PostprocessUtils.DetectBoxes(
                map.Data,
                map.Dims,
                imageWidth,
                imageHeight,
                options.DetThresh,
                options.DetBoxThresh,
                options.DetUnclipRatio,
                options.UseDilation,
                options.BoxType);
            return NonMaxSuppression(boxes, 0.6f);
        }
    }

    private sealed class EastPostprocessor : IDetPostprocessorStrategy
    {
        public List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight)
        {
            var score = PickScoreLike(outputs);
            var boxes = PostprocessUtils.DetectBoxes(
                score.Data,
                score.Dims,
                imageWidth,
                imageHeight,
                options.DetEastScoreThresh,
                options.DetEastCoverThresh,
                1.0f,
                false,
                options.BoxType);
            return NonMaxSuppression(boxes, options.DetEastNmsThresh);
        }
    }

    private sealed class SastPostprocessor : IDetPostprocessorStrategy
    {
        public List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight)
        {
            var score = PickScoreLike(outputs);
            var boxes = PostprocessUtils.DetectBoxes(
                score.Data,
                score.Dims,
                imageWidth,
                imageHeight,
                options.DetSastScoreThresh,
                options.DetSastScoreThresh,
                1.0f,
                false,
                options.BoxType);
            return NonMaxSuppression(boxes, options.DetSastNmsThresh);
        }
    }

    private sealed class PsePostprocessor : IDetPostprocessorStrategy
    {
        public List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight)
        {
            var map = PickPrimary(outputs);
            var boxes = PostprocessUtils.DetectBoxes(
                map.Data,
                map.Dims,
                imageWidth,
                imageHeight,
                options.DetPseThresh,
                options.DetPseBoxThresh,
                Math.Max(1.0f, options.DetPseScale),
                options.UseDilation,
                options.BoxType);
            var minSide = Math.Max(1f, MathF.Sqrt(Math.Max(1f, options.DetPseMinArea)));
            return boxes
                .Where(b => EstimateBoxShortSide(b) >= minSide)
                .ToList();
        }
    }

    private sealed class FcePostprocessor : IDetPostprocessorStrategy
    {
        public List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight)
        {
            var boxes = new List<OcrBox>();
            var baseThresh = Math.Clamp(options.DetThresh * options.FceAlpha / Math.Max(0.1f, options.FceBeta), 0.01f, 0.99f);
            var levelCount = Math.Min(options.FceScales.Count, outputs.Count);
            for (var i = 0; i < levelCount; i++)
            {
                var level = outputs[i];
                var levelBoxes = PostprocessUtils.DetectBoxes(
                    level.Data,
                    level.Dims,
                    imageWidth,
                    imageHeight,
                    baseThresh,
                    options.DetBoxThresh,
                    options.DetUnclipRatio,
                    options.UseDilation,
                    options.BoxType);
                boxes.AddRange(levelBoxes);
            }

            return NonMaxSuppression(boxes, 0.8f);
        }
    }

    private sealed class CtPostprocessor : IDetPostprocessorStrategy
    {
        public List<OcrBox> Decode(DetOnnxOptions options, IReadOnlyList<TensorOutput> outputs, int imageWidth, int imageHeight)
        {
            var score = PickScoreLike(outputs);
            return PostprocessUtils.DetectBoxes(
                score.Data,
                score.Dims,
                imageWidth,
                imageHeight,
                options.DetThresh,
                options.DetBoxThresh,
                1.0f,
                false,
                options.BoxType);
        }
    }
}

public sealed record DetRuntimeProfile(
    double PreprocessMs,
    double InferenceMs,
    double PostprocessMs,
    double TotalMs,
    int OriginalWidth = 0,
    int OriginalHeight = 0,
    int InputWidth = 0,
    int InputHeight = 0);

internal sealed record QualitySummary(
    float Precision,
    float Recall,
    float Hmean,
    int TruePositive,
    int FalsePositive,
    int FalseNegative,
    float IouThreshold,
    IReadOnlyDictionary<string, QualityRow> Rows);

internal sealed record QualityRow(
    int GtCount,
    int TruePositive,
    int FalsePositive,
    int FalseNegative,
    float Precision,
    float Recall,
    float Hmean);

internal sealed record MatchSummary(
    int TruePositive,
    int FalsePositive,
    int FalseNegative,
    float Precision,
    float Recall,
    float Hmean);
