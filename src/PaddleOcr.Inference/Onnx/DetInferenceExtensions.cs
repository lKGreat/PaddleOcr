using System.Text.Json;

namespace PaddleOcr.Inference.Onnx;

public static class DetInferenceExtensions
{
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

        if (options.DetAlgorithm.Equals("DB", StringComparison.OrdinalIgnoreCase) ||
            options.DetAlgorithm.Equals("DB++", StringComparison.OrdinalIgnoreCase))
        {
            var map = PickPrimary(outputs);
            return PostprocessUtils.DetectBoxes(
                map.Data,
                map.Dims,
                imageWidth,
                imageHeight,
                options.DetThresh,
                options.DetBoxThresh,
                options.DetUnclipRatio,
                options.UseDilation,
                options.BoxType);
        }

        if (options.DetAlgorithm.Equals("EAST", StringComparison.OrdinalIgnoreCase))
        {
            var score = PickScoreLike(outputs);
            return PostprocessUtils.DetectBoxes(
                score.Data,
                score.Dims,
                imageWidth,
                imageHeight,
                options.DetEastScoreThresh,
                options.DetEastCoverThresh,
                1.0f,
                false,
                options.BoxType);
        }

        if (options.DetAlgorithm.Equals("SAST", StringComparison.OrdinalIgnoreCase))
        {
            var score = PickScoreLike(outputs);
            return PostprocessUtils.DetectBoxes(
                score.Data,
                score.Dims,
                imageWidth,
                imageHeight,
                options.DetSastScoreThresh,
                options.DetSastNmsThresh,
                1.0f,
                false,
                options.BoxType);
        }

        if (options.DetAlgorithm.Equals("PSE", StringComparison.OrdinalIgnoreCase))
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
            return boxes.Where(b => EstimateBoxShortSide(b) >= minSide).ToList();
        }

        if (options.DetAlgorithm.Equals("FCE", StringComparison.OrdinalIgnoreCase))
        {
            var boxes = new List<OcrBox>();
            var baseThresh = Math.Clamp(options.DetThresh * options.FceAlpha / Math.Max(0.1f, options.FceBeta), 0.01f, 0.99f);
            foreach (var level in outputs)
            {
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

            return DeduplicateByIou(boxes, 0.8f);
        }

        if (options.DetAlgorithm.Equals("CT", StringComparison.OrdinalIgnoreCase))
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

        var fallback = PickPrimary(outputs);
        return PostprocessUtils.DetectBoxes(
            fallback.Data,
            fallback.Dims,
            imageWidth,
            imageHeight,
            options.DetThresh,
            options.DetBoxThresh,
            options.DetUnclipRatio,
            options.UseDilation,
            options.BoxType);
    }

    public static void WriteDetMetrics(
        DetOnnxOptions options,
        string outputDir,
        IReadOnlyDictionary<string, List<OcrBox>> predictions)
    {
        var payload = BuildMetricsPayload(options, predictions);
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

    private static object BuildMetricsPayload(DetOnnxOptions options, IReadOnlyDictionary<string, List<OcrBox>> predictions)
    {
        var totalBoxes = predictions.Sum(x => x.Value.Count);
        var avgBoxes = predictions.Count == 0 ? 0f : totalBoxes / (float)predictions.Count;
        var quality = EvaluateQuality(predictions, options.DetGtLabelPath, options.DetEvalIouThresh);
        return new
        {
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
            quality,
            generated_at_utc = DateTime.UtcNow
        };
    }

    private static object? EvaluateQuality(
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
        foreach (var key in imageKeys)
        {
            var predBoxes = predictions.TryGetValue(key, out var pred) ? pred : [];
            var gtBoxes = gt.TryGetValue(key, out var g) ? g : [];

            var gtMatched = new bool[gtBoxes.Count];
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

            fn += gtMatched.Count(x => !x);
        }

        var precision = tp + fp == 0 ? 0f : tp / (float)(tp + fp);
        var recall = tp + fn == 0 ? 0f : tp / (float)(tp + fn);
        var hmean = precision + recall <= 0f ? 0f : 2f * precision * recall / (precision + recall);
        return new
        {
            precision,
            recall,
            hmean,
            true_positive = tp,
            false_positive = fp,
            false_negative = fn,
            iou_thresh = iouThresh
        };
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

    private static float EstimateBoxShortSide(OcrBox box)
    {
        var xs = box.Points.Select(x => x[0]).ToArray();
        var ys = box.Points.Select(x => x[1]).ToArray();
        var w = Math.Max(1, xs.Max() - xs.Min() + 1);
        var h = Math.Max(1, ys.Max() - ys.Min() + 1);
        return Math.Min(w, h);
    }

    private static List<OcrBox> DeduplicateByIou(List<OcrBox> boxes, float iouThreshold)
    {
        if (boxes.Count <= 1)
        {
            return boxes;
        }

        var result = new List<OcrBox>(boxes.Count);
        foreach (var box in boxes)
        {
            var duplicate = result.Any(existing => ComputeIou(existing, box) >= iouThreshold);
            if (!duplicate)
            {
                result.Add(box);
            }
        }

        return result;
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
}
