namespace PaddleOcr.Training;

public static class DetMetricEvaluator
{
    public static DetMatchSummary EvaluateSingle(
        bool[] predMask,
        bool[] gtMask,
        int width,
        int height,
        float iouThreshold,
        int minComponentArea = 3)
    {
        var predBoxes = ExtractBoxes(predMask, width, height, minComponentArea);
        var gtBoxes = ExtractBoxes(gtMask, width, height, minComponentArea);
        return MatchBoxes(predBoxes, gtBoxes, iouThreshold);
    }

    private static List<DetRect> ExtractBoxes(bool[] mask, int width, int height, int minComponentArea)
    {
        var boxes = new List<DetRect>();
        if (mask.Length != width * height)
        {
            return boxes;
        }

        var visited = new bool[mask.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var idx = y * width + x;
                if (visited[idx] || !mask[idx])
                {
                    continue;
                }

                var queue = new Queue<(int X, int Y)>();
                queue.Enqueue((x, y));
                visited[idx] = true;
                var count = 0;
                var minX = x;
                var minY = y;
                var maxX = x;
                var maxY = y;
                while (queue.Count > 0)
                {
                    var (cx, cy) = queue.Dequeue();
                    count++;
                    minX = Math.Min(minX, cx);
                    minY = Math.Min(minY, cy);
                    maxX = Math.Max(maxX, cx);
                    maxY = Math.Max(maxY, cy);

                    foreach (var (nx, ny) in Neighbors(cx, cy, width, height))
                    {
                        var nidx = ny * width + nx;
                        if (visited[nidx] || !mask[nidx])
                        {
                            continue;
                        }

                        visited[nidx] = true;
                        queue.Enqueue((nx, ny));
                    }
                }

                if (count >= minComponentArea)
                {
                    boxes.Add(new DetRect(minX, minY, maxX, maxY));
                }
            }
        }

        return boxes;
    }

    private static DetMatchSummary MatchBoxes(IReadOnlyList<DetRect> predBoxes, IReadOnlyList<DetRect> gtBoxes, float iouThreshold)
    {
        var threshold = Math.Clamp(iouThreshold, 0f, 1f);
        var gtMatched = new bool[gtBoxes.Count];
        var tp = 0;
        var fp = 0;
        var fn = 0;
        var iouSum = 0f;

        foreach (var pred in predBoxes)
        {
            var best = -1;
            var bestIou = 0f;
            for (var i = 0; i < gtBoxes.Count; i++)
            {
                if (gtMatched[i])
                {
                    continue;
                }

                var iou = IoU(pred, gtBoxes[i]);
                if (iou > bestIou)
                {
                    bestIou = iou;
                    best = i;
                }
            }

            if (best >= 0 && bestIou >= threshold)
            {
                gtMatched[best] = true;
                tp++;
                iouSum += bestIou;
            }
            else
            {
                fp++;
            }
        }

        for (var i = 0; i < gtBoxes.Count; i++)
        {
            if (!gtMatched[i])
            {
                fn++;
            }
        }

        return new DetMatchSummary(tp, fp, fn, iouSum);
    }

    private static float IoU(DetRect a, DetRect b)
    {
        var ix1 = Math.Max(a.X1, b.X1);
        var iy1 = Math.Max(a.Y1, b.Y1);
        var ix2 = Math.Min(a.X2, b.X2);
        var iy2 = Math.Min(a.Y2, b.Y2);
        if (ix2 < ix1 || iy2 < iy1)
        {
            return 0f;
        }

        var inter = (ix2 - ix1 + 1f) * (iy2 - iy1 + 1f);
        var areaA = a.Area;
        var areaB = b.Area;
        var union = areaA + areaB - inter;
        return union <= 0f ? 0f : inter / union;
    }

    private static IEnumerable<(int X, int Y)> Neighbors(int x, int y, int width, int height)
    {
        if (x > 0) yield return (x - 1, y);
        if (x + 1 < width) yield return (x + 1, y);
        if (y > 0) yield return (x, y - 1);
        if (y + 1 < height) yield return (x, y + 1);
    }
}

public readonly record struct DetRect(int X1, int Y1, int X2, int Y2)
{
    public float Area => Math.Max(1, X2 - X1 + 1) * Math.Max(1, Y2 - Y1 + 1);
}

public readonly record struct DetMatchSummary(int TruePositive, int FalsePositive, int FalseNegative, float MatchedIouSum)
{
    public float Precision => TruePositive + FalsePositive == 0 ? 0f : TruePositive / (float)(TruePositive + FalsePositive);
    public float Recall => TruePositive + FalseNegative == 0 ? 0f : TruePositive / (float)(TruePositive + FalseNegative);
    public float Fscore => Precision + Recall <= 0f ? 0f : 2f * Precision * Recall / (Precision + Recall);
    public float MeanIou => TruePositive == 0 ? 0f : MatchedIouSum / TruePositive;

    public static DetMatchSummary operator +(DetMatchSummary left, DetMatchSummary right)
    {
        return new DetMatchSummary(
            left.TruePositive + right.TruePositive,
            left.FalsePositive + right.FalsePositive,
            left.FalseNegative + right.FalseNegative,
            left.MatchedIouSum + right.MatchedIouSum);
    }
}
