using System.Globalization;
using System.Drawing;
using System.Text;
using System.Text.Json;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.Data;

public sealed class E2eToolsExecutor : ICommandExecutor
{
    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (subCommand.Equals("convert-label", StringComparison.OrdinalIgnoreCase))
        {
            if (!context.Options.TryGetValue("--label_path", out var labelPath) ||
                !context.Options.TryGetValue("--save_folder", out var saveFolder))
            {
                return Task.FromResult(CommandResult.Fail("e2e convert-label requires --label_path and --save_folder"));
            }

            var mode = context.Options.TryGetValue("--mode", out var m) ? m : "gt";
            var count = ConvertLabels(labelPath, saveFolder, mode);
            return Task.FromResult(CommandResult.Ok($"e2e convert-label completed: {count} files in {saveFolder}"));
        }

        if (subCommand.Equals("eval", StringComparison.OrdinalIgnoreCase))
        {
            var (gtDir, predDir) = ResolveEvalDirs(context);
            if (string.IsNullOrWhiteSpace(gtDir) || string.IsNullOrWhiteSpace(predDir))
            {
                return Task.FromResult(CommandResult.Fail("e2e eval requires <gt_dir> <pred_dir>"));
            }

            var summary = Evaluate(gtDir, predDir);
            var message =
                $"e2e eval completed: p={summary.Precision:F4}, r={summary.Recall:F4}, f={summary.Fmeasure:F4}, " +
                $"char_acc={summary.CharacterAccuracy:F4}, gt={summary.GtCount}, dt={summary.DtCount}";
            return Task.FromResult(CommandResult.Ok(message));
        }

        return Task.FromResult(CommandResult.Fail($"Unsupported e2e subcommand: {subCommand}"));
    }

    private static (string? GtDir, string? PredDir) ResolveEvalDirs(PaddleOcr.Core.Cli.ExecutionContext context)
    {
        context.Options.TryGetValue("--gt_dir", out var gtDir);
        context.Options.TryGetValue("--pred_dir", out var predDir);
        if (!string.IsNullOrWhiteSpace(gtDir) && !string.IsNullOrWhiteSpace(predDir))
        {
            return (gtDir, predDir);
        }

        var positionals = context.RawArgs.Skip(2).Where(x => !x.StartsWith('-')).ToArray();
        if (positionals.Length >= 2)
        {
            return (positionals[0], positionals[1]);
        }

        return (null, null);
    }

    private static int ConvertLabels(string labelPath, string saveFolder, string mode)
    {
        if (!File.Exists(labelPath))
        {
            throw new FileNotFoundException($"The file {labelPath} does not exist.");
        }

        Directory.CreateDirectory(saveFolder);
        var dict = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in File.ReadLines(labelPath))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var split = line.Split('\t', 2);
            if (split.Length != 2)
            {
                split = line.Split("    ", 2, StringSplitOptions.None);
                if (split.Length != 2)
                {
                    continue;
                }
            }

            var imagePath = split[0].Trim();
            if (string.IsNullOrWhiteSpace(imagePath))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(split[1]);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            var rows = new List<string>();
            foreach (var ann in doc.RootElement.EnumerateArray())
            {
                if (ann.TryGetProperty("score", out var scoreEl) &&
                    scoreEl.TryGetDouble(out var score) &&
                    score < 0.5)
                {
                    continue;
                }

                var text = ann.TryGetProperty("transcription", out var txtEl) ? txtEl.GetString() ?? string.Empty : string.Empty;
                text = text.Replace('\u3000', ' ');
                var txtTag = text == "###" ? 1 : 0;
                if (!ann.TryGetProperty("points", out var pointsEl) || pointsEl.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                var flattened = FlattenPoints(pointsEl);
                var poly = string.Join('\t', flattened.Select(v => v.ToString(CultureInfo.InvariantCulture)));
                var row = mode.Equals("gt", StringComparison.OrdinalIgnoreCase)
                    ? $"{poly}\t{txtTag}\t{text}"
                    : $"{poly}\t{text}";
                rows.Add(row);
            }

            dict[imagePath] = rows;
        }

        foreach (var (img, rows) in dict)
        {
            var fileName = Path.GetFileName(img) + ".txt";
            var path = Path.Combine(saveFolder, fileName);
            File.WriteAllLines(path, rows, new UTF8Encoding(false));
        }

        return dict.Count;
    }

    private static IReadOnlyList<float> FlattenPoints(JsonElement pointsEl)
    {
        var values = new List<float>(8);
        foreach (var point in pointsEl.EnumerateArray())
        {
            if (point.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            foreach (var v in point.EnumerateArray())
            {
                if (v.TryGetSingle(out var f))
                {
                    values.Add(f);
                }
            }
        }

        return values;
    }

    private static E2eSummary Evaluate(string gtDir, string predDir)
    {
        if (!Directory.Exists(gtDir))
        {
            throw new DirectoryNotFoundException($"gt_dir not found: {gtDir}");
        }

        if (!Directory.Exists(predDir))
        {
            throw new DirectoryNotFoundException($"pred_dir not found: {predDir}");
        }

        var gtFiles = Directory.EnumerateFiles(gtDir, "*.txt", SearchOption.TopDirectoryOnly).Select(Path.GetFileName).Where(x => x is not null).Cast<string>().ToList();
        var numGtChars = 0;
        var gtCount = 0;
        var dtCount = 0;
        var hit = 0;
        var edSum = 0;
        const float iouThresh = 0.5f;
        const float eps = 1e-9f;

        foreach (var name in gtFiles)
        {
            var gts = LoadGt(Path.Combine(gtDir, name), out var ignoreMasks);
            var dts = LoadDt(Path.Combine(predDir, name));
            var dtMatch = new bool[dts.Count];
            var gtMatch = new bool[gts.Count];
            var pairIous = new List<(int Gt, int Dt, float Iou)>();

            for (var i = 0; i < gts.Count; i++)
            {
                for (var j = 0; j < dts.Count; j++)
                {
                    var iou = PolygonIou(gts[i].Coords, dts[j].Coords);
                    if (iou >= iouThresh)
                    {
                        pairIous.Add((i, j, iou));
                    }
                }
            }

            foreach (var pair in pairIous.OrderByDescending(x => x.Iou))
            {
                if (gtMatch[pair.Gt] || dtMatch[pair.Dt])
                {
                    continue;
                }

                gtMatch[pair.Gt] = true;
                dtMatch[pair.Dt] = true;
                if (ignoreMasks[pair.Gt] != "0")
                {
                    continue;
                }

                var gtText = ToHalfWidth(gts[pair.Gt].Text);
                var dtText = ToHalfWidth(dts[pair.Dt].Text);
                edSum += Levenshtein(gtText, dtText);
                numGtChars += gtText.Length;
                if (string.Equals(gtText, dtText, StringComparison.Ordinal))
                {
                    hit++;
                }

                gtCount++;
                dtCount++;
            }

            for (var i = 0; i < dtMatch.Length; i++)
            {
                if (dtMatch[i])
                {
                    continue;
                }

                edSum += Levenshtein(dts[i].Text, string.Empty);
                dtCount++;
            }

            for (var i = 0; i < gtMatch.Length; i++)
            {
                if (gtMatch[i] || ignoreMasks[i] != "0")
                {
                    continue;
                }

                edSum += Levenshtein(gts[i].Text, string.Empty);
                numGtChars += gts[i].Text.Length;
                gtCount++;
            }
        }

        var precision = hit / (dtCount + eps);
        var recall = hit / (gtCount + eps);
        var fmeasure = 2f * precision * recall / (precision + recall + eps);
        var avgEditDistImg = gtFiles.Count == 0 ? 0 : (float)edSum / gtFiles.Count;
        var avgEditDistField = edSum / (gtCount + eps);
        var characterAcc = 1f - edSum / (numGtChars + eps);
        return new E2eSummary(precision, recall, fmeasure, characterAcc, avgEditDistField, avgEditDistImg, gtCount, dtCount);
    }

    private static List<E2eLine> LoadGt(string path, out List<string> ignoreMasks)
    {
        var lines = new List<E2eLine>();
        ignoreMasks = [];
        if (!File.Exists(path))
        {
            return lines;
        }

        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split('\t');
            if (parts.Length < 9)
            {
                continue;
            }

            if (!TryParseCoords(parts, 8, out var coords))
            {
                continue;
            }

            var text = parts.Length == 9 ? string.Empty : parts[^1];
            lines.Add(new E2eLine(coords, text));
            ignoreMasks.Add(parts[8]);
        }

        return lines;
    }

    private static List<E2eLine> LoadDt(string path)
    {
        var lines = new List<E2eLine>();
        if (!File.Exists(path))
        {
            return lines;
        }

        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split('\t');
            if (parts.Length < 8)
            {
                continue;
            }

            if (!TryParseCoords(parts, 8, out var coords))
            {
                continue;
            }

            var text = parts.Length == 8 ? string.Empty : parts[^1];
            lines.Add(new E2eLine(coords, text));
        }

        return lines;
    }

    private static bool TryParseCoords(string[] parts, int count, out float[] coords)
    {
        coords = new float[count];
        for (var i = 0; i < count; i++)
        {
            if (!float.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
            {
                return false;
            }

            coords[i] = v;
        }

        return true;
    }

    private static string ToHalfWidth(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var ch in text)
        {
            var code = ch;
            if (code == 12288)
            {
                code = (char)32;
            }
            else if (code is >= (char)65281 and <= (char)65374)
            {
                code = (char)(code - 65248);
            }

            sb.Append(code);
        }

        return sb.ToString();
    }

    private static int Levenshtein(string left, string right)
    {
        if (left.Length == 0)
        {
            return right.Length;
        }

        if (right.Length == 0)
        {
            return left.Length;
        }

        var prev = new int[right.Length + 1];
        var curr = new int[right.Length + 1];
        for (var j = 0; j <= right.Length; j++)
        {
            prev[j] = j;
        }

        for (var i = 1; i <= left.Length; i++)
        {
            curr[0] = i;
            for (var j = 1; j <= right.Length; j++)
            {
                var cost = left[i - 1] == right[j - 1] ? 0 : 1;
                curr[j] = Math.Min(
                    Math.Min(curr[j - 1] + 1, prev[j] + 1),
                    prev[j - 1] + cost);
            }

            (prev, curr) = (curr, prev);
        }

        return prev[right.Length];
    }

    private static float PolygonIou(float[] a, float[] b)
    {
        var pa = ToPoints(a);
        var pb = ToPoints(b);
        var inter = IntersectConvex(pa, pb);
        var interArea = PolygonArea(inter);
        var areaA = PolygonArea(pa);
        var areaB = PolygonArea(pb);
        var union = areaA + areaB - interArea;
        return union <= 1e-6f ? 0f : interArea / union;
    }

    private static List<PointF> ToPoints(float[] coords)
    {
        var points = new List<PointF>(coords.Length / 2);
        for (var i = 0; i + 1 < coords.Length; i += 2)
        {
            points.Add(new PointF(coords[i], coords[i + 1]));
        }

        return points;
    }

    private static List<PointF> IntersectConvex(List<PointF> subject, List<PointF> clip)
    {
        var output = new List<PointF>(subject);
        for (var i = 0; i < clip.Count; i++)
        {
            var cp1 = clip[i];
            var cp2 = clip[(i + 1) % clip.Count];
            var input = output;
            output = [];
            if (input.Count == 0)
            {
                break;
            }

            var s = input[^1];
            foreach (var e in input)
            {
                if (Inside(e, cp1, cp2))
                {
                    if (!Inside(s, cp1, cp2))
                    {
                        output.Add(Intersection(s, e, cp1, cp2));
                    }

                    output.Add(e);
                }
                else if (Inside(s, cp1, cp2))
                {
                    output.Add(Intersection(s, e, cp1, cp2));
                }

                s = e;
            }
        }

        return output;
    }

    private static bool Inside(PointF p, PointF a, PointF b)
    {
        return ((b.X - a.X) * (p.Y - a.Y) - (b.Y - a.Y) * (p.X - a.X)) >= 0f;
    }

    private static PointF Intersection(PointF s, PointF e, PointF a, PointF b)
    {
        var dc = new PointF(a.X - b.X, a.Y - b.Y);
        var dp = new PointF(s.X - e.X, s.Y - e.Y);
        var n1 = a.X * b.Y - a.Y * b.X;
        var n2 = s.X * e.Y - s.Y * e.X;
        var n3 = dc.X * dp.Y - dc.Y * dp.X;
        if (Math.Abs(n3) < 1e-6f)
        {
            return e;
        }

        return new PointF((n1 * dp.X - n2 * dc.X) / n3, (n1 * dp.Y - n2 * dc.Y) / n3);
    }

    private static float PolygonArea(List<PointF> pts)
    {
        if (pts.Count < 3)
        {
            return 0f;
        }

        var sum = 0f;
        for (var i = 0; i < pts.Count; i++)
        {
            var j = (i + 1) % pts.Count;
            sum += pts[i].X * pts[j].Y - pts[j].X * pts[i].Y;
        }

        return Math.Abs(sum) * 0.5f;
    }

    private sealed record E2eLine(float[] Coords, string Text);
    private sealed record E2eSummary(float Precision, float Recall, float Fmeasure, float CharacterAccuracy, float AvgEditDistField, float AvgEditDistImg, int GtCount, int DtCount);
}
