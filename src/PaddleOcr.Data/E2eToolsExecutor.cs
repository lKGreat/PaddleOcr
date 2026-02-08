using System.Globalization;
using System.Drawing;
using System.Text;
using System.Text.Json;
using PaddleOcr.Core.Cli;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using DrawingPointF = System.Drawing.PointF;

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

            var thresholds = ParseIouThresholds(context.Options.TryGetValue("--iou_threshes", out var t) ? t : "0.5");
            var result = Evaluate(gtDir, predDir, thresholds);
            if (context.Options.TryGetValue("--detail_json", out var detailJson) && !string.IsNullOrWhiteSpace(detailJson))
            {
                var dir = Path.GetDirectoryName(detailJson);
                if (!string.IsNullOrWhiteSpace(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                File.WriteAllText(
                    detailJson,
                    JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
            }

            var summary = result.Summaries.First();
            var message =
                $"e2e eval completed: p={summary.Precision:F4}, r={summary.Recall:F4}, f={summary.Fmeasure:F4}, " +
                $"char_acc={summary.CharacterAccuracy:F4}, gt={summary.GtCount}, dt={summary.DtCount}, iou_threshes={string.Join(',', thresholds.Select(x => x.ToString("0.##", CultureInfo.InvariantCulture)))}";
            return Task.FromResult(CommandResult.Ok(message));
        }

        if (subCommand.Equals("prepare-rec-det", StringComparison.OrdinalIgnoreCase))
        {
            if (!context.Options.TryGetValue("--label_path", out var labelPath) ||
                !context.Options.TryGetValue("--image_root", out var imageRoot) ||
                !context.Options.TryGetValue("--output_dir", out var outputDir))
            {
                return Task.FromResult(CommandResult.Fail("e2e prepare-rec-det requires --label_path --image_root --output_dir"));
            }

            var trainRatio = ParseFloat(context.Options, "--train_ratio", 0.9f);
            trainRatio = Math.Clamp(trainRatio, 0.5f, 0.99f);
            var seed = ParseInt(context.Options, "--seed", 42);
            var maxSamples = ParseInt(context.Options, "--max_samples", 0);
            var valLabelPath = context.Options.TryGetValue("--val_label_path", out var vl) ? vl : null;
            var prepared = PrepareRecDatasetFromDet(labelPath, valLabelPath, imageRoot, outputDir, trainRatio, seed, maxSamples);
            var msg =
                $"e2e prepare-rec-det completed: total={prepared.TotalSamples}, train={prepared.TrainSamples}, val={prepared.ValSamples}, " +
                $"skipped={prepared.SkippedAnnotations}, output={prepared.OutputDir}";
            return Task.FromResult(CommandResult.Ok(msg));
        }

        return Task.FromResult(CommandResult.Fail($"Unsupported e2e subcommand: {subCommand}"));
    }

    private static RecPreparedSummary PrepareRecDatasetFromDet(
        string labelPath,
        string? valLabelPath,
        string imageRoot,
        string outputDir,
        float trainRatio,
        int seed,
        int maxSamples)
    {
        if (!File.Exists(labelPath))
        {
            throw new FileNotFoundException($"label file not found: {labelPath}");
        }

        if (!Directory.Exists(imageRoot))
        {
            throw new DirectoryNotFoundException($"image_root not found: {imageRoot}");
        }

        var imagesDir = Path.Combine(outputDir, "images");
        Directory.CreateDirectory(outputDir);
        Directory.CreateDirectory(imagesDir);

        var saved = 0;
        var skipped = 0;
        var trainEntries = CollectRecEntries(labelPath, imageRoot, imagesDir, maxSamples, ref saved, ref skipped);
        string[] train;
        string[] val;

        if (!string.IsNullOrWhiteSpace(valLabelPath))
        {
            if (!File.Exists(valLabelPath))
            {
                throw new FileNotFoundException($"val label file not found: {valLabelPath}");
            }

            var valEntries = CollectRecEntries(valLabelPath, imageRoot, imagesDir, maxSamples: 0, ref saved, ref skipped);
            if (trainEntries.Count == 0 || valEntries.Count == 0)
            {
                throw new InvalidOperationException("No rec samples generated for train/val from detection labels.");
            }

            train = trainEntries.ToArray();
            val = valEntries.ToArray();
        }
        else
        {
            if (trainEntries.Count == 0)
            {
                throw new InvalidOperationException("No rec samples generated from detection labels.");
            }

            var entries = trainEntries;
            var rng = new Random(seed);
            for (var i = entries.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (entries[i], entries[j]) = (entries[j], entries[i]);
            }

            var trainCount = entries.Count == 1
                ? 1
                : Math.Clamp((int)Math.Round(entries.Count * trainRatio), 1, entries.Count - 1);
            train = entries.Take(trainCount).ToArray();
            val = entries.Skip(trainCount).ToArray();
        }

        File.WriteAllLines(Path.Combine(outputDir, "train_list.txt"), train, new UTF8Encoding(false));
        File.WriteAllLines(Path.Combine(outputDir, "val_list.txt"), val, new UTF8Encoding(false));

        var summary = new RecPreparedSummary(outputDir, train.Length + val.Length, train.Length, val.Length, skipped);
        var summaryJson = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(outputDir, "prepare_rec_summary.json"), summaryJson);
        return summary;
    }

    private static List<string> CollectRecEntries(
        string labelPath,
        string imageRoot,
        string imagesDir,
        int maxSamples,
        ref int saved,
        ref int skipped)
    {
        var entries = new List<string>();
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

            var rawImagePath = split[0].Trim();
            var fullImagePath = Path.IsPathRooted(rawImagePath)
                ? rawImagePath
                : Path.GetFullPath(Path.Combine(imageRoot, rawImagePath));
            if (!File.Exists(fullImagePath))
            {
                skipped++;
                continue;
            }

            using var image = Image.Load<Rgb24>(fullImagePath);
            using var doc = JsonDocument.Parse(split[1]);
            if (doc.RootElement.ValueKind != JsonValueKind.Array)
            {
                skipped++;
                continue;
            }

            foreach (var ann in doc.RootElement.EnumerateArray())
            {
                if (!ann.TryGetProperty("transcription", out var txtEl))
                {
                    skipped++;
                    continue;
                }

                var text = SanitizeRecText(txtEl.GetString() ?? string.Empty);
                if (string.IsNullOrWhiteSpace(text) || text == "###")
                {
                    skipped++;
                    continue;
                }

                if (!ann.TryGetProperty("points", out var pointsEl) || pointsEl.ValueKind != JsonValueKind.Array)
                {
                    skipped++;
                    continue;
                }

                if (!TryCreateRecCrop(image, pointsEl, out var crop))
                {
                    skipped++;
                    continue;
                }

                var cropName = $"{Path.GetFileNameWithoutExtension(fullImagePath)}_{saved:D7}.jpg";
                var cropPath = Path.Combine(imagesDir, cropName);
                crop.Save(cropPath, new JpegEncoder { Quality = 95 });
                entries.Add($"images/{cropName}\t{text}");
                saved++;

                if (maxSamples > 0 && saved >= maxSamples)
                {
                    break;
                }
            }

            if (maxSamples > 0 && saved >= maxSamples)
            {
                break;
            }
        }

        return entries;
    }

    private static bool TryCreateRecCrop(Image<Rgb24> image, JsonElement pointsEl, out Image<Rgb24> crop)
    {
        crop = default!;
        if (!TryParsePoints(pointsEl, out var points))
        {
            return false;
        }

        if (points.Count == 4 && TryPerspectiveCrop(image, points, out crop))
        {
            return true;
        }

        if (TryGetCropRect(points, image.Width, image.Height, out var rect))
        {
            crop = image.Clone(ctx => ctx.Crop(rect));
            return true;
        }

        return false;
    }

    private static bool TryParsePoints(JsonElement pointsEl, out List<DrawingPointF> points)
    {
        points = new List<DrawingPointF>(8);
        if (pointsEl.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        foreach (var point in pointsEl.EnumerateArray())
        {
            if (point.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            var p = point.EnumerateArray().ToArray();
            if (p.Length < 2)
            {
                continue;
            }

            if (!p[0].TryGetSingle(out var x) || !p[1].TryGetSingle(out var y))
            {
                continue;
            }

            points.Add(new DrawingPointF(x, y));
        }

        return points.Count > 0;
    }

    private static bool TryGetCropRect(IReadOnlyList<DrawingPointF> points, int imageW, int imageH, out SixLabors.ImageSharp.Rectangle rect)
    {
        if (points.Count == 0)
        {
            rect = default;
            return false;
        }

        var minX = Math.Max(0, (int)Math.Floor(points.Min(p => p.X)));
        var minY = Math.Max(0, (int)Math.Floor(points.Min(p => p.Y)));
        var maxX = Math.Min(imageW - 1, (int)Math.Ceiling(points.Max(p => p.X)));
        var maxY = Math.Min(imageH - 1, (int)Math.Ceiling(points.Max(p => p.Y)));

        var width = maxX - minX + 1;
        var height = maxY - minY + 1;
        if (width < 4 || height < 4)
        {
            rect = default;
            return false;
        }

        rect = new SixLabors.ImageSharp.Rectangle(minX, minY, width, height);
        return true;
    }

    private static bool TryPerspectiveCrop(Image<Rgb24> source, IReadOnlyList<DrawingPointF> rawPoints, out Image<Rgb24> crop)
    {
        crop = default!;
        if (rawPoints.Count != 4)
        {
            return false;
        }

        var (tl, tr, br, bl) = OrderQuad(rawPoints);
        var widthTop = Distance(tl, tr);
        var widthBottom = Distance(bl, br);
        var heightLeft = Distance(tl, bl);
        var heightRight = Distance(tr, br);
        var targetW = (int)Math.Round(Math.Max(widthTop, widthBottom));
        var targetH = (int)Math.Round(Math.Max(heightLeft, heightRight));
        if (targetW < 4 || targetH < 4)
        {
            return false;
        }

        var dst = new[]
        {
            new DrawingPointF(0, 0),
            new DrawingPointF(targetW - 1, 0),
            new DrawingPointF(targetW - 1, targetH - 1),
            new DrawingPointF(0, targetH - 1)
        };

        var src = new[] { tl, tr, br, bl };
        if (!TrySolveHomography(dst, src, out var h))
        {
            return false;
        }

        var output = new Image<Rgb24>(targetW, targetH);
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var denom = h[6] * x + h[7] * y + 1.0;
                if (Math.Abs(denom) < 1e-6)
                {
                    output[x, y] = default;
                    continue;
                }

                var sx = (h[0] * x + h[1] * y + h[2]) / denom;
                var sy = (h[3] * x + h[4] * y + h[5]) / denom;
                output[x, y] = SampleBilinear(source, sx, sy);
            }
        }

        crop = output;
        return true;
    }

    private static (DrawingPointF Tl, DrawingPointF Tr, DrawingPointF Br, DrawingPointF Bl) OrderQuad(IReadOnlyList<DrawingPointF> points)
    {
        var sorted = points
            .OrderBy(p => p.Y)
            .ThenBy(p => p.X)
            .Take(4)
            .ToArray();
        var top = sorted.Take(2).OrderBy(p => p.X).ToArray();
        var bottom = sorted.Skip(2).OrderBy(p => p.X).ToArray();
        var tl = top[0];
        var tr = top[1];
        var bl = bottom[0];
        var br = bottom[1];
        return (tl, tr, br, bl);
    }

    private static double Distance(DrawingPointF a, DrawingPointF b)
    {
        var dx = a.X - b.X;
        var dy = a.Y - b.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static bool TrySolveHomography(IReadOnlyList<DrawingPointF> from, IReadOnlyList<DrawingPointF> to, out double[] h)
    {
        h = new double[8];
        if (from.Count != 4 || to.Count != 4)
        {
            return false;
        }

        var a = new double[8, 9];
        for (var i = 0; i < 4; i++)
        {
            var x = from[i].X;
            var y = from[i].Y;
            var u = to[i].X;
            var v = to[i].Y;
            var r = i * 2;

            a[r, 0] = x;
            a[r, 1] = y;
            a[r, 2] = 1;
            a[r, 6] = -u * x;
            a[r, 7] = -u * y;
            a[r, 8] = u;

            a[r + 1, 3] = x;
            a[r + 1, 4] = y;
            a[r + 1, 5] = 1;
            a[r + 1, 6] = -v * x;
            a[r + 1, 7] = -v * y;
            a[r + 1, 8] = v;
        }

        for (var col = 0; col < 8; col++)
        {
            var pivot = col;
            var maxAbs = Math.Abs(a[pivot, col]);
            for (var row = col + 1; row < 8; row++)
            {
                var v = Math.Abs(a[row, col]);
                if (v > maxAbs)
                {
                    pivot = row;
                    maxAbs = v;
                }
            }

            if (maxAbs < 1e-8)
            {
                return false;
            }

            if (pivot != col)
            {
                for (var c = col; c <= 8; c++)
                {
                    (a[col, c], a[pivot, c]) = (a[pivot, c], a[col, c]);
                }
            }

            var divisor = a[col, col];
            for (var c = col; c <= 8; c++)
            {
                a[col, c] /= divisor;
            }

            for (var row = 0; row < 8; row++)
            {
                if (row == col)
                {
                    continue;
                }

                var factor = a[row, col];
                if (Math.Abs(factor) < 1e-12)
                {
                    continue;
                }

                for (var c = col; c <= 8; c++)
                {
                    a[row, c] -= factor * a[col, c];
                }
            }
        }

        for (var i = 0; i < 8; i++)
        {
            h[i] = a[i, 8];
        }

        return true;
    }

    private static Rgb24 SampleBilinear(Image<Rgb24> source, double x, double y)
    {
        x = Math.Clamp(x, 0, source.Width - 1);
        y = Math.Clamp(y, 0, source.Height - 1);

        var x0 = (int)Math.Floor(x);
        var y0 = (int)Math.Floor(y);
        var x1 = Math.Min(x0 + 1, source.Width - 1);
        var y1 = Math.Min(y0 + 1, source.Height - 1);
        var dx = x - x0;
        var dy = y - y0;

        var c00 = source[x0, y0];
        var c10 = source[x1, y0];
        var c01 = source[x0, y1];
        var c11 = source[x1, y1];

        static byte Blend(byte a, byte b, byte c, byte d, double dx, double dy)
        {
            var top = a + (b - a) * dx;
            var bottom = c + (d - c) * dx;
            var value = top + (bottom - top) * dy;
            return (byte)Math.Clamp((int)Math.Round(value), 0, 255);
        }

        return new Rgb24(
            Blend(c00.R, c10.R, c01.R, c11.R, dx, dy),
            Blend(c00.G, c10.G, c01.G, c11.G, dx, dy),
            Blend(c00.B, c10.B, c01.B, c11.B, dx, dy));
    }

    private static string SanitizeRecText(string text)
    {
        return text
            .Replace('\t', ' ')
            .Replace('\r', ' ')
            .Replace('\n', ' ')
            .Trim();
    }

    private static int ParseInt(IReadOnlyDictionary<string, string> options, string key, int fallback)
    {
        return options.TryGetValue(key, out var raw) && int.TryParse(raw, out var v) ? v : fallback;
    }

    private static float ParseFloat(IReadOnlyDictionary<string, string> options, string key, float fallback)
    {
        return options.TryGetValue(key, out var raw) &&
               float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v
            : fallback;
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
                    var jsonStart = line.IndexOf('[');
                    if (jsonStart <= 0)
                    {
                        continue;
                    }

                    split = [line[..jsonStart].Trim(), line[jsonStart..].Trim()];
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

    private static E2eEvalResult Evaluate(string gtDir, string predDir, IReadOnlyList<float> iouThresholds)
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
        const float eps = 1e-9f;
        var summaries = new List<E2eSummary>(iouThresholds.Count);
        var fileDetails = new List<E2eFileDetail>(gtFiles.Count * Math.Max(1, iouThresholds.Count));

        foreach (var iouThresh in iouThresholds)
        {
            var numGtChars = 0;
            var gtCount = 0;
            var dtCount = 0;
            var hit = 0;
            var edSum = 0;
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

                var fileHit = 0;
                var fileGtCount = 0;
                var fileDtCount = 0;
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
                        fileHit++;
                    }

                    gtCount++;
                    dtCount++;
                    fileGtCount++;
                    fileDtCount++;
                }

                for (var i = 0; i < dtMatch.Length; i++)
                {
                    if (dtMatch[i])
                    {
                        continue;
                    }

                    edSum += Levenshtein(dts[i].Text, string.Empty);
                    dtCount++;
                    fileDtCount++;
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
                    fileGtCount++;
                }

                fileDetails.Add(new E2eFileDetail(name, iouThresh, fileHit, fileGtCount, fileDtCount));
            }

            var precision = hit / (dtCount + eps);
            var recall = hit / (gtCount + eps);
            var fmeasure = 2f * precision * recall / (precision + recall + eps);
            var avgEditDistImg = gtFiles.Count == 0 ? 0 : (float)edSum / gtFiles.Count;
            var avgEditDistField = edSum / (gtCount + eps);
            var characterAcc = 1f - edSum / (numGtChars + eps);
            summaries.Add(new E2eSummary(iouThresh, precision, recall, fmeasure, characterAcc, avgEditDistField, avgEditDistImg, gtCount, dtCount));
        }

        return new E2eEvalResult(summaries, fileDetails);
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

    private static List<DrawingPointF> ToPoints(float[] coords)
    {
        var points = new List<DrawingPointF>(coords.Length / 2);
        for (var i = 0; i + 1 < coords.Length; i += 2)
        {
            points.Add(new DrawingPointF(coords[i], coords[i + 1]));
        }

        return points;
    }

    private static List<DrawingPointF> IntersectConvex(List<DrawingPointF> subject, List<DrawingPointF> clip)
    {
        var output = new List<DrawingPointF>(subject);
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

    private static bool Inside(DrawingPointF p, DrawingPointF a, DrawingPointF b)
    {
        return ((b.X - a.X) * (p.Y - a.Y) - (b.Y - a.Y) * (p.X - a.X)) >= 0f;
    }

    private static DrawingPointF Intersection(DrawingPointF s, DrawingPointF e, DrawingPointF a, DrawingPointF b)
    {
        var dc = new DrawingPointF(a.X - b.X, a.Y - b.Y);
        var dp = new DrawingPointF(s.X - e.X, s.Y - e.Y);
        var n1 = a.X * b.Y - a.Y * b.X;
        var n2 = s.X * e.Y - s.Y * e.X;
        var n3 = dc.X * dp.Y - dc.Y * dp.X;
        if (Math.Abs(n3) < 1e-6f)
        {
            return e;
        }

        return new DrawingPointF((n1 * dp.X - n2 * dc.X) / n3, (n1 * dp.Y - n2 * dc.Y) / n3);
    }

    private static float PolygonArea(List<DrawingPointF> pts)
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

    private static List<float> ParseIouThresholds(string text)
    {
        var values = text.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(x => float.TryParse(x, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : float.NaN)
            .Where(x => !float.IsNaN(x))
            .Select(x => Math.Clamp(x, 0.01f, 0.99f))
            .Distinct()
            .OrderBy(x => x)
            .ToList();
        if (values.Count == 0)
        {
            values.Add(0.5f);
        }

        return values;
    }

    private sealed record E2eLine(float[] Coords, string Text);
    private sealed record E2eSummary(float IouThreshold, float Precision, float Recall, float Fmeasure, float CharacterAccuracy, float AvgEditDistField, float AvgEditDistImg, int GtCount, int DtCount);
    private sealed record E2eFileDetail(string FileName, float IouThreshold, int Hit, int GtCount, int DtCount);
    private sealed record E2eEvalResult(IReadOnlyList<E2eSummary> Summaries, IReadOnlyList<E2eFileDetail> Files);
    private sealed record RecPreparedSummary(string OutputDir, int TotalSamples, int TrainSamples, int ValSamples, int SkippedAnnotations);
}
