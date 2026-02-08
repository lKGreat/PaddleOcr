using PaddleOcr.Data;
using PaddleOcr.Data.LabelEncoders;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Training;

internal sealed class ConfigRecDataset
{
    private readonly List<ConfigRecSample> _samples;
    private readonly int _targetH;
    private readonly int _targetW;
    private readonly int[] _multiScaleWidths;
    private readonly int[] _multiScaleHeights;
    private readonly bool _useMultiScale;
    private readonly int _maxTextLength;
    private readonly IRecLabelEncoder _ctcEncoder;
    private readonly IRecLabelEncoder? _gtcEncoder;
    private readonly IRecTrainingResize _resizer;
    private readonly bool _enableAugmentation;
    private readonly RecConcatAugmentOptions _concatOptions;
    private readonly bool _dsWidth;
    private readonly float[] _whRatios;
    private readonly int[] _whRatioSort;

    public ConfigRecDataset(
        IReadOnlyList<string> labelFiles,
        string dataDir,
        int targetH,
        int targetW,
        int maxTextLength,
        IRecLabelEncoder ctcEncoder,
        IRecLabelEncoder? gtcEncoder = null,
        IRecTrainingResize? resizer = null,
        bool enableAugmentation = false,
        bool useMultiScale = false,
        int[]? multiScaleWidths = null,
        int[]? multiScaleHeights = null,
        string? delimiter = null,
        IReadOnlyList<float>? ratioList = null,
        int seed = 1024,
        RecConcatAugmentOptions? concatOptions = null,
        bool dsWidth = false)
    {
        _targetH = targetH;
        _targetW = targetW;
        _maxTextLength = maxTextLength;
        _ctcEncoder = ctcEncoder;
        _gtcEncoder = gtcEncoder;
        _resizer = resizer ?? new RecResizeImg();
        _enableAugmentation = enableAugmentation;
        _concatOptions = concatOptions ?? RecConcatAugmentOptions.Disabled;
        _useMultiScale = useMultiScale;
        _dsWidth = dsWidth;
        _multiScaleWidths = (multiScaleWidths is { Length: > 0 } ? multiScaleWidths : [targetW])
            .Where(w => w > 0)
            .ToArray();
        if (_multiScaleWidths.Length == 0)
        {
            _multiScaleWidths = [targetW];
        }

        _multiScaleHeights = (multiScaleHeights is { Length: > 0 } ? multiScaleHeights : [targetH])
            .Where(h => h > 0)
            .ToArray();
        if (_multiScaleHeights.Length == 0)
        {
            _multiScaleHeights = [targetH];
        }

        _samples = LoadSamples(labelFiles, dataDir, delimiter, ratioList, seed, _dsWidth);

        if (_dsWidth)
        {
            _whRatios = _samples.Select(x => x.WidthHeightRatio ?? 1f).ToArray();
            _whRatioSort = Enumerable.Range(0, _whRatios.Length)
                .OrderBy(i => _whRatios[i])
                .ToArray();
        }
        else
        {
            _whRatios = [];
            _whRatioSort = [];
        }
    }

    public int Count => _samples.Count;
    public IReadOnlyList<float> WidthHeightRatios => _whRatios;
    public IReadOnlyList<int> WidthHeightSortIndices => _whRatioSort;

    public IEnumerable<ConfigRecBatch> GetBatches(
        int batchSize,
        bool shuffle,
        Random rng,
        bool dropLast = false,
        OfficialMultiScaleSampler? sampler = null)
    {
        if (sampler is not null)
        {
            foreach (var sampledBatch in sampler.BuildEpochBatches(rng))
            {
                var built = BuildBatch(sampledBatch.SampleIndices, sampledBatch.Height, sampledBatch.Width, rng);
                if (built is not null)
                {
                    yield return built;
                }
            }

            yield break;
        }

        var indices = Enumerable.Range(0, _samples.Count).ToList();
        if (shuffle)
        {
            for (var i = indices.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        var widthIdx = 0;
        for (var offset = 0; offset < indices.Count; offset += batchSize)
        {
            var take = Math.Min(batchSize, indices.Count - offset);
            if (dropLast && take < batchSize)
            {
                break;
            }

            var batchW = _useMultiScale ? _multiScaleWidths[widthIdx % _multiScaleWidths.Length] : _targetW;
            var batchH = _useMultiScale ? _multiScaleHeights[widthIdx % _multiScaleHeights.Length] : _targetH;
            widthIdx++;

            var slice = indices.Skip(offset).Take(take).ToArray();
            var built = BuildBatch(slice, batchH, batchW, rng);
            if (built is not null)
            {
                yield return built;
            }
        }
    }

    private ConfigRecBatch? BuildBatch(IReadOnlyList<int> sampleIndices, int batchH, int batchW, Random rng)
    {
        if (sampleIndices.Count == 0)
        {
            return null;
        }

        var imageList = new List<float[]>(sampleIndices.Count);
        var ctcLabels = new List<long[]>(sampleIndices.Count);
        var gtcLabels = new List<long[]>(sampleIndices.Count);
        var lengths = new List<int>(sampleIndices.Count);
        var validRatios = new List<float>(sampleIndices.Count);

        for (var i = 0; i < sampleIndices.Count; i++)
        {
            var sample = _samples[sampleIndices[i]];
            var encodedCtc = _ctcEncoder.Encode(sample.Text);
            if (encodedCtc is null)
            {
                continue;
            }

            var encodedGtc = _gtcEncoder?.Encode(sample.Text) ?? encodedCtc;
            if (encodedGtc is null)
            {
                continue;
            }

            using var baseImg = Image.Load<Rgb24>(sample.ImagePath);
            var (augImg, augText) = MaybeConcatAugment(baseImg, sample.Text, rng);
            if (_enableAugmentation)
            {
                RecAugmentation.ApplyAugmentation(augImg);
            }

            var resized = _resizer.Resize(augImg, 3, batchH, batchW);
            imageList.Add(resized.Data);
            var finalCtc = _ctcEncoder.Encode(augText) ?? encodedCtc;
            var finalGtc = _gtcEncoder?.Encode(augText) ?? encodedGtc;
            ctcLabels.Add(FitLabel(finalCtc.Label, _maxTextLength));
            gtcLabels.Add(FitLabel(finalGtc.Label, _maxTextLength));
            lengths.Add(Math.Min(_maxTextLength, finalCtc.Length));
            validRatios.Add(resized.ValidRatio);
            augImg.Dispose();
        }

        if (imageList.Count == 0)
        {
            return null;
        }

        var validCount = imageList.Count;
        var singleImageSize = imageList[0].Length;
        var images = new float[validCount * singleImageSize];
        var labelCtcFlat = new long[validCount * _maxTextLength];
        var labelGtcFlat = new long[validCount * _maxTextLength];
        for (var i = 0; i < validCount; i++)
        {
            Array.Copy(imageList[i], 0, images, i * singleImageSize, singleImageSize);
            Array.Copy(ctcLabels[i], 0, labelCtcFlat, i * _maxTextLength, _maxTextLength);
            Array.Copy(gtcLabels[i], 0, labelGtcFlat, i * _maxTextLength, _maxTextLength);
        }

        return new ConfigRecBatch(
            images,
            labelCtcFlat,
            labelGtcFlat,
            lengths.ToArray(),
            validRatios.ToArray(),
            validCount,
            batchH,
            batchW);
    }

    private (Image<Rgb24> Image, string Text) MaybeConcatAugment(Image<Rgb24> image, string text, Random rng)
    {
        if (!_concatOptions.Enabled || rng.NextDouble() > _concatOptions.Prob)
        {
            return (image.Clone(), text);
        }

        var current = image.Clone();
        var currentText = text;

        for (var i = 0; i < _concatOptions.ExtDataNum; i++)
        {
            var extSample = _samples[rng.Next(_samples.Count)];
            if (currentText.Length + extSample.Text.Length > _maxTextLength)
            {
                break;
            }

            using var extImg = Image.Load<Rgb24>(extSample.ImagePath);
            var concatRatio = (current.Width / (float)current.Height) + (extImg.Width / (float)extImg.Height);
            if (concatRatio > _concatOptions.MaxWhRatio)
            {
                break;
            }

            var concatHeight = _concatOptions.ImageHeight > 0 ? _concatOptions.ImageHeight : _targetH;
            var oriW = Math.Max(1, (int)Math.Round(current.Width / (float)current.Height * concatHeight));
            var extW = Math.Max(1, (int)Math.Round(extImg.Width / (float)extImg.Height * concatHeight));
            using var resizedCurrent = current.Clone(ctx => ctx.Resize(oriW, concatHeight));
            using var resizedExt = extImg.Clone(ctx => ctx.Resize(extW, concatHeight));
            var canvas = new Image<Rgb24>(oriW + extW, concatHeight);
            canvas.Mutate(ctx =>
            {
                ctx.DrawImage(resizedCurrent, new Point(0, 0), 1f);
                ctx.DrawImage(resizedExt, new Point(oriW, 0), 1f);
            });

            current.Dispose();
            current = canvas;
            currentText += extSample.Text;
        }

        return (current, currentText);
    }

    private static long[] FitLabel(long[] label, int targetLen)
    {
        var result = new long[targetLen];
        var len = Math.Min(targetLen, label.Length);
        Array.Copy(label, 0, result, 0, len);
        return result;
    }

    private static List<ConfigRecSample> LoadSamples(
        IReadOnlyList<string> labelFiles,
        string dataDir,
        string? delimiter,
        IReadOnlyList<float>? ratioList,
        int seed,
        bool parseWidthHeightRatio)
    {
        var all = new List<ConfigRecSample>();
        for (var fileIdx = 0; fileIdx < labelFiles.Count; fileIdx++)
        {
            var labelFile = labelFiles[fileIdx];
            if (!File.Exists(labelFile))
            {
                throw new FileNotFoundException($"Label file not found: {labelFile}");
            }

            var lines = File.ReadLines(labelFile)
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .ToList();
            var ratio = ResolveRatio(ratioList, fileIdx);
            if (ratio <= 0f)
            {
                continue;
            }

            var selectedLines = SelectByRatio(lines, ratio, seed + fileIdx);
            foreach (var line in selectedLines)
            {
                if (!TryParseSampleLine(line, delimiter, parseWidthHeightRatio, out var img, out var text, out var whRatio))
                {
                    continue;
                }

                var fullPath = Path.IsPathRooted(img) ? img : Path.GetFullPath(Path.Combine(dataDir, img));
                if (!File.Exists(fullPath))
                {
                    continue;
                }

                var ratioResolved = whRatio;
                if (parseWidthHeightRatio && (!ratioResolved.HasValue || ratioResolved.Value <= 0f))
                {
                    var info = Image.Identify(fullPath);
                    if (info is not null && info.Height > 0)
                    {
                        ratioResolved = info.Width / (float)info.Height;
                    }
                }

                all.Add(new ConfigRecSample(fullPath, text, ratioResolved));
            }
        }

        if (all.Count == 0)
        {
            throw new InvalidOperationException("No valid rec samples found.");
        }

        return all;
    }

    private static bool TryParseSampleLine(
        string line,
        string? delimiter,
        bool parseWidthHeightRatio,
        out string imagePath,
        out string text,
        out float? widthHeightRatio)
    {
        imagePath = string.Empty;
        text = string.Empty;
        widthHeightRatio = null;
        if (!parseWidthHeightRatio)
        {
            return RecLabelLineParser.TryParse(line, delimiter, out imagePath, out text);
        }

        if (TryParseWithWidthHeight(line, delimiter, out imagePath, out text, out var ratio))
        {
            widthHeightRatio = ratio;
            return true;
        }

        return RecLabelLineParser.TryParse(line, delimiter, out imagePath, out text);
    }

    private static bool TryParseWithWidthHeight(
        string line,
        string? delimiter,
        out string imagePath,
        out string text,
        out float ratio)
    {
        imagePath = string.Empty;
        text = string.Empty;
        ratio = 0f;

        var normalizedDelimiter = string.IsNullOrWhiteSpace(delimiter)
            ? "\t"
            : delimiter
                .Replace("\\t", "\t", StringComparison.Ordinal)
                .Replace("\\n", "\n", StringComparison.Ordinal)
                .Replace("\\r", "\r", StringComparison.Ordinal);
        if (string.IsNullOrEmpty(normalizedDelimiter))
        {
            return false;
        }

        var parts = line.Split([normalizedDelimiter], StringSplitOptions.None);
        if (parts.Length < 4)
        {
            return false;
        }

        if (!float.TryParse(parts[^2].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var width) ||
            !float.TryParse(parts[^1].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var height) ||
            width <= 0f ||
            height <= 0f)
        {
            return false;
        }

        imagePath = parts[0].Trim();
        text = string.Join(normalizedDelimiter, parts.Skip(1).Take(parts.Length - 3)).Trim();
        if (imagePath.Length == 0 || text.Length == 0)
        {
            return false;
        }

        ratio = width / height;
        return ratio > 0f && float.IsFinite(ratio);
    }

    private static IEnumerable<string> SelectByRatio(IReadOnlyList<string> lines, float ratio, int seed)
    {
        if (lines.Count == 0)
        {
            return [];
        }

        if (ratio >= 0.9999f)
        {
            return lines;
        }

        var sampleCount = Math.Clamp((int)Math.Round(lines.Count * ratio), 1, lines.Count);
        var indices = Enumerable.Range(0, lines.Count).ToArray();
        var rng = new Random(seed);
        for (var i = indices.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices.Take(sampleCount).Select(i => lines[i]);
    }

    private static float ResolveRatio(IReadOnlyList<float>? ratioList, int fileIndex)
    {
        if (ratioList is null || ratioList.Count == 0)
        {
            return 1f;
        }

        if (ratioList.Count == 1)
        {
            return Math.Clamp(ratioList[0], 0f, 1f);
        }

        if (fileIndex < ratioList.Count)
        {
            return Math.Clamp(ratioList[fileIndex], 0f, 1f);
        }

        return 1f;
    }
}

internal sealed record ConfigRecBatch(
    float[] Images,
    long[] LabelCtc,
    long[] LabelGtc,
    int[] Lengths,
    float[] ValidRatios,
    int Batch,
    int Height,
    int Width);

internal sealed record ConfigRecSample(string ImagePath, string Text, float? WidthHeightRatio);
