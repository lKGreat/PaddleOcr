using PaddleOcr.Data;
using PaddleOcr.Data.LabelEncoders;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Training;

internal sealed class ConfigRecDataset
{
    private readonly List<(string ImagePath, string Text)> _samples;
    private readonly int _targetH;
    private readonly int _targetW;
    private readonly int[] _multiScaleWidths;
    private readonly bool _useMultiScale;
    private readonly int _maxTextLength;
    private readonly IRecLabelEncoder _ctcEncoder;
    private readonly IRecLabelEncoder? _gtcEncoder;
    private readonly IRecTrainingResize _resizer;
    private readonly bool _enableAugmentation;

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
        int[]? multiScaleWidths = null)
    {
        _targetH = targetH;
        _targetW = targetW;
        _maxTextLength = maxTextLength;
        _ctcEncoder = ctcEncoder;
        _gtcEncoder = gtcEncoder;
        _resizer = resizer ?? new RecResizeImg();
        _enableAugmentation = enableAugmentation;
        _useMultiScale = useMultiScale;
        _multiScaleWidths = (multiScaleWidths is { Length: > 0 } ? multiScaleWidths : [targetW])
            .Where(w => w > 0)
            .ToArray();
        if (_multiScaleWidths.Length == 0)
        {
            _multiScaleWidths = [targetW];
        }

        _samples = LoadSamples(labelFiles, dataDir);
    }

    public int Count => _samples.Count;

    public IEnumerable<ConfigRecBatch> GetBatches(int batchSize, bool shuffle, Random rng)
    {
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
            var batchW = _useMultiScale ? _multiScaleWidths[widthIdx++ % _multiScaleWidths.Length] : _targetW;

            var imageList = new List<float[]>(take);
            var ctcLabels = new List<long[]>(take);
            var gtcLabels = new List<long[]>(take);
            var lengths = new List<int>(take);
            var validRatios = new List<float>(take);

            for (var bi = 0; bi < take; bi++)
            {
                var sample = _samples[indices[offset + bi]];
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

                using var img = Image.Load<Rgb24>(sample.ImagePath);
                if (_enableAugmentation)
                {
                    RecAugmentation.ApplyAugmentation(img);
                }

                var resized = _resizer.Resize(img, 3, _targetH, batchW);
                imageList.Add(resized.Data);
                ctcLabels.Add(FitLabel(encodedCtc.Label, _maxTextLength));
                gtcLabels.Add(FitLabel(encodedGtc.Label, _maxTextLength));
                lengths.Add(Math.Min(_maxTextLength, encodedCtc.Length));
                validRatios.Add(resized.ValidRatio);
            }

            if (imageList.Count == 0)
            {
                continue;
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

            yield return new ConfigRecBatch(
                images,
                labelCtcFlat,
                labelGtcFlat,
                lengths.ToArray(),
                validRatios.ToArray(),
                validCount,
                batchW);
        }
    }

    private static long[] FitLabel(long[] label, int targetLen)
    {
        var result = new long[targetLen];
        var len = Math.Min(targetLen, label.Length);
        Array.Copy(label, 0, result, 0, len);
        return result;
    }

    private static List<(string ImagePath, string Text)> LoadSamples(IReadOnlyList<string> labelFiles, string dataDir)
    {
        var all = new List<(string, string)>();
        foreach (var labelFile in labelFiles)
        {
            if (!File.Exists(labelFile))
            {
                throw new FileNotFoundException($"Label file not found: {labelFile}");
            }

            foreach (var line in File.ReadLines(labelFile))
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var split = line.Split('\t', 2);
                if (split.Length != 2)
                {
                    continue;
                }

                var img = split[0].Trim();
                var fullPath = Path.IsPathRooted(img) ? img : Path.GetFullPath(Path.Combine(dataDir, img));
                if (!File.Exists(fullPath))
                {
                    continue;
                }

                var text = split[1].Trim();
                if (text.Length == 0)
                {
                    continue;
                }

                all.Add((fullPath, text));
            }
        }

        if (all.Count == 0)
        {
            throw new InvalidOperationException("No valid rec samples found.");
        }

        return all;
    }
}

internal sealed record ConfigRecBatch(
    float[] Images,
    long[] LabelCtc,
    long[] LabelGtc,
    int[] Lengths,
    float[] ValidRatios,
    int Batch,
    int Width);
