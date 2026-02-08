namespace PaddleOcr.Training;

internal sealed record OfficialSamplerBatch(int Width, int Height, int[] SampleIndices, float? RatioCurrent);

/// <summary>
/// Single-process approximation of PaddleOCR MultiScaleSampler.
/// </summary>
internal sealed class OfficialMultiScaleSampler
{
    private readonly int _sampleCount;
    private readonly bool _shuffle;
    private readonly bool _dsWidth;
    private readonly float _maxW;
    private readonly float _ratioWh;
    private readonly float[] _whRatios;
    private readonly int[] _whRatioSort;
    private readonly List<(int Width, int Height, int BatchSize)> _batchTemplates;

    public OfficialMultiScaleSampler(
        int sampleCount,
        IReadOnlyList<(int Width, int Height)> scales,
        int firstBatchSize,
        bool fixBatchSize,
        IReadOnlyList<int> dividedFactor,
        bool isTraining,
        bool dsWidth,
        IReadOnlyList<float>? whRatios = null,
        IReadOnlyList<int>? whRatioSort = null,
        float ratioWh = 0.8f,
        float maxW = 480f)
    {
        _sampleCount = Math.Max(0, sampleCount);
        _shuffle = isTraining;
        _dsWidth = dsWidth;
        _maxW = Math.Max(1f, maxW);
        _ratioWh = ratioWh;
        _whRatios = whRatios is null ? [] : whRatios.ToArray();
        _whRatioSort = whRatioSort is null ? [] : whRatioSort.ToArray();

        if (_sampleCount == 0)
        {
            _batchTemplates = [];
            return;
        }

        var normalizedScales = NormalizeScales(scales, dividedFactor, isTraining);
        if (normalizedScales.Count == 0)
        {
            normalizedScales.Add((320, 48));
        }

        var baseBatchSize = Math.Max(1, firstBatchSize);
        var baseElements = normalizedScales[0].Width * normalizedScales[0].Height * baseBatchSize;
        var pairs = new List<(int Width, int Height, int BatchSize)>(normalizedScales.Count);
        foreach (var (width, height) in normalizedScales)
        {
            var batchSize = fixBatchSize ? baseBatchSize : Math.Max(1, (int)(baseElements / (double)(width * height)));
            pairs.Add((width, height, batchSize));
        }

        _batchTemplates = BuildBatchTemplates(_sampleCount, pairs);
    }

    public int EstimatedStepsPerEpoch => _batchTemplates.Count;

    public IReadOnlyList<OfficialSamplerBatch> BuildEpochBatches(Random rng)
    {
        if (_sampleCount == 0 || _batchTemplates.Count == 0)
        {
            return [];
        }

        var logicalIndices = Enumerable.Range(0, _sampleCount).ToArray();
        if (_shuffle && !_dsWidth)
        {
            Shuffle(logicalIndices, rng);
        }

        var batches = new List<OfficialSamplerBatch>(_batchTemplates.Count);
        var startIndex = 0;
        foreach (var (baseWidth, height, batchSize) in _batchTemplates)
        {
            var logicalBatch = TakeWithWrap(logicalIndices, startIndex, batchSize);
            startIndex += batchSize;

            float? ratioCurrent = null;
            var sampleIndices = logicalBatch;
            if (_dsWidth)
            {
                sampleIndices = MapToSortedSampleIndices(logicalBatch);
                ratioCurrent = ComputeRatioCurrent(sampleIndices, height);
            }

            var width = baseWidth;
            if (_dsWidth && ratioCurrent.HasValue)
            {
                var rounded = (int)Math.Round(ratioCurrent.Value);
                if (rounded <= 0)
                {
                    rounded = 1;
                }

                width = Math.Max(1, rounded * height);
            }

            batches.Add(new OfficialSamplerBatch(width, height, sampleIndices, ratioCurrent));
        }

        if (_shuffle)
        {
            ShuffleList(batches, rng);
        }

        return batches;
    }

    private int[] MapToSortedSampleIndices(IReadOnlyList<int> logicalBatch)
    {
        if (_whRatioSort.Length != _sampleCount)
        {
            return logicalBatch.ToArray();
        }

        var mapped = new int[logicalBatch.Count];
        for (var i = 0; i < logicalBatch.Count; i++)
        {
            var logical = logicalBatch[i];
            if (logical < 0 || logical >= _whRatioSort.Length)
            {
                mapped[i] = Math.Clamp(logical, 0, _sampleCount - 1);
                continue;
            }

            var actual = _whRatioSort[logical];
            mapped[i] = actual >= 0 && actual < _sampleCount ? actual : Math.Clamp(logical, 0, _sampleCount - 1);
        }

        return mapped;
    }

    private float ComputeRatioCurrent(IReadOnlyList<int> sampleIndices, int currentHeight)
    {
        if (sampleIndices.Count == 0)
        {
            return 1f;
        }

        var sum = 0.0;
        for (var i = 0; i < sampleIndices.Count; i++)
        {
            var idx = sampleIndices[i];
            var ratio = idx >= 0 && idx < _whRatios.Length ? _whRatios[idx] : 1f;
            sum += ratio > 0f ? ratio : 1f;
        }

        var mean = (float)(sum / sampleIndices.Count);
        var maxRatio = _maxW / Math.Max(1, currentHeight);
        if (mean * currentHeight >= _maxW)
        {
            mean = maxRatio;
        }

        return float.IsFinite(mean) && mean > 0f ? mean : Math.Max(1f, _ratioWh);
    }

    private static List<(int Width, int Height, int BatchSize)> BuildBatchTemplates(
        int sampleCount,
        IReadOnlyList<(int Width, int Height, int BatchSize)> pairs)
    {
        var templates = new List<(int Width, int Height, int BatchSize)>();
        var current = 0;
        while (current < sampleCount)
        {
            for (var i = 0; i < pairs.Count; i++)
            {
                templates.Add(pairs[i]);
                current += pairs[i].BatchSize;
            }
        }

        return templates;
    }

    private static List<(int Width, int Height)> NormalizeScales(
        IReadOnlyList<(int Width, int Height)> scales,
        IReadOnlyList<int> dividedFactor,
        bool isTraining)
    {
        var widthFactor = dividedFactor.Count > 0 ? Math.Max(1, dividedFactor[0]) : 8;
        var heightFactor = dividedFactor.Count > 1 ? Math.Max(1, dividedFactor[1]) : 16;
        var normalized = new List<(int Width, int Height)>(scales.Count);
        for (var i = 0; i < scales.Count; i++)
        {
            var (w, h) = scales[i];
            if (w <= 0 || h <= 0)
            {
                continue;
            }

            if (isTraining)
            {
                w = Math.Max(widthFactor, (w / widthFactor) * widthFactor);
                h = Math.Max(heightFactor, (h / heightFactor) * heightFactor);
            }

            normalized.Add((w, h));
        }

        return normalized;
    }

    private static int[] TakeWithWrap(IReadOnlyList<int> source, int start, int count)
    {
        var result = new int[count];
        if (source.Count == 0)
        {
            return result;
        }

        for (var i = 0; i < count; i++)
        {
            result[i] = source[(start + i) % source.Count];
        }

        return result;
    }

    private static void Shuffle(int[] values, Random rng)
    {
        for (var i = values.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static void ShuffleList<T>(IList<T> values, Random rng)
    {
        for (var i = values.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }
}
