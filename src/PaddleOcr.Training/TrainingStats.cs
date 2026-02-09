using System.Collections.Concurrent;
using System.Text;

namespace PaddleOcr.Training;

/// <summary>
/// Track a series of values and provide access to smoothed values over a window.
/// Uses median smoothing matching Python PaddleOCR ppocr/utils/stats.py SmoothedValue.
/// </summary>
internal sealed class SmoothedValue
{
    private readonly int _windowSize;
    private readonly Queue<double> _deque;

    public SmoothedValue(int windowSize)
    {
        _windowSize = windowSize;
        _deque = new Queue<double>(windowSize);
    }

    public void AddValue(double value)
    {
        if (_deque.Count >= _windowSize)
        {
            _deque.Dequeue();
        }
        _deque.Enqueue(value);
    }

    /// <summary>
    /// Get the median value of the current window.
    /// Matches Python: np.median(self.deque)
    /// </summary>
    public double GetMedianValue()
    {
        if (_deque.Count == 0)
        {
            return 0.0;
        }

        var sorted = _deque.OrderBy(x => x).ToArray();
        var n = sorted.Length;
        if (n % 2 == 0)
        {
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }
        return sorted[n / 2];
    }

    /// <summary>
    /// Get the mean value of the current window.
    /// </summary>
    public double GetMeanValue()
    {
        return _deque.Count == 0 ? 0.0 : _deque.Average();
    }

    public int Count => _deque.Count;
}

/// <summary>
/// Training statistics tracker with median-smoothed values.
/// 1:1 port of Python PaddleOCR ppocr/utils/stats.py TrainingStats.
/// 
/// Usage:
///   var stats = new TrainingStats(windowSize: 20, ["lr"]);
///   stats.Update(new Dictionary&lt;string, double&gt; { ["loss"] = 0.5, ["lr"] = 0.001 });
///   var smoothed = stats.Get();           // OrderedDict with median-smoothed values
///   var logStr = stats.Log();             // "loss: 0.500000, lr: 0.001000"
/// </summary>
internal sealed class TrainingStats
{
    private readonly int _windowSize;
    private readonly Dictionary<string, SmoothedValue> _smoothedLossesAndMetrics;

    public TrainingStats(int windowSize, IEnumerable<string>? statsKeys = null)
    {
        _windowSize = Math.Max(1, windowSize);
        _smoothedLossesAndMetrics = new Dictionary<string, SmoothedValue>();
        if (statsKeys is not null)
        {
            foreach (var key in statsKeys)
            {
                _smoothedLossesAndMetrics[key] = new SmoothedValue(_windowSize);
            }
        }
    }

    /// <summary>
    /// Update tracked values. New keys are automatically added.
    /// Matches Python: for k, v in stats.items(): smoothed[k].add_value(v)
    /// </summary>
    public void Update(IReadOnlyDictionary<string, double> stats)
    {
        foreach (var (k, v) in stats)
        {
            if (!_smoothedLossesAndMetrics.TryGetValue(k, out var sv))
            {
                sv = new SmoothedValue(_windowSize);
                _smoothedLossesAndMetrics[k] = sv;
            }
            sv.AddValue(v);
        }
    }

    /// <summary>
    /// Convenience overload accepting float values.
    /// </summary>
    public void Update(IReadOnlyDictionary<string, float> stats)
    {
        foreach (var (k, v) in stats)
        {
            if (!_smoothedLossesAndMetrics.TryGetValue(k, out var sv))
            {
                sv = new SmoothedValue(_windowSize);
                _smoothedLossesAndMetrics[k] = sv;
            }
            sv.AddValue(v);
        }
    }

    /// <summary>
    /// Convenience: update a single key-value pair.
    /// </summary>
    public void Update(string key, double value)
    {
        if (!_smoothedLossesAndMetrics.TryGetValue(key, out var sv))
        {
            sv = new SmoothedValue(_windowSize);
            _smoothedLossesAndMetrics[key] = sv;
        }
        sv.AddValue(value);
    }

    /// <summary>
    /// Get smoothed statistics. Extras are prepended without smoothing.
    /// Returns OrderedDictionary matching Python: round(v.get_median_value(), 6)
    /// </summary>
    public Dictionary<string, double> Get(IReadOnlyDictionary<string, double>? extras = null)
    {
        var stats = new Dictionary<string, double>();
        if (extras is not null)
        {
            foreach (var (k, v) in extras)
            {
                stats[k] = v;
            }
        }

        foreach (var (k, v) in _smoothedLossesAndMetrics)
        {
            stats[k] = Math.Round(v.GetMedianValue(), 6);
        }

        return stats;
    }

    /// <summary>
    /// Format smoothed stats as a log string.
    /// Matches Python: ", ".join(["{}: {:x<6f}".format(k, v)])
    /// </summary>
    public string Log(IReadOnlyDictionary<string, double>? extras = null)
    {
        var d = Get(extras);
        var sb = new StringBuilder();
        var first = true;
        foreach (var (k, v) in d)
        {
            if (!first) sb.Append(", ");
            sb.Append($"{k}: {v:F6}");
            first = false;
        }
        return sb.ToString();
    }

    /// <summary>
    /// Get the raw SmoothedValue for a specific key (for advanced inspection).
    /// </summary>
    public SmoothedValue? GetSmoothedValue(string key)
    {
        return _smoothedLossesAndMetrics.TryGetValue(key, out var sv) ? sv : null;
    }

    /// <summary>
    /// Get all currently tracked keys.
    /// </summary>
    public IReadOnlyCollection<string> Keys => _smoothedLossesAndMetrics.Keys;
}

/// <summary>
/// Per-iteration timing and throughput tracker for training loops.
/// Provides reader cost, batch cost, IPS, and ETA calculation matching Python PaddleOCR.
/// </summary>
internal sealed class IterationTimer
{
    private readonly System.Diagnostics.Stopwatch _batchWatch = new();
    private readonly System.Diagnostics.Stopwatch _readerWatch = new();
    private readonly SmoothedValue _batchCost;
    private readonly SmoothedValue _readerCost;
    private int _totalSamples;
    private int _printBatchStep;

    public IterationTimer(int smoothWindow = 20, int printBatchStep = 10)
    {
        _batchCost = new SmoothedValue(smoothWindow);
        _readerCost = new SmoothedValue(smoothWindow);
        _printBatchStep = Math.Max(1, printBatchStep);
    }

    /// <summary>Start timing data loading (call before batch loading).</summary>
    public void StartReader()
    {
        _readerWatch.Restart();
    }

    /// <summary>End timing data loading (call after batch is loaded).</summary>
    public void EndReader()
    {
        _readerWatch.Stop();
        _readerCost.AddValue(_readerWatch.Elapsed.TotalSeconds);
    }

    /// <summary>Start timing the full batch step (forward+backward+update).</summary>
    public void StartBatch()
    {
        _batchWatch.Restart();
    }

    /// <summary>End timing the full batch step.</summary>
    public void EndBatch(int sampleCount)
    {
        _batchWatch.Stop();
        _batchCost.AddValue(_batchWatch.Elapsed.TotalSeconds);
        _totalSamples += sampleCount;
    }

    /// <summary>Average reader cost in seconds (median-smoothed).</summary>
    public double AvgReaderCost => _readerCost.GetMedianValue();

    /// <summary>Average batch cost in seconds (median-smoothed).</summary>
    public double AvgBatchCost => _batchCost.GetMedianValue();

    /// <summary>Images per second based on recent batch cost.</summary>
    public double Ips
    {
        get
        {
            var cost = _batchCost.GetMedianValue();
            return cost > 0 ? _printBatchStep / cost : 0;
        }
    }

    /// <summary>
    /// Calculate ETA string based on remaining iterations.
    /// </summary>
    public string GetEta(int currentEpoch, int totalEpochs, int currentBatchIdx, int totalBatches)
    {
        var remainingBatches = ((long)(totalEpochs - currentEpoch)) * totalBatches + (totalBatches - currentBatchIdx - 1);
        var batchCost = _batchCost.GetMedianValue();
        if (batchCost <= 0 || remainingBatches <= 0)
        {
            return "0:00:00";
        }

        var etaSeconds = (long)(remainingBatches * batchCost);
        return TimeSpan.FromSeconds(etaSeconds).ToString(@"d\:hh\:mm\:ss");
    }

    public void Reset()
    {
        _totalSamples = 0;
    }
}
