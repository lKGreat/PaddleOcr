namespace PaddleOcr.Training.Cls;

/// <summary>
/// Classification metric evaluator.
/// 1:1 port of Python PaddleOCR ppocr/metrics/cls_metric.py ClsMetric.
/// 
/// Tracks accuracy as correct_num / all_num with configurable main_indicator.
/// Supports reset() and get_metric() interface matching Python.
/// </summary>
internal sealed class ClsMetric
{
    private readonly string _mainIndicator;
    private readonly float _eps;
    private long _correctNum;
    private long _allNum;

    /// <summary>
    /// Creates a new ClsMetric instance.
    /// </summary>
    /// <param name="mainIndicator">Main metric indicator name. Default: "acc"</param>
    public ClsMetric(string mainIndicator = "acc")
    {
        _mainIndicator = mainIndicator;
        _eps = 1e-5f;
        Reset();
    }

    /// <summary>
    /// Update metrics with a batch of predictions and labels.
    /// Matches Python: __call__(self, pred_label, *args, **kwargs)
    /// </summary>
    /// <param name="predictions">Array of (predicted_label, confidence) tuples.</param>
    /// <param name="labels">Array of (ground_truth_label, _) tuples.</param>
    /// <returns>Dictionary with per-batch accuracy.</returns>
    public Dictionary<string, float> Update(
        IReadOnlyList<(string Label, float Confidence)> predictions,
        IReadOnlyList<(string Label, float Confidence)> labels)
    {
        long correct = 0;
        long total = Math.Min(predictions.Count, labels.Count);

        for (var i = 0; i < total; i++)
        {
            if (predictions[i].Label == labels[i].Label)
            {
                correct++;
            }
        }

        _correctNum += correct;
        _allNum += total;

        return new Dictionary<string, float>
        {
            ["acc"] = total > 0 ? (float)correct / (total + _eps) : 0f
        };
    }

    /// <summary>
    /// Update metrics with raw prediction indices and ground truth indices.
    /// </summary>
    public Dictionary<string, float> Update(long[] predictedIndices, long[] groundTruthIndices)
    {
        long correct = 0;
        var total = Math.Min(predictedIndices.Length, groundTruthIndices.Length);

        for (var i = 0; i < total; i++)
        {
            if (predictedIndices[i] == groundTruthIndices[i])
            {
                correct++;
            }
        }

        _correctNum += correct;
        _allNum += total;

        return new Dictionary<string, float>
        {
            ["acc"] = total > 0 ? (float)correct / (total + _eps) : 0f
        };
    }

    /// <summary>
    /// Get the accumulated metric.
    /// Matches Python: get_metric(self) -> {"acc": float}
    /// Resets after getting metrics.
    /// </summary>
    public Dictionary<string, float> GetMetric()
    {
        var acc = _allNum > 0 ? (float)(1.0 * _correctNum / (_allNum + _eps)) : 0f;
        Reset();
        return new Dictionary<string, float>
        {
            ["acc"] = acc
        };
    }

    /// <summary>
    /// Get the main indicator name.
    /// </summary>
    public string MainIndicator => _mainIndicator;

    /// <summary>
    /// Reset accumulated counts.
    /// Matches Python: reset(self)
    /// </summary>
    public void Reset()
    {
        _correctNum = 0;
        _allNum = 0;
    }
}
