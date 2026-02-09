namespace PaddleOcr.Inference.Cls;

/// <summary>
/// Classification post-processor.
/// 1:1 port of Python PaddleOCR ppocr/postprocess/cls_postprocess.py ClsPostProcess.
/// 
/// Applies argmax to model output and maps to label string.
/// Default labels: ['0', '180'] for text direction classification.
/// </summary>
public sealed class ClsPostProcess
{
    private readonly IReadOnlyList<string> _labelList;

    /// <summary>
    /// Creates a new ClsPostProcess instance.
    /// </summary>
    /// <param name="labelList">List of class labels. Default: ['0', '180']</param>
    public ClsPostProcess(IReadOnlyList<string>? labelList = null)
    {
        _labelList = labelList ?? new[] { "0", "180" };
    }

    /// <summary>
    /// Decode classification predictions.
    /// Matches Python: __call__(self, preds, label=None, *args, **kwargs)
    /// </summary>
    /// <param name="predictions">Model output logits [B, num_classes].</param>
    /// <param name="numClasses">Number of classes.</param>
    /// <returns>Array of (label, confidence) for each sample.</returns>
    public ClsResult[] Decode(float[] predictions, int batchSize, int numClasses)
    {
        var results = new ClsResult[batchSize];

        for (var i = 0; i < batchSize; i++)
        {
            var offset = i * numClasses;

            // Softmax
            var maxVal = float.MinValue;
            for (var c = 0; c < numClasses; c++)
            {
                if (predictions[offset + c] > maxVal)
                {
                    maxVal = predictions[offset + c];
                }
            }

            var sumExp = 0f;
            var probs = new float[numClasses];
            for (var c = 0; c < numClasses; c++)
            {
                probs[c] = MathF.Exp(predictions[offset + c] - maxVal);
                sumExp += probs[c];
            }

            // Argmax
            var bestIdx = 0;
            var bestProb = 0f;
            for (var c = 0; c < numClasses; c++)
            {
                probs[c] /= sumExp;
                if (probs[c] > bestProb)
                {
                    bestProb = probs[c];
                    bestIdx = c;
                }
            }

            var label = bestIdx < _labelList.Count ? _labelList[bestIdx] : bestIdx.ToString();
            results[i] = new ClsResult(label, bestProb, bestIdx);
        }

        return results;
    }
}

/// <summary>
/// Classification result for a single sample.
/// </summary>
/// <param name="Label">Predicted label string (e.g., "0" or "180").</param>
/// <param name="Confidence">Prediction confidence (softmax probability).</param>
/// <param name="ClassIndex">Predicted class index.</param>
public sealed record ClsResult(string Label, float Confidence, int ClassIndex);
