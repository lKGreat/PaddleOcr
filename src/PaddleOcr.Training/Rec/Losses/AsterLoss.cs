using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// AsterLoss：ASTER 算法的组合损失（CrossEntropy + CosineEmbeddingLoss）。
/// 参考: ppocr/losses/rec_aster_loss.py
/// </summary>
public sealed class AsterLoss : IRecLoss
{
    private readonly bool _sampleNormalize;

    public AsterLoss(bool sampleNormalize = true)
    {
        _sampleNormalize = sampleNormalize;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var recPred = predictions["rec_pred"]; // [B, T, numClasses]
        var targets = batch["label"].to(ScalarType.Int64);
        var batchSize = recPred.shape[0];
        var maxLength = recPred.shape[1];

        // Truncate targets
        targets = targets.slice(1, 0, maxLength, 1);

        // Compute mask from label lengths if available
        Tensor mask;
        if (batch.TryGetValue("length", out var lengths))
        {
            mask = torch.zeros(batchSize, maxLength, device: recPred.device);
            for (var i = 0; i < batchSize; i++)
            {
                var len = Math.Min((int)lengths[i].item<long>(), (int)maxLength);
                if (len > 0)
                {
                    mask[i, ..(int)len] = 1;
                }
            }
        }
        else
        {
            mask = torch.ones(batchSize, maxLength, device: recPred.device);
        }

        // Rec loss: cross-entropy with masking
        var logSoftmax = functional.log_softmax(recPred.reshape(-1, recPred.shape[2]), dim: 1);
        var flatTargets = targets.reshape(-1);
        var ce = functional.nll_loss(logSoftmax, flatTargets, reduction: Reduction.None);
        var maskedCe = (ce * mask.reshape(-1)).sum();

        if (_sampleNormalize)
        {
            maskedCe = maskedCe / batchSize;
        }

        // Semantic loss (cosine embedding)
        var semLoss = torch.tensor(0.0f, device: recPred.device);
        if (predictions.TryGetValue("embedding_vectors", out var embedVec) &&
            batch.TryGetValue("sem_target", out var semTarget))
        {
            var similarity = functional.cosine_similarity(embedVec, semTarget, dim: 1);
            semLoss = (1.0f - similarity).mean();
        }

        var loss = maskedCe + semLoss * 0.1f;
        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
