using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// NRTR loss with padding mask and optional label smoothing.
/// Aligns target slicing behavior with Paddle rec_nrtr_loss.
/// </summary>
public sealed class NRTRLoss : IRecLoss
{
    private readonly float _labelSmoothing;
    private readonly int _paddingIdx;

    public NRTRLoss(float labelSmoothing = 0.0f, int paddingIdx = 0)
    {
        _labelSmoothing = labelSmoothing;
        _paddingIdx = paddingIdx;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"]; // [B, T, C]
        var targets = batch.ContainsKey("label_gtc") ? batch["label_gtc"] : batch["label"]; // [B, T]

        // Paddle parity: tgt = label[:, 1 : 2 + max(length)]
        var effectiveTargets = targets;
        if (batch.TryGetValue("length", out var lengths) || batch.TryGetValue("target_lengths", out lengths))
        {
            using var lengthsCpu = lengths.to_type(ScalarType.Int64).cpu();
            var lengthArr = lengthsCpu.data<long>().ToArray();
            if (lengthArr.Length > 0)
            {
                var maxLen = (int)Math.Max(0L, lengthArr.Max());
                var targetEndExclusive = Math.Min((int)targets.shape[1], 2 + maxLen);
                if (targetEndExclusive > 1)
                {
                    effectiveTargets = targets.narrow(1, 1, targetEndExclusive - 1);
                }
            }
        }

        var targetTime = (int)effectiveTargets.shape[1];
        var predTime = (int)logits.shape[1];
        var alignedTime = Math.Max(1, Math.Min(targetTime, predTime));
        if (targetTime != alignedTime)
        {
            effectiveTargets = effectiveTargets.narrow(1, 0, alignedTime);
        }

        var effectiveLogits = predTime == alignedTime ? logits : logits.narrow(1, 0, alignedTime);
        var b = effectiveLogits.shape[0];
        var c = effectiveLogits.shape[2];

        var logits2d = effectiveLogits.reshape(b * alignedTime, c);
        var targets1d = effectiveTargets.reshape(b * alignedTime).to(ScalarType.Int64);
        var mask = targets1d.ne(_paddingIdx).to(ScalarType.Float32);

        var loss = functional.cross_entropy(logits2d, targets1d, reduction: Reduction.None);
        loss = (loss * mask).sum() / (mask.sum() + 1e-8f);

        if (_labelSmoothing > 0.0f)
        {
            using var logProbs = functional.log_softmax(logits2d, dim: -1);
            using var uniformLoss = -logProbs.mean(new long[] { -1 });
            var maskedUniformLoss = (uniformLoss * mask).sum() / (mask.sum() + 1e-8f);
            loss = (1.0f - _labelSmoothing) * loss + _labelSmoothing * maskedUniformLoss;
        }

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
