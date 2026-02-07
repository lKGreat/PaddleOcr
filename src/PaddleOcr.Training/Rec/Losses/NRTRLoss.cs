using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// NRTRLoss：NRTR 损失，CrossEntropy + label smoothing + padding mask。
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
        var targets = batch["label"]; // [B, T]

        // 创建 padding mask
        var mask = targets.ne(_paddingIdx).to(ScalarType.Float32); // [B, T]

        // Reshape: [B, T, C] -> [B*T, C]
        var b = logits.shape[0];
        var t = logits.shape[1];
        var c = logits.shape[2];
        logits = logits.reshape(b * t, c);
        targets = targets.reshape(b * t).to(ScalarType.Int64);
        mask = mask.reshape(b * t);

        // CrossEntropy loss
        var loss = functional.cross_entropy(logits, targets, reduction: Reduction.None);
        
        // Apply mask
        loss = (loss * mask).sum() / (mask.sum() + 1e-8f);

        // Label smoothing：对非 padding 位置应用标准 label smoothing
        // 参考: https://arxiv.org/abs/1512.00567
        if (_labelSmoothing > 0.0f)
        {
            using var logProbs = functional.log_softmax(logits, dim: -1); // [B*T, C]
            // 均匀分布的交叉熵 = -mean(log_softmax)（对每个时间步）
            using var uniformLoss = -logProbs.mean(new long[] { -1 }); // [B*T]
            var maskedUniformLoss = (uniformLoss * mask).sum() / (mask.sum() + 1e-8f);
            loss = (1.0f - _labelSmoothing) * loss + _labelSmoothing * maskedUniformLoss;
        }

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
