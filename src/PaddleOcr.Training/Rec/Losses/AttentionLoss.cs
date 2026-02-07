using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// AttentionLoss：用于 attention 解码的 CrossEntropy 损失。
/// </summary>
public sealed class AttentionLoss : IRecLoss
{
    private readonly int _ignoreIndex;

    public AttentionLoss(int ignoreIndex = 0)
    {
        _ignoreIndex = ignoreIndex;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"]; // [B, T, C]
        var targets = batch.ContainsKey("label_gtc") ? batch["label_gtc"] : batch["label"]; // [B, T]

        // Reshape: [B, T, C] -> [B*T, C]
        var b = logits.shape[0];
        var t = logits.shape[1];
        var c = logits.shape[2];
        logits = logits.reshape(b * t, c);
        targets = targets.reshape(b * t).to(ScalarType.Int64);

        // CrossEntropy loss
        var loss = functional.cross_entropy(logits, targets, ignore_index: _ignoreIndex);

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
