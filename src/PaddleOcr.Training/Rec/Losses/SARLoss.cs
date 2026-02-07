using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// SARLoss：SAR 损失，CrossEntropy + ignore_index。
/// </summary>
public sealed class SARLoss : IRecLoss
{
    private readonly int _ignoreIndex;

    public SARLoss(int ignoreIndex = 0)
    {
        _ignoreIndex = ignoreIndex;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"]; // [B, T, C]
        var targets = batch["label"]; // [B, T]

        // Reshape: [B, T, C] -> [B*T, C]
        var b = logits.shape[0];
        var t = logits.shape[1];
        var c = logits.shape[2];
        logits = logits.reshape(b * t, c);
        targets = targets.reshape(b * t).to(ScalarType.Int64);

        // CrossEntropy loss with ignore_index
        var loss = functional.cross_entropy(logits, targets, ignore_index: _ignoreIndex);

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
