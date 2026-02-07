using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// SRNLoss：SRN 专用损失。
/// </summary>
public sealed class SRNLoss : IRecLoss
{
    private readonly float _weight;

    public SRNLoss(float weight = 1.0f)
    {
        _weight = weight;
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

        // CrossEntropy loss
        var loss = functional.cross_entropy(logits, targets);

        return new Dictionary<string, Tensor> { ["loss"] = loss * _weight };
    }
}
