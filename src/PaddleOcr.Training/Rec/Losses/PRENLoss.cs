using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// PRENLoss：PREN 损失。
/// </summary>
public sealed class PRENLoss : IRecLoss
{
    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"];
        var targets = batch["label"].to(ScalarType.Int64);
        var loss = functional.cross_entropy(logits.reshape(-1, logits.shape[^1]), targets.reshape(-1));
        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
