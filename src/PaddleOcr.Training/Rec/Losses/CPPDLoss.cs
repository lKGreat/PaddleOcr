using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// CPPDLoss：CPPD 损失，类似 NRTR。
/// </summary>
public sealed class CPPDLoss : IRecLoss
{
    private readonly float _labelSmoothing;

    public CPPDLoss(float labelSmoothing = 0.0f)
    {
        _labelSmoothing = labelSmoothing;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"];
        var targets = batch["label"].to(ScalarType.Int64);
        var loss = functional.cross_entropy(logits.reshape(-1, logits.shape[^1]), targets.reshape(-1));
        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
