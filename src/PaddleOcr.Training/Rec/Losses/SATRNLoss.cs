using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// SATRNLoss：SATRN 损失，类似 SAR。
/// </summary>
public sealed class SATRNLoss : IRecLoss
{
    private readonly int _ignoreIndex;

    public SATRNLoss(int ignoreIndex = 0)
    {
        _ignoreIndex = ignoreIndex;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"];
        var targets = batch["label"].to(ScalarType.Int64);
        var loss = functional.cross_entropy(logits.reshape(-1, logits.shape[^1]), targets.reshape(-1), ignore_index: _ignoreIndex);
        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
