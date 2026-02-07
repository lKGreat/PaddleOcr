using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// RFLLoss：RFL 损失，包含文本和长度损失。
/// </summary>
public sealed class RFLLoss : IRecLoss
{
    private readonly float _lengthWeight;

    public RFLLoss(float lengthWeight = 0.5f)
    {
        _lengthWeight = lengthWeight;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var textLogits = predictions["predict"];
        var lengthLogits = predictions["length"];
        var textTargets = batch["label"].to(ScalarType.Int64);
        var lengthTargets = batch.ContainsKey("length") ? batch["length"].to(ScalarType.Int64) : null;

        var textLoss = functional.cross_entropy(textLogits.reshape(-1, textLogits.shape[^1]), textTargets.reshape(-1));
        var result = new Dictionary<string, Tensor> { ["text_loss"] = textLoss };
        var totalLoss = textLoss;

        if (batch.ContainsKey("length") && predictions.ContainsKey("length"))
        {
            var lenLoss = functional.cross_entropy(lengthLogits.reshape(-1, lengthLogits.shape[^1]), lengthTargets!.reshape(-1));
            result["length_loss"] = lenLoss;
            totalLoss = totalLoss + lenLoss * _lengthWeight;
        }

        result["loss"] = totalLoss;
        return result;
    }
}
