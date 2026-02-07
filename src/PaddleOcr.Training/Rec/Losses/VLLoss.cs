using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// VLLoss：VisionLAN 损失，包含长度和字符损失。
/// </summary>
public sealed class VLLoss : IRecLoss
{
    private readonly float _lengthWeight;

    public VLLoss(float lengthWeight = 0.5f)
    {
        _lengthWeight = lengthWeight;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var charLogits = predictions["predict"];
        var lengthLogits = predictions["length"];
        var charTargets = batch["label"].to(ScalarType.Int64);
        var lengthTargets = batch.ContainsKey("length") ? batch["length"].to(ScalarType.Int64) : null;

        var charLoss = functional.cross_entropy(charLogits.reshape(-1, charLogits.shape[^1]), charTargets.reshape(-1));
        var result = new Dictionary<string, Tensor> { ["char_loss"] = charLoss };
        var totalLoss = charLoss;

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
