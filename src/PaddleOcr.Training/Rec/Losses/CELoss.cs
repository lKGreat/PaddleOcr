using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// CELoss：通用 Cross-Entropy 损失函数（用于 NRTR/ViTSTR 等 attention 系列）。
/// 参考: ppocr/losses/rec_ce_loss.py
/// </summary>
public sealed class CELoss : IRecLoss
{
    private readonly float _labelSmoothing;
    private readonly int _ignoreIndex;

    public CELoss(float labelSmoothing = 0.0f, int ignoreIndex = 0)
    {
        _labelSmoothing = labelSmoothing;
        _ignoreIndex = ignoreIndex;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var predict = predictions.TryGetValue("predict", out var p)
            ? p
            : predictions.Values.First();
        var label = batch["label"].to(ScalarType.Int64);

        // [B, T, C] -> [B*T, C]
        if (predict.dim() == 3)
        {
            var b = predict.shape[0];
            var t = predict.shape[1];
            var c = predict.shape[2];
            predict = predict.reshape(b * t, c);
            label = label.reshape(-1);
        }

        var loss = functional.cross_entropy(predict, label,
            ignore_index: _ignoreIndex,
            label_smoothing: _labelSmoothing);

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
