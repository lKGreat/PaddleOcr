using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// LaTeXOCRLoss：LaTeXOCR 损失。
/// 使用 CrossEntropy + ignore_index=-100 处理 padding。
/// 标签跳过第一个 token（BOS）。
/// 参考: ppocr/losses/rec_latexocr_loss.py
/// </summary>
public sealed class LaTeXOCRLoss : IRecLoss
{
    private const int IgnoreIndex = -100;

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var wordProbs = predictions["predict"];
        var labels = batch["label"].to(ScalarType.Int64);

        // 标签跳过第一个 token（BOS），对齐序列
        Tensor targets;
        if (labels.shape[1] > 1)
        {
            targets = labels[.., 1..]; // 跳过 BOS
        }
        else
        {
            targets = labels;
        }

        // 将 padding 位置（值为 0）标记为 ignore_index
        using var paddingMask = targets.eq(0);
        using var maskedTargets = targets.masked_fill(paddingMask, IgnoreIndex);

        var loss = functional.cross_entropy(
            wordProbs.reshape(-1, wordProbs.shape[^1]),
            maskedTargets.reshape(-1),
            ignore_index: IgnoreIndex);

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
