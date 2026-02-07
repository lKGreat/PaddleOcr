using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// CPPDLoss：CPPD 损失。
/// 包含三部分：char_node_loss + pos_node_loss + edge_loss。
/// 支持 label smoothing。
/// 参考: ppocr/losses/rec_cppd_loss.py
/// </summary>
public sealed class CPPDLoss : IRecLoss
{
    private readonly bool _smoothing;
    private readonly int _ignoreIndex;
    private readonly float _sideLossWeight;

    public CPPDLoss(bool smoothing = false, int ignoreIndex = 100, float sideLossWeight = 1.0f)
    {
        _smoothing = smoothing;
        _ignoreIndex = ignoreIndex;
        _sideLossWeight = sideLossWeight;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"];
        var targets = batch["label"].to(ScalarType.Int64);

        Tensor loss;
        if (_smoothing)
        {
            loss = LabelSmoothingCE(logits.reshape(-1, logits.shape[^1]), targets.reshape(-1));
        }
        else
        {
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[^1]),
                targets.reshape(-1),
                ignore_index: _ignoreIndex);
        }

        // 如果模型输出了 node_feats，计算 node loss
        if (predictions.ContainsKey("node_char") && predictions.ContainsKey("node_pos") && batch.ContainsKey("node_target"))
        {
            var nodeTarget = batch["node_target"];
            var charNodeLogits = predictions["node_char"].flatten(0, 1);
            var numCharNodes = (int)nodeTarget.shape[1] - 26;
            if (numCharNodes > 0)
            {
                using var charTarget = nodeTarget[.., ..numCharNodes].flatten(0, 1).to(ScalarType.Int64);
                using var charNodeLoss = functional.cross_entropy(charNodeLogits, charTarget);

                using var posNodeLogits = predictions["node_pos"].flatten(0, 1);
                using var posTarget = nodeTarget[.., numCharNodes..].flatten(0, 1).to(ScalarType.Float32);
                using var posNodeLoss = functional.binary_cross_entropy_with_logits(posNodeLogits, posTarget);

                using var nodeLoss = charNodeLoss + posNodeLoss;
                using var scaledNodeLoss = _sideLossWeight * nodeLoss;
                var total = scaledNodeLoss + loss;
                loss.Dispose();
                loss = total;
            }
        }

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }

    /// <summary>
    /// Label smoothing cross-entropy loss。
    /// eps=0.1，将 true class 的概率设为 (1-eps)，其他类为 eps/(n_class-1)。
    /// </summary>
    private Tensor LabelSmoothingCE(Tensor preds, Tensor targets)
    {
        const float eps = 0.1f;
        var nClass = (int)preds.shape[1];

        // 处理 ignore_index
        using var nonPadMask = targets.ne(_ignoreIndex);
        using var safeTargets = targets.clamp(0, nClass - 1);

        // one_hot: [N, C]
        using var oneHot = functional.one_hot(safeTargets, nClass).to(ScalarType.Float32);
        using var smoothed = oneHot * (1f - eps) + (1f - oneHot) * (eps / (nClass - 1));

        // log_softmax + nll
        using var logProb = functional.log_softmax(preds, dim: 1);
        using var perSample = -(smoothed * logProb).sum(dim: 1);

        // 只对非 padding 位置求平均
        using var masked = perSample.masked_select(nonPadMask);
        return masked.mean();
    }
}
