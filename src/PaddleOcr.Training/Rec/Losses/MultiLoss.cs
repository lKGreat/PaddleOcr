using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// Weighted composite loss for multi-head training (CTC + GTC branch).
/// </summary>
public sealed class MultiLoss : IRecLoss
{
    private readonly IRecLoss _ctcLoss;
    private readonly IRecLoss? _gtcLoss;
    private readonly float _ctcWeight;
    private readonly float _gtcWeight;
    private readonly string _ctcLabelKey;
    private readonly string _gtcLabelKey;

    public MultiLoss(
        IRecLoss? ctcLoss = null,
        IRecLoss? gtcLoss = null,
        float ctcWeight = 1.0f,
        float gtcWeight = 1.0f,
        string ctcLabelKey = "label",
        string gtcLabelKey = "label")
    {
        _ctcLoss = ctcLoss ?? new CTCLoss();
        _gtcLoss = gtcLoss;
        _ctcWeight = ctcWeight;
        _gtcWeight = gtcWeight;
        _ctcLabelKey = ctcLabelKey;
        _gtcLabelKey = gtcLabelKey;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var result = new Dictionary<string, Tensor>();

        var ctcPredTensor = predictions.TryGetValue("ctc", out var ctc) ? ctc : predictions["predict"];
        var ctcBatch = BuildBatchForLabel(batch, _ctcLabelKey);
        var ctcResult = _ctcLoss.Forward(new Dictionary<string, Tensor> { ["predict"] = ctcPredTensor }, ctcBatch);
        var ctcLoss = ctcResult["loss"];
        result["ctc_loss"] = ctcLoss;

        var totalLoss = ctcLoss * _ctcWeight;

        if (_gtcLoss is not null && predictions.TryGetValue("gtc", out var gtcPred))
        {
            var gtcBatch = BuildBatchForLabel(batch, _gtcLabelKey);
            var gtcResult = _gtcLoss.Forward(new Dictionary<string, Tensor> { ["predict"] = gtcPred }, gtcBatch);
            var gtcLoss = gtcResult["loss"];
            result["gtc_loss"] = gtcLoss;
            totalLoss += gtcLoss * _gtcWeight;
        }

        result["loss"] = totalLoss;
        return result;
    }

    private static Dictionary<string, Tensor> BuildBatchForLabel(Dictionary<string, Tensor> batch, string labelKey)
    {
        var result = new Dictionary<string, Tensor>(batch);
        if (batch.TryGetValue(labelKey, out var label))
        {
            result["label"] = label;
        }

        return result;
    }
}
