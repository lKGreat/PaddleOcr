using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// Weighted composite loss for multi-head training (CTC + GTC branch).
/// 1:1 port of ppocr/losses/rec_multi_loss.py MultiLoss.
///
/// Python MultiLoss looks up predictions by loss type:
/// - CTCLoss uses predicts["ctc"]
/// - SARLoss uses predicts["sar"]
/// - NRTRLoss uses predicts["gtc"]
///
/// This C# version checks for "ctc", "gtc", and "sar" keys accordingly.
/// </summary>
public sealed class MultiLoss : IRecLoss
{
    private readonly IRecLoss _ctcLoss;
    private readonly IRecLoss? _gtcLoss;
    private readonly float _ctcWeight;
    private readonly float _gtcWeight;
    private readonly string _ctcLabelKey;
    private readonly string _gtcLabelKey;
    private readonly string _gtcPredKey; // "gtc" for NRTR, "sar" for SAR

    public MultiLoss(
        IRecLoss? ctcLoss = null,
        IRecLoss? gtcLoss = null,
        float ctcWeight = 1.0f,
        float gtcWeight = 1.0f,
        string ctcLabelKey = "label",
        string gtcLabelKey = "label",
        string gtcPredKey = "gtc")
    {
        _ctcLoss = ctcLoss ?? new CTCLoss();
        _gtcLoss = gtcLoss;
        _ctcWeight = ctcWeight;
        _gtcWeight = gtcWeight;
        _ctcLabelKey = ctcLabelKey;
        _gtcLabelKey = gtcLabelKey;
        _gtcPredKey = gtcPredKey;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var result = new Dictionary<string, Tensor>();

        // CTC loss: predicts["ctc"]
        var ctcPredTensor = predictions.TryGetValue("ctc", out var ctc) ? ctc : predictions["predict"];
        var ctcBatch = BuildBatchForLabel(batch, _ctcLabelKey);
        var ctcResult = _ctcLoss.Forward(new Dictionary<string, Tensor> { ["predict"] = ctcPredTensor }, ctcBatch);
        var ctcLoss = ctcResult["loss"];
        result["ctc_loss"] = ctcLoss;

        var totalLoss = ctcLoss * _ctcWeight;

        // GTC loss: predicts["gtc"] (NRTR) or predicts["sar"] (SAR)
        if (_gtcLoss is not null)
        {
            Tensor? gtcPred = null;
            if (predictions.TryGetValue(_gtcPredKey, out var pred))
            {
                gtcPred = pred;
            }
            else if (predictions.TryGetValue("gtc", out var gtcFallback))
            {
                gtcPred = gtcFallback;
            }
            else if (predictions.TryGetValue("sar", out var sarFallback))
            {
                gtcPred = sarFallback;
            }

            if (gtcPred is not null)
            {
                var gtcBatch = BuildBatchForLabel(batch, _gtcLabelKey);
                var gtcResult = _gtcLoss.Forward(new Dictionary<string, Tensor> { ["predict"] = gtcPred }, gtcBatch);
                var gtcLoss = gtcResult["loss"];
                result["gtc_loss"] = gtcLoss;
                totalLoss += gtcLoss * _gtcWeight;
            }
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
