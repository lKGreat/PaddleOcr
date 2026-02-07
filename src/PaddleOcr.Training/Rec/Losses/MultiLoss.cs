using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// MultiLoss：加权组合 CTC + SAR/NRTR。
/// </summary>
public sealed class MultiLoss : IRecLoss
{
    private readonly IRecLoss _ctcLoss;
    private readonly IRecLoss? _attnLoss;
    private readonly float _ctcWeight;
    private readonly float _attnWeight;

    public MultiLoss(float ctcWeight = 1.0f, float attnWeight = 1.0f, int ignoreIndex = 0)
    {
        _ctcWeight = ctcWeight;
        _attnWeight = attnWeight;
        _ctcLoss = new CTCLoss();
        if (attnWeight > 0.0f)
        {
            _attnLoss = new AttentionLoss(ignoreIndex);
        }
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var result = new Dictionary<string, Tensor>();

        // CTC loss
        var ctcPred = new Dictionary<string, Tensor> { ["predict"] = predictions["ctc"] };
        var ctcResult = _ctcLoss.Forward(ctcPred, batch);
        var ctcLoss = ctcResult["loss"];
        result["ctc_loss"] = ctcLoss;

        var totalLoss = ctcLoss * _ctcWeight;

        // Attention loss (if available)
        if (_attnLoss != null && predictions.ContainsKey("gtc"))
        {
            var attnPred = new Dictionary<string, Tensor> { ["predict"] = predictions["gtc"] };
            var attnResult = _attnLoss.Forward(attnPred, batch);
            var attnLoss = attnResult["loss"];
            result["attn_loss"] = attnLoss;
            totalLoss = totalLoss + attnLoss * _attnWeight;
        }

        result["loss"] = totalLoss;
        return result;
    }
}
