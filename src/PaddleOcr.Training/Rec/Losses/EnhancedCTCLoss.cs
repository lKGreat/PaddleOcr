using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// EnhancedCTCLoss：增强的 CTC 损失，支持更多选项。
/// </summary>
public sealed class EnhancedCTCLoss : IRecLoss
{
    private readonly int _blank;
    private readonly bool _reduction;
    private readonly float _zeroInfinity;

    public EnhancedCTCLoss(int blank = 0, bool reduction = true, float zeroInfinity = 0.0f)
    {
        _blank = blank;
        _reduction = reduction;
        _zeroInfinity = zeroInfinity;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"]; // [B, T, C]
        var targets = batch["label"]; // [B, L]
        var inputLengths = batch.ContainsKey("input_lengths") ? batch["input_lengths"] : null;
        var targetLengths = batch.ContainsKey("target_lengths") ? batch["target_lengths"] : null;

        logits = logits.permute(1, 0, 2);

        var b = logits.shape[1];
        var t = logits.shape[0];
        Tensor inputLengthsTensor;
        Tensor targetLengthsTensor;
        
        if (inputLengths is not null)
        {
            inputLengthsTensor = inputLengths.to(ScalarType.Int64);
        }
        else
        {
            inputLengthsTensor = ones(new long[] { b }, ScalarType.Int64, device: logits.device) * t;
        }

        if (targetLengths is not null)
        {
            targetLengthsTensor = targetLengths.to(ScalarType.Int64);
        }
        else
        {
            targetLengthsTensor = ones(new long[] { b }, ScalarType.Int64, device: logits.device) * targets.shape[1];
        }

        var loss = functional.ctc_loss(
            logits.log_softmax(2),
            targets.to(ScalarType.Int64),
            inputLengthsTensor,
            targetLengthsTensor,
            blank: _blank,
            reduction: _reduction ? Reduction.Mean : Reduction.None,
            zero_infinity: _zeroInfinity > 0.0f
        );

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
