using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// CTCLoss：CTC 损失函数，使用 TorchSharp 内置的 ctc_loss。
/// </summary>
public sealed class CTCLoss : IRecLoss
{
    private readonly int _blank;
    private readonly bool _reduction;

    public CTCLoss(int blank = 0, bool reduction = true)
    {
        _blank = blank;
        _reduction = reduction;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"]; // [B, T, C]
        var targets = batch.ContainsKey("label_ctc") ? batch["label_ctc"] : batch["label"]; // [B, L]
        var inputLengths = batch.ContainsKey("input_lengths") ? batch["input_lengths"] : null;
        var targetLengths = batch.ContainsKey("target_lengths") ? batch["target_lengths"] : null;

        // 转换 logits: [B, T, C] -> [T, B, C]
        logits = logits.permute(1, 0, 2);

        // 计算 input_lengths 和 target_lengths
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

        // 计算 CTC loss
        var loss = functional.ctc_loss(
            logits.log_softmax(2),
            targets.to(ScalarType.Int64),
            inputLengthsTensor,
            targetLengthsTensor,
            blank: _blank,
            reduction: _reduction ? Reduction.Mean : Reduction.None
        );

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }
}
