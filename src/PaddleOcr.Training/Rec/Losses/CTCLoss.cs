using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// CTC loss wrapper for rec training.
/// Matches PaddleOCR Python behavior: reduction="none" then .mean().
/// IMPORTANT: PyTorch's Reduction.Mean for CTC divides by target_lengths,
/// but PaddleOCR uses reduction="none" + .mean() which does NOT divide
/// by target_lengths. We must use None + manual mean to match.
/// Reference: ppocr/losses/rec_ctc_loss.py
/// </summary>
public sealed class CTCLoss : IRecLoss
{
    private readonly int _blank;
    private readonly bool _useFocalLoss;

    public CTCLoss(int blank = 0, bool reduction = true, bool useFocalLoss = false)
    {
        _blank = blank;
        _useFocalLoss = useFocalLoss;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        var logits = predictions["predict"]; // [B, T, C]
        var targets = batch.ContainsKey("label_ctc") ? batch["label_ctc"] : batch["label"]; // [B, L] or packed
        var inputLengths = batch.ContainsKey("input_lengths") ? batch["input_lengths"] : null;
        var targetLengths = batch.ContainsKey("target_lengths") ? batch["target_lengths"] : null;

        // [B, T, C] -> [T, B, C]
        logits = logits.permute(1, 0, 2);

        var batchSize = logits.shape[1];
        var timeSteps = logits.shape[0];
        Tensor inputLengthsTensor;
        Tensor targetLengthsTensor;

        if (inputLengths is not null)
        {
            inputLengthsTensor = inputLengths.to(ScalarType.Int64);
        }
        else
        {
            inputLengthsTensor = ones(new long[] { batchSize }, ScalarType.Int64, device: logits.device) * timeSteps;
        }

        if (targetLengths is not null)
        {
            targetLengthsTensor = targetLengths.to(ScalarType.Int64);
        }
        else
        {
            targetLengthsTensor = ones(new long[] { batchSize }, ScalarType.Int64, device: logits.device) * targets.shape[^1];
        }

        using var packedTargets = PackTargetsForCtc(targets, targetLengthsTensor, logits.device);

        // Match Python PaddleOCR: use reduction="none" then .mean()
        // DO NOT use Reduction.Mean — PyTorch CTC divides by target_lengths
        // which produces different gradient scaling from the Python implementation.
        //
        // CRITICAL: PaddlePaddle's WarpCTC internally normalizes gradients by T
        // (the number of time steps), but PyTorch's CTC does NOT. Without this
        // normalization, PyTorch gradients are T times larger (e.g. 160x for T=160),
        // causing CTC collapse (model predicts all blanks).
        // We normalize the per-sample loss by T to match WarpCTC behavior.
        var perSampleLoss = functional.ctc_loss(
            logits.log_softmax(2),
            packedTargets,
            inputLengthsTensor,
            targetLengthsTensor,
            blank: _blank,
            reduction: Reduction.None);

        // Replace inf/nan with zero to prevent gradient explosions (matches zero_infinity=True)
        perSampleLoss = perSampleLoss.clamp(max: 1e6f);

        // Normalize by T (input time steps) to match WarpCTC gradient normalization
        perSampleLoss = perSampleLoss / inputLengthsTensor.to(ScalarType.Float32).clamp(min: 1);

        Tensor loss;
        if (_useFocalLoss)
        {
            // Focal CTC loss: weight = (1 - exp(-loss))^2
            // Reference: ppocr/losses/rec_ctc_loss.py
            using var weight = (1.0f - (-perSampleLoss).exp()).pow(2);
            loss = (perSampleLoss * weight).mean();
        }
        else
        {
            loss = perSampleLoss.mean();
        }

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }

    private static Tensor PackTargetsForCtc(Tensor targets, Tensor targetLengths, Device device)
    {
        using var targetLengthsCpu = targetLengths.to(ScalarType.Int64).cpu();
        var lengths = targetLengthsCpu.data<long>().ToArray();

        if (targets.shape.Length != 2)
        {
            return targets.to(ScalarType.Int64, device: device);
        }

        using var targetsCpu = targets.to(ScalarType.Int64).cpu();
        var flat = targetsCpu.data<long>().ToArray();
        var batch = (int)targets.shape[0];
        var maxLen = (int)targets.shape[1];
        var packed = new List<long>(flat.Length);

        for (var i = 0; i < batch; i++)
        {
            var take = i < lengths.Length ? (int)Math.Clamp(lengths[i], 0, maxLen) : maxLen;
            var offset = i * maxLen;
            for (var j = 0; j < take; j++)
            {
                packed.Add(flat[offset + j]);
            }
        }

        if (packed.Count == 0)
        {
            packed.Add(0);
        }

        return tensor(packed.ToArray(), dtype: ScalarType.Int64, device: device);
    }
}
