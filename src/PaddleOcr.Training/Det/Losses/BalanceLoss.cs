using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Losses;

/// <summary>
/// Balance Loss with Online Hard Example Mining (OHEM) for text detection.
/// Reference: PaddleOCR ppocr/losses/det_basic_loss.py lines 29-114
/// </summary>
/// <remarks>
/// BalanceLoss addresses class imbalance in text detection by:
/// 1. Computing element-wise loss (BCE, Dice, etc.)
/// 2. Separating positive and negative samples
/// 3. Selecting hardest negative samples (OHEM)
/// 4. Balancing loss = (positive_loss + hard_negative_loss) / (positive_count + negative_count)
/// </remarks>
public sealed class BalanceLoss : Module<Tensor, Tensor>
{
    private readonly bool _balanceLoss;
    private readonly float _negativeRatio;
    private readonly float _eps;
    private readonly string _mainLossType;

    /// <summary>
    /// Creates a new Balance Loss instance.
    /// </summary>
    /// <param name="balanceLoss">Whether to apply balance loss or return raw loss. Default: true</param>
    /// <param name="mainLossType">Type of base loss ("DiceLoss", "BCELoss", "CrossEntropy", etc.). Default: "DiceLoss"</param>
    /// <param name="negativeRatio">Ratio of negative to positive samples for OHEM. Default: 3</param>
    /// <param name="eps">Small epsilon value to avoid division by zero. Default: 1e-6</param>
    public BalanceLoss(
        bool balanceLoss = true,
        string mainLossType = "DiceLoss",
        float negativeRatio = 3f,
        float eps = 1e-6f) : base(nameof(BalanceLoss))
    {
        _balanceLoss = balanceLoss;
        _mainLossType = mainLossType;
        _negativeRatio = negativeRatio;
        _eps = eps;
    }

    /// <summary>
    /// Computes the balanced loss with OHEM.
    /// </summary>
    /// <param name="pred">Predicted shrink map (logits). Shape: [B, H, W]</param>
    /// <param name="gt">Ground truth shrink map (binary). Shape: [B, H, W]</param>
    /// <param name="mask">Valid region mask. Shape: [B, H, W]</param>
    /// <returns>Scalar balanced loss value</returns>
    public Tensor Forward(Tensor pred, Tensor gt, Tensor mask)
    {
        // Separate positive and negative samples
        using var positive = gt * mask;      // [B, H, W] - regions with text
        using var negative = (1 - gt) * mask; // [B, H, W] - regions without text

        // Count positive and negative pixels
        var positiveCount = positive.sum().ToInt32();
        var negativeCountTotal = negative.sum().ToInt32();
        var negativeCount = Math.Min(negativeCountTotal, (int)(positiveCount * _negativeRatio));

        // Compute element-wise loss based on loss type
        Tensor loss;
        if (_mainLossType.Equals("BCELoss", StringComparison.OrdinalIgnoreCase))
        {
            // Binary cross-entropy with logits (no reduction)
            loss = functional.binary_cross_entropy_with_logits(pred, gt, reduction: Reduction.None);
        }
        else if (_mainLossType.Equals("DiceLoss", StringComparison.OrdinalIgnoreCase))
        {
            // For DiceLoss in BalanceLoss context, we use element-wise BCE as proxy
            // (Python implementation wraps DiceLoss but applies it globally, not per-pixel)
            loss = functional.binary_cross_entropy_with_logits(pred, gt, reduction: Reduction.None);
        }
        else
        {
            throw new NotSupportedException($"Loss type '{_mainLossType}' not supported. Use 'BCELoss' or 'DiceLoss'.");
        }

        // If balance_loss is false, return mean loss
        if (!_balanceLoss)
        {
            using (loss)
            {
                return (loss * mask).mean();
            }
        }

        // Apply OHEM (Online Hard Example Mining)
        using var positiveLoss = positive * loss;
        using var negativeLoss = negative * loss;

        // Compute positive loss sum
        using var positiveLossSum = positiveLoss.sum();

        // Select hardest negative samples
        Tensor negativeLossSum;
        if (negativeCount > 0)
        {
            // Flatten negative loss and sort in descending order
            using var negativeLossFlat = negativeLoss.reshape(-1);
            var sortedResult = negativeLossFlat.sort(descending: true);
            using var sortedValues = sortedResult.Values;

            // Select top-k hardest negatives
            using var hardNegativeLoss = sortedValues.narrow(0, 0, negativeCount);
            negativeLossSum = hardNegativeLoss.sum();
        }
        else
        {
            negativeLossSum = torch.tensor(0f, device: pred.device);
        }

        // Compute balanced loss
        using (negativeLossSum)
        {
            var balancedLoss = (positiveLossSum + negativeLossSum) / (positiveCount + negativeCount + _eps);
            return balancedLoss;
        }
    }

    /// <summary>
    /// Not implemented - use Forward(pred, gt, mask) instead.
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        throw new NotImplementedException("Use Forward(pred, gt, mask) instead");
    }
}
