using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Losses;

/// <summary>
/// Dice Loss for Differentiable Binarization (DB) text detection.
/// Reference: PaddleOCR ppocr/losses/det_basic_loss.py lines 117-137
/// </summary>
/// <remarks>
/// Dice Loss computes the overlap between prediction and ground truth:
/// loss = 1 - 2 * intersection / union
/// where:
///   intersection = sum(pred * gt * mask)
///   union = sum(pred * mask) + sum(gt * mask) + eps
/// </remarks>
public sealed class DiceLoss : Module<Tensor, Tensor>
{
    private readonly float _eps;

    /// <summary>
    /// Creates a new Dice Loss instance.
    /// </summary>
    /// <param name="eps">Small epsilon value to avoid division by zero. Default: 1e-6</param>
    public DiceLoss(float eps = 1e-6f) : base(nameof(DiceLoss))
    {
        _eps = eps;
    }

    /// <summary>
    /// Computes the Dice loss between prediction and ground truth.
    /// </summary>
    /// <param name="pred">Predicted probability map. Shape: [B, H, W] or [B, 1, H, W]</param>
    /// <param name="gt">Ground truth binary map. Shape: [B, H, W] or [B, 1, H, W]</param>
    /// <param name="mask">Valid region mask. Shape: [B, H, W] or [B, 1, H, W]</param>
    /// <param name="weights">Optional per-pixel weights. Shape: same as mask</param>
    /// <returns>Scalar loss value</returns>
    public Tensor Forward(Tensor pred, Tensor gt, Tensor mask, Tensor? weights = null)
    {
        // Validate shapes
        if (!pred.shape.SequenceEqual(gt.shape))
        {
            throw new ArgumentException($"pred shape {string.Join(",", pred.shape)} != gt shape {string.Join(",", gt.shape)}");
        }
        if (!pred.shape.SequenceEqual(mask.shape))
        {
            throw new ArgumentException($"pred shape {string.Join(",", pred.shape)} != mask shape {string.Join(",", mask.shape)}");
        }

        // Apply optional weights to mask
        var effectiveMask = mask;
        if (weights is not null)
        {
            if (!weights.shape.SequenceEqual(mask.shape))
            {
                throw new ArgumentException($"weights shape {string.Join(",", weights.shape)} != mask shape {string.Join(",", mask.shape)}");
            }
            effectiveMask = weights * mask;
        }

        // Compute Dice loss
        using var intersection = (pred * gt * effectiveMask).sum();
        using var unionPart1 = (pred * effectiveMask).sum();
        using var unionPart2 = (gt * effectiveMask).sum();
        using var union = unionPart1 + unionPart2 + _eps;

        var loss = 1.0f - 2.0f * intersection / union;

        return loss;
    }

    /// <summary>
    /// Not implemented - use Forward(pred, gt, mask, weights) instead.
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        throw new NotImplementedException("Use Forward(pred, gt, mask, weights?) instead");
    }
}
