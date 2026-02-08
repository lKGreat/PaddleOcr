using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Losses;

/// <summary>
/// Masked L1 Loss for Differentiable Binarization (DB) text detection.
/// Reference: PaddleOCR ppocr/losses/det_basic_loss.py lines 140-151
/// </summary>
/// <remarks>
/// Computes L1 distance between prediction and ground truth, weighted by mask:
/// loss = mean(sum(|pred - gt| * mask) / (sum(mask) + eps))
/// </remarks>
public sealed class MaskL1Loss : Module<Tensor, Tensor>
{
    private readonly float _eps;

    /// <summary>
    /// Creates a new Masked L1 Loss instance.
    /// </summary>
    /// <param name="eps">Small epsilon value to avoid division by zero. Default: 1e-6</param>
    public MaskL1Loss(float eps = 1e-6f) : base(nameof(MaskL1Loss))
    {
        _eps = eps;
    }

    /// <summary>
    /// Computes the masked L1 loss between prediction and ground truth.
    /// </summary>
    /// <param name="pred">Predicted threshold map. Shape: [B, H, W] or [B, 1, H, W]</param>
    /// <param name="gt">Ground truth threshold map. Shape: [B, H, W] or [B, 1, H, W]</param>
    /// <param name="mask">Valid region mask. Shape: [B, H, W] or [B, 1, H, W]</param>
    /// <returns>Scalar loss value</returns>
    public Tensor Forward(Tensor pred, Tensor gt, Tensor mask)
    {
        // Compute absolute difference
        using var diff = torch.abs(pred - gt);

        // Apply mask and compute sum
        using var maskedDiff = diff * mask;
        using var numerator = maskedDiff.sum();

        // Compute denominator (sum of mask + eps)
        using var denominator = mask.sum() + _eps;

        // Compute mean loss
        var loss = numerator / denominator;
        var meanLoss = loss.mean();

        return meanLoss;
    }

    /// <summary>
    /// Not implemented - use Forward(pred, gt, mask) instead.
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        throw new NotImplementedException("Use Forward(pred, gt, mask) instead");
    }
}
