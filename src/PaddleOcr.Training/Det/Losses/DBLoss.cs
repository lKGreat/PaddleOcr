using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Losses;

/// <summary>
/// Differentiable Binarization (DB) Loss for text detection.
/// Reference: PaddleOCR ppocr/losses/det_db_loss.py lines 29-99
/// </summary>
/// <remarks>
/// DBLoss combines three loss components:
/// 1. Shrink map loss (BalanceLoss with OHEM) - weighted by alpha
/// 2. Threshold map loss (MaskL1Loss) - weighted by beta
/// 3. Binary map loss (DiceLoss) - unweighted
///
/// Total loss = alpha * loss_shrink + beta * loss_threshold + loss_binary
///
/// Default weights: alpha=5, beta=10
/// </remarks>
public sealed class DBLoss : Module<Dictionary<string, Tensor>, Dictionary<string, Tensor>>, IDetLoss
{
    private readonly DiceLoss _diceLoss;
    private readonly MaskL1Loss _l1Loss;
    private readonly BalanceLoss _bceLoss;
    private readonly float _alpha;
    private readonly float _beta;

    /// <summary>
    /// Creates a new DB Loss instance.
    /// </summary>
    /// <param name="alpha">Weight for shrink map loss. Default: 5</param>
    /// <param name="beta">Weight for threshold map loss. Default: 10</param>
    /// <param name="balanceLoss">Whether to apply balance loss with OHEM. Default: true</param>
    /// <param name="ohemRatio">Ratio of negative to positive samples for OHEM. Default: 3</param>
    /// <param name="eps">Small epsilon value to avoid division by zero. Default: 1e-6</param>
    public DBLoss(
        float alpha = 5f,
        float beta = 10f,
        bool balanceLoss = true,
        float ohemRatio = 3f,
        float eps = 1e-6f) : base(nameof(DBLoss))
    {
        _alpha = alpha;
        _beta = beta;

        // Initialize component losses
        _diceLoss = new DiceLoss(eps);
        _l1Loss = new MaskL1Loss(eps);
        _bceLoss = new BalanceLoss(
            balanceLoss: balanceLoss,
            mainLossType: "BCELoss",
            negativeRatio: ohemRatio,
            eps: eps);

        RegisterComponents();
    }

    /// <summary>
    /// Computes the DB loss from predictions and ground truth labels.
    /// </summary>
    /// <param name="predictions">Dictionary containing "maps" tensor [B, 3, H, W] with:
    ///   - Channel 0: shrink map (probability map)
    ///   - Channel 1: threshold map
    ///   - Channel 2: binary map (differentiable binarization)
    /// </param>
    /// <param name="batch">Dictionary containing ground truth labels:
    ///   - "shrink_map": [B, H, W] - binary ground truth shrink map
    ///   - "shrink_mask": [B, H, W] - valid region mask for shrink map
    ///   - "threshold_map": [B, H, W] - ground truth threshold map
    ///   - "threshold_mask": [B, H, W] - valid region mask for threshold map
    /// </param>
    /// <returns>Dictionary with loss values:
    ///   - "loss": total weighted loss
    ///   - "loss_shrink_maps": weighted shrink map loss
    ///   - "loss_threshold_maps": weighted threshold map loss
    ///   - "loss_binary_maps": binary map loss
    /// </returns>
    public Dictionary<string, Tensor> Forward(
        Dictionary<string, Tensor> predictions,
        Dictionary<string, Tensor> batch)
    {
        // Extract prediction maps: [B, 3, H, W]
        if (!predictions.TryGetValue("maps", out var maps))
        {
            throw new ArgumentException("predictions must contain 'maps' key");
        }

        if (maps.shape.Length != 4 || maps.shape[1] != 3)
        {
            throw new ArgumentException($"maps must have shape [B, 3, H, W], got {string.Join(",", maps.shape)}");
        }

        // Split into individual maps
        using var shrinkMaps = maps.narrow(1, 0, 1).squeeze(1);      // [B, H, W]
        using var thresholdMaps = maps.narrow(1, 1, 1).squeeze(1);   // [B, H, W]
        using var binaryMaps = maps.narrow(1, 2, 1).squeeze(1);      // [B, H, W]

        // Extract ground truth labels
        if (!batch.TryGetValue("shrink_map", out var gtShrinkMap))
        {
            throw new ArgumentException("batch must contain 'shrink_map' key");
        }
        if (!batch.TryGetValue("shrink_mask", out var gtShrinkMask))
        {
            throw new ArgumentException("batch must contain 'shrink_mask' key");
        }
        if (!batch.TryGetValue("threshold_map", out var gtThreshMap))
        {
            throw new ArgumentException("batch must contain 'threshold_map' key");
        }
        if (!batch.TryGetValue("threshold_mask", out var gtThreshMask))
        {
            throw new ArgumentException("batch must contain 'threshold_mask' key");
        }

        // Compute individual losses
        using var lossShrink = _bceLoss.Forward(shrinkMaps, gtShrinkMap, gtShrinkMask);
        using var lossThresh = _l1Loss.Forward(thresholdMaps, gtThreshMap, gtThreshMask);
        using var lossBinary = _diceLoss.Forward(binaryMaps, gtShrinkMap, gtShrinkMask);

        // Apply weights
        using var lossShrinkWeighted = lossShrink * _alpha;
        using var lossThreshWeighted = lossThresh * _beta;

        // Compute total loss
        var lossAll = lossShrinkWeighted + lossThreshWeighted + lossBinary;

        return new Dictionary<string, Tensor>
        {
            ["loss"] = lossAll,
            ["loss_shrink_maps"] = lossShrinkWeighted.clone(),
            ["loss_threshold_maps"] = lossThreshWeighted.clone(),
            ["loss_binary_maps"] = lossBinary.clone()
        };
    }

    /// <summary>
    /// Not implemented - use Forward(predictions, batch) instead.
    /// </summary>
    public override Dictionary<string, Tensor> forward(Dictionary<string, Tensor> input)
    {
        throw new NotImplementedException("Use Forward(predictions, batch) instead");
    }
}
