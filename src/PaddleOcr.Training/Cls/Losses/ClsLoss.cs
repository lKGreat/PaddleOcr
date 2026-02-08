using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Cls.Losses;

/// <summary>
/// Classification loss using cross-entropy.
/// Reference: PaddleOCR ppocr/losses/cls_loss.py
/// </summary>
public sealed class ClsLoss : Module<Tensor, Tensor>, IClsLoss
{
    /// <summary>
    /// Creates a new classification loss instance.
    /// </summary>
    public ClsLoss() : base(nameof(ClsLoss))
    {
    }

    /// <summary>
    /// Computes the classification loss.
    /// </summary>
    /// <param name="predictions">Predicted logits. Shape: [B, num_classes]</param>
    /// <param name="labels">Ground truth class labels. Shape: [B] with values in [0, num_classes)</param>
    /// <returns>Dictionary with "loss" key containing scalar loss value</returns>
    public Dictionary<string, Tensor> Forward(Tensor predictions, Tensor labels)
    {
        // Compute cross-entropy loss (expects logits, not probabilities)
        var loss = functional.cross_entropy(predictions, labels, reduction: Reduction.Mean);

        return new Dictionary<string, Tensor> { ["loss"] = loss };
    }

    /// <summary>
    /// Not implemented - use Forward(predictions, labels) instead.
    /// </summary>
    public override Tensor forward(Tensor input)
    {
        throw new NotImplementedException("Use Forward(predictions, labels) instead");
    }
}
