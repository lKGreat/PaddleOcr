using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Cls.Heads;

/// <summary>
/// Classification head for text orientation detection.
/// Reference: PaddleOCR ppocr/modeling/heads/cls_head.py
/// </summary>
/// <remarks>
/// Architecture:
///   Input: [B, C, H, W] from backbone
///   ↓ AdaptiveAvgPool2d(1)
///   [B, C, 1, 1]
///   ↓ Flatten
///   [B, C]
///   ↓ Linear(C, num_classes)
///   [B, num_classes] logits
/// </remarks>
public sealed class ClsHead : Module<Tensor, Tensor>, IClsHead
{
    private readonly Module<Tensor, Tensor> _pool;
    private readonly Linear _fc;
    private readonly int _numClasses;

    /// <summary>
    /// Creates a new classification head.
    /// </summary>
    /// <param name="inChannels">Number of input channels from backbone</param>
    /// <param name="numClasses">Number of output classes (e.g., 2 for 0°/180°, 4 for 0°/90°/180°/270°)</param>
    public ClsHead(int inChannels, int numClasses) : base(nameof(ClsHead))
    {
        if (inChannels <= 0)
        {
            throw new ArgumentException("inChannels must be positive", nameof(inChannels));
        }
        if (numClasses <= 0)
        {
            throw new ArgumentException("numClasses must be positive", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Global average pooling to reduce spatial dimensions to 1x1
        _pool = AdaptiveAvgPool2d(1);

        // Linear layer for classification
        // Note: TorchSharp uses default initialization (Xavier uniform),
        // Python version uses uniform(-stdv, stdv) where stdv = 1/sqrt(in_channels)
        _fc = Linear(inChannels, numClasses);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through the classification head (implements IClsHead).
    /// </summary>
    /// <param name="input">Feature map from backbone. Shape: [B, C, H, W]</param>
    /// <returns>Class logits (training) or probabilities (inference). Shape: [B, num_classes]</returns>
    public Tensor Forward(Tensor input)
    {
        return forward(input);
    }

    /// <summary>
    /// Forward pass through the classification head (Module override).
    /// </summary>
    /// <param name="input">Feature map from backbone. Shape: [B, C, H, W]</param>
    /// <returns>Class logits (training) or probabilities (inference). Shape: [B, num_classes]</returns>
    public override Tensor forward(Tensor input)
    {
        // Apply global average pooling: [B, C, H, W] → [B, C, 1, 1]
        using var pooled = _pool.call(input);

        // Flatten: [B, C, 1, 1] → [B, C]
        using var flattened = pooled.reshape(input.shape[0], -1);

        // Linear projection: [B, C] → [B, num_classes]
        var logits = _fc.call(flattened);

        // Apply softmax during inference (not during training, as loss function handles it)
        if (!training)
        {
            using (logits)
            {
                return functional.softmax(logits, dim: 1);
            }
        }

        return logits;
    }

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;
}
