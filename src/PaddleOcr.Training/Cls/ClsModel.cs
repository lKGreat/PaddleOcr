using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Cls;

/// <summary>
/// Complete classification model orchestrating Backbone + Head.
/// Follows the pattern established by RecModel.
/// </summary>
public sealed class ClsModel : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _head;

    /// <summary>
    /// Gets the name of the backbone architecture.
    /// </summary>
    public string BackboneName { get; }

    /// <summary>
    /// Gets the name of the head architecture.
    /// </summary>
    public string HeadName { get; }

    /// <summary>
    /// Creates a new classification model.
    /// </summary>
    /// <param name="backbone">Backbone module for feature extraction</param>
    /// <param name="head">Classification head module</param>
    /// <param name="backboneName">Name of the backbone (e.g., "MobileNetV3")</param>
    /// <param name="headName">Name of the head (e.g., "ClsHead")</param>
    public ClsModel(
        Module<Tensor, Tensor> backbone,
        Module<Tensor, Tensor> head,
        string backboneName,
        string headName) : base(nameof(ClsModel))
    {
        _backbone = backbone ?? throw new ArgumentNullException(nameof(backbone));
        _head = head ?? throw new ArgumentNullException(nameof(head));
        BackboneName = backboneName ?? throw new ArgumentNullException(nameof(backboneName));
        HeadName = headName ?? throw new ArgumentNullException(nameof(headName));

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass through the complete model.
    /// </summary>
    /// <param name="input">Input image tensor. Shape: [B, 3, H, W]</param>
    /// <returns>Class logits or probabilities. Shape: [B, num_classes]</returns>
    public override Tensor forward(Tensor input)
    {
        // Extract features from backbone
        using var features = _backbone.call(input);

        // Apply classification head
        return _head.call(features);
    }

    /// <summary>
    /// Forward pass returning results in a dictionary (for compatibility with training pipeline).
    /// </summary>
    /// <param name="input">Input image tensor. Shape: [B, 3, H, W]</param>
    /// <returns>Dictionary with "predict" key containing logits/probabilities</returns>
    public Dictionary<string, Tensor> ForwardDict(Tensor input)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }

    /// <summary>
    /// Gets a summary string describing the model architecture.
    /// </summary>
    public string GetArchitectureSummary()
    {
        return $"ClsModel(Backbone={BackboneName}, Head={HeadName})";
    }
}
