using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using PaddleOcr.Training.Cls.Backbones;
using PaddleOcr.Training.Cls.Heads;

namespace PaddleOcr.Training.Cls;

/// <summary>
/// Factory for building classification models from configuration.
/// Follows the pattern established by RecModelBuilder.
/// </summary>
public static class ClsModelBuilder
{
    /// <summary>
    /// Builds a classification backbone from name and configuration.
    /// </summary>
    /// <param name="name">Backbone name (e.g., "MobileNetV3", "MobileNetV3_large", "ResNet18")</param>
    /// <param name="inChannels">Number of input channels (default: 3 for RGB)</param>
    /// <param name="scale">Channel scaling factor (default: 1.0)</param>
    /// <returns>Tuple of (backbone module, output channels)</returns>
    public static (Module<Tensor, Tensor> Module, int OutChannels) BuildBackbone(
        string name,
        int inChannels = 3,
        float scale = 1.0f)
    {
        return name.ToLowerInvariant() switch
        {
            "mobilenetv3" or "mobilenet_v3" or "mv3" => BuildMobileNetV3(inChannels, "small", scale),
            "mobilenetv3_large" or "mobilenet_v3_large" or "mv3_large" => BuildMobileNetV3(inChannels, "large", scale),
            "mobilenetv3_small" or "mobilenet_v3_small" or "mv3_small" => BuildMobileNetV3(inChannels, "small", scale),
            _ => BuildMobileNetV3(inChannels, "small", scale) // Default
        };
    }

    /// <summary>
    /// Builds a classification head from name and configuration.
    /// </summary>
    /// <param name="name">Head name (e.g., "ClsHead")</param>
    /// <param name="inChannels">Number of input channels from backbone</param>
    /// <param name="numClasses">Number of output classes</param>
    /// <returns>Classification head module</returns>
    public static Module<Tensor, Tensor> BuildHead(
        string name,
        int inChannels,
        int numClasses)
    {
        return name.ToLowerInvariant() switch
        {
            "clshead" or "cls" => new ClsHead(inChannels, numClasses),
            _ => new ClsHead(inChannels, numClasses) // Default
        };
    }

    /// <summary>
    /// Builds a complete classification model (Backbone + Head).
    /// </summary>
    /// <param name="backboneName">Backbone architecture name</param>
    /// <param name="headName">Head architecture name</param>
    /// <param name="numClasses">Number of output classes</param>
    /// <param name="inChannels">Number of input channels (default: 3)</param>
    /// <param name="scale">Backbone channel scaling factor (default: 1.0)</param>
    /// <returns>Complete ClsModel</returns>
    public static ClsModel Build(
        string backboneName,
        string headName,
        int numClasses,
        int inChannels = 3,
        float scale = 1.0f)
    {
        var (backbone, backboneOutCh) = BuildBackbone(backboneName, inChannels, scale);
        var head = BuildHead(headName, backboneOutCh, numClasses);
        return new ClsModel(backbone, head, backboneName, headName);
    }

    /// <summary>
    /// Builds a complete classification model from configuration dictionary.
    /// </summary>
    /// <param name="config">Configuration dictionary with keys:
    ///   - Backbone.name (string): Backbone architecture
    ///   - Backbone.scale (float, optional): Channel scaling
    ///   - Head.name (string): Head architecture
    ///   - Head.num_classes or Head.class_dim (int): Number of classes
    /// </param>
    /// <param name="inChannels">Number of input channels (default: 3)</param>
    /// <returns>Complete ClsModel</returns>
    public static ClsModel BuildFromConfig(
        Dictionary<string, object> config,
        int inChannels = 3)
    {
        // Extract backbone config
        if (!config.TryGetValue("Backbone", out var backboneObj) || backboneObj is not Dictionary<string, object> backboneConfig)
        {
            throw new ArgumentException("config must contain 'Backbone' dictionary");
        }

        if (!backboneConfig.TryGetValue("name", out var backboneNameObj))
        {
            throw new ArgumentException("Backbone config must contain 'name'");
        }

        var backboneName = backboneNameObj.ToString()!;
        var scale = backboneConfig.TryGetValue("scale", out var scaleObj) ? Convert.ToSingle(scaleObj) : 1.0f;

        // Extract head config
        if (!config.TryGetValue("Head", out var headObj) || headObj is not Dictionary<string, object> headConfig)
        {
            throw new ArgumentException("config must contain 'Head' dictionary");
        }

        if (!headConfig.TryGetValue("name", out var headNameObj))
        {
            throw new ArgumentException("Head config must contain 'name'");
        }

        var headName = headNameObj.ToString()!;

        // Try both "num_classes" and "class_dim" (Python uses both naming conventions)
        var numClasses = headConfig.TryGetValue("num_classes", out var numClassesObj)
            ? Convert.ToInt32(numClassesObj)
            : headConfig.TryGetValue("class_dim", out var classDimObj)
                ? Convert.ToInt32(classDimObj)
                : throw new ArgumentException("Head config must contain 'num_classes' or 'class_dim'");

        return Build(backboneName, headName, numClasses, inChannels, scale);
    }

    /// <summary>
    /// Builds a MobileNetV3 backbone.
    /// </summary>
    private static (Module<Tensor, Tensor>, int) BuildMobileNetV3(int inChannels, string modelName, float scale)
    {
        var backbone = new ClsMobileNetV3(inChannels, modelName, scale);
        return (backbone, backbone.OutChannels);
    }
}
