namespace PaddleOcr.Training.Cls.Losses;

/// <summary>
/// Factory for building classification loss functions from configuration.
/// </summary>
public static class ClsLossBuilder
{
    /// <summary>
    /// Builds a classification loss function from name and configuration.
    /// </summary>
    /// <param name="name">Loss function name (e.g., "ClsLoss", "CrossEntropy")</param>
    /// <param name="config">Optional configuration dictionary (currently unused)</param>
    /// <returns>Configured loss function implementing IClsLoss</returns>
    public static IClsLoss BuildLoss(string name, Dictionary<string, object>? config = null)
    {
        return name.ToLowerInvariant() switch
        {
            "clsloss" or "cls" or "crossentropy" or "ce" => new ClsLoss(),
            _ => new ClsLoss() // Default to ClsLoss
        };
    }
}
