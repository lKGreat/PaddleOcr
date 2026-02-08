using PaddleOcr.Training.Det.Losses;

namespace PaddleOcr.Training.Det;

/// <summary>
/// Factory for building detection loss functions from configuration.
/// Follows the pattern established by RecModelBuilder and RecLossBuilder.
/// </summary>
public static class DetLossBuilder
{
    /// <summary>
    /// Builds a detection loss function from name and configuration.
    /// </summary>
    /// <param name="name">Loss function name (e.g., "DBLoss", "DB")</param>
    /// <param name="config">Optional configuration dictionary with loss parameters</param>
    /// <returns>Configured loss function implementing IDetLoss</returns>
    public static IDetLoss BuildLoss(string name, Dictionary<string, object>? config = null)
    {
        config ??= new Dictionary<string, object>();

        return name.ToLowerInvariant() switch
        {
            "dbloss" or "db" => BuildDBLoss(config),
            _ => BuildDBLoss(config) // Default to DBLoss
        };
    }

    /// <summary>
    /// Builds a DBLoss instance from configuration.
    /// </summary>
    /// <param name="config">Configuration dictionary with optional parameters:
    ///   - alpha (float): Weight for shrink map loss. Default: 5
    ///   - beta (float): Weight for threshold map loss. Default: 10
    ///   - balance_loss (bool): Whether to apply OHEM. Default: true
    ///   - ohem_ratio (float): Negative to positive sample ratio. Default: 3
    ///   - eps (float): Small epsilon to avoid division by zero. Default: 1e-6
    /// </param>
    /// <returns>Configured DBLoss instance</returns>
    private static DBLoss BuildDBLoss(Dictionary<string, object> config)
    {
        // Extract parameters with defaults
        var alpha = GetConfigValue<float>(config, "alpha", 5f);
        var beta = GetConfigValue<float>(config, "beta", 10f);
        var balanceLoss = GetConfigValue<bool>(config, "balance_loss", true);
        var ohemRatio = GetConfigValue<float>(config, "ohem_ratio", 3f);
        var eps = GetConfigValue<float>(config, "eps", 1e-6f);

        return new DBLoss(alpha, beta, balanceLoss, ohemRatio, eps);
    }

    /// <summary>
    /// Helper method to extract typed configuration values with defaults.
    /// </summary>
    private static T GetConfigValue<T>(Dictionary<string, object> config, string key, T defaultValue)
    {
        if (!config.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        // Handle type conversions
        try
        {
            if (value is T typedValue)
            {
                return typedValue;
            }

            // Try to convert
            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }
}
