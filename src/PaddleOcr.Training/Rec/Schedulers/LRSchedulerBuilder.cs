using PaddleOcr.Training.Rec.Schedulers;

namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// LRSchedulerBuilder：从配置构建学习率调度器。
/// </summary>
public static class LRSchedulerBuilder
{
    /// <summary>
    /// 构建学习率调度器。
    /// </summary>
    public static ILRScheduler Build(string name, Dictionary<string, object>? config = null)
    {
        config ??= new Dictionary<string, object>();
        return name.ToLowerInvariant() switch
        {
            "piecewise" or "piecewisedecay" => new PiecewiseDecay(
                milestones: GetFloatArray(config, "milestones", new[] { 10.0f, 20.0f }),
                values: GetFloatArray(config, "values", new[] { 0.1f, 0.01f })
            ),
            "cosine" or "cosineannealing" or "cosineannealingdecay" => new CosineAnnealingDecay(
                initialLr: GetFloat(config, "initial_lr", 0.001f),
                minLr: GetFloat(config, "min_lr", 0.0001f),
                maxEpochs: GetInt(config, "max_epochs", 100)
            ),
            "linearwarmupcosine" or "warmupcosine" => new LinearWarmupCosine(
                initialLr: GetFloat(config, "initial_lr", 0.001f),
                minLr: GetFloat(config, "min_lr", 0.0001f),
                warmupEpochs: GetInt(config, "warmup_epochs", 5),
                maxEpochs: GetInt(config, "max_epochs", 100)
            ),
            "polynomial" or "polynomialdecay" => new PolynomialDecay(
                initialLr: GetFloat(config, "initial_lr", 0.001f),
                endLr: GetFloat(config, "end_lr", 0.0001f),
                maxEpochs: GetInt(config, "max_epochs", 100),
                power: GetFloat(config, "power", 1.0f)
            ),
            "exponential" or "exponentialdecay" => new ExponentialDecay(
                initialLr: GetFloat(config, "initial_lr", 0.001f),
                gamma: GetFloat(config, "gamma", 0.95f)
            ),
            "noam" or "noamdecay" => new NoamDecay(
                dModel: GetInt(config, "d_model", 512),
                warmupSteps: GetInt(config, "warmup_steps", 4000),
                learningRate: GetFloat(config, "learning_rate", 0f)
            ),
            "cyclicalcosine" or "cyclicalcosinedecay" or "cycliccosine" => new CyclicalCosineDecay(
                initialLr: GetFloat(config, "initial_lr", 0.001f),
                minLr: GetFloat(config, "min_lr", 0.0001f),
                cycleLength: GetInt(config, "cycle_length", 50),
                decayFactor: GetFloat(config, "decay_factor", 0.5f)
            ),
            _ => new CosineAnnealingDecay(0.001f, 0.0001f, 100)
        };
    }

    private static int GetInt(Dictionary<string, object> config, string key, int defaultValue)
    {
        return config.ContainsKey(key) && config[key] is int value ? value : defaultValue;
    }

    private static float GetFloat(Dictionary<string, object> config, string key, float defaultValue)
    {
        return config.ContainsKey(key) && config[key] is float value ? value : defaultValue;
    }

    private static float[] GetFloatArray(Dictionary<string, object> config, string key, float[] defaultValue)
    {
        return config.ContainsKey(key) && config[key] is float[] value ? value : defaultValue;
    }
}
