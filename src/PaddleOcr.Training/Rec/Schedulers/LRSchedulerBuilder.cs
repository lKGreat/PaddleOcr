using System.Globalization;

namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// Build LR schedulers from YAML-like config dictionaries.
/// </summary>
public static class LRSchedulerBuilder
{
    public static ILRScheduler Build(string name, Dictionary<string, object>? config = null)
    {
        config ??= new Dictionary<string, object>();
        var initialLr = GetFloatAny(config, ["initial_lr", "learning_rate"], 0.001f);
        var maxEpochs = GetInt(config, "max_epochs", 100);
        var maxSteps = GetInt(config, "max_steps", 0);
        var warmupSteps = GetInt(config, "warmup_steps", 0);

        var warmupEpoch = GetIntAny(config, ["warmup_epoch", "warmup_epochs"], 0);
        var stepsPerEpoch = GetInt(config, "steps_per_epoch", 0);
        var warmupStepsTotal = warmupSteps > 0 ? warmupSteps : warmupEpoch * Math.Max(stepsPerEpoch, 1);

        ILRScheduler inner = name.ToLowerInvariant() switch
        {
            "piecewise" or "piecewisedecay" => new PiecewiseDecay(
                milestones: GetFloatArray(config, "milestones", [10f, 20f]),
                values: GetFloatArray(config, "values", [initialLr, initialLr * 0.1f, initialLr * 0.01f])),
            "cosine" or "cosineannealing" or "cosineannealingdecay" => new CosineAnnealingDecay(
                initialLr: initialLr,
                minLr: GetFloat(config, "min_lr", Math.Max(1e-7f, initialLr * 0.01f)),
                maxEpochs: maxEpochs,
                maxSteps: maxSteps),
            "linearwarmupcosine" or "warmupcosine" => new LinearWarmupCosine(
                initialLr: initialLr,
                minLr: GetFloat(config, "min_lr", Math.Max(1e-7f, initialLr * 0.01f)),
                warmupEpochs: GetIntAny(config, ["warmup_epochs", "warmup_epoch"], 5),
                maxEpochs: maxEpochs,
                warmupSteps: warmupSteps,
                maxSteps: maxSteps),
            "polynomial" or "polynomialdecay" or "linear" => new PolynomialDecay(
                initialLr: initialLr,
                endLr: GetFloat(config, "end_lr", Math.Max(1e-7f, initialLr * 0.01f)),
                maxEpochs: maxEpochs,
                power: GetFloat(config, "power", 1.0f),
                maxSteps: maxSteps),
            "exponential" or "exponentialdecay" => new ExponentialDecay(
                initialLr: initialLr,
                gamma: GetFloat(config, "gamma", 0.95f)),
            "noam" or "noamdecay" => new NoamDecay(
                dModel: GetInt(config, "d_model", 512),
                warmupSteps: GetInt(config, "warmup_steps", 4000),
                learningRate: GetFloatAny(config, ["learning_rate", "initial_lr"], 0f)),
            "cyclicalcosine" or "cyclicalcosinedecay" or "cycliccosine" => new CyclicalCosineDecay(
                initialLr: initialLr,
                minLr: GetFloat(config, "min_lr", Math.Max(1e-7f, initialLr * 0.01f)),
                cycleLength: GetInt(config, "cycle_length", 50),
                decayFactor: GetFloat(config, "decay_factor", 0.5f)),
            "step" or "stepdecay" => new StepDecay(
                initialLr: initialLr,
                stepSize: GetInt(config, "step_size", 10),
                gamma: GetFloat(config, "gamma", 0.1f)),
            "onecycle" or "onecycledecay" => new OneCycleDecay(
                maxLr: initialLr,
                minLr: GetFloat(config, "min_lr", 0.0f),
                totalSteps: maxSteps > 0 ? maxSteps : maxEpochs * Math.Max(stepsPerEpoch, 1),
                pctStart: GetFloat(config, "pct_start", 0.3f)),
            "multistep" or "multistepdecay" => new MultiStepDecay(
                initialLr: initialLr,
                milestones: GetFloatArray(config, "milestones", [30f, 60f]).Select(f => (int)f).ToArray(),
                gamma: GetFloat(config, "gamma", 0.1f)),
            "twostepcosine" or "twostepcosinedecay" => new TwoStepCosineDecay(
                initialLr: initialLr,
                midLr: GetFloat(config, "mid_lr", initialLr * 0.1f),
                minLr: GetFloat(config, "min_lr", Math.Max(1e-7f, initialLr * 0.01f)),
                switchEpoch: GetInt(config, "switch_epoch", maxEpochs / 2),
                maxEpochs: maxEpochs),
            "const" or "constant" => new ConstLR(initialLr),
            _ => new CosineAnnealingDecay(initialLr, Math.Max(1e-7f, initialLr * 0.01f), maxEpochs, maxSteps)
        };

        // Apply generic warmup wrapper if warmup is configured and not already handled
        if (warmupStepsTotal > 0 && inner is not LinearWarmupCosine && inner is not OneCycleDecay)
        {
            return new LinearWarmup(inner, warmupStepsTotal, startLr: 0.0, endLr: initialLr);
        }

        return inner;
    }

    private static int GetIntAny(Dictionary<string, object> config, IReadOnlyList<string> keys, int defaultValue)
    {
        foreach (var key in keys)
        {
            if (config.ContainsKey(key))
            {
                return GetInt(config, key, defaultValue);
            }
        }

        return defaultValue;
    }

    private static float GetFloatAny(Dictionary<string, object> config, IReadOnlyList<string> keys, float defaultValue)
    {
        foreach (var key in keys)
        {
            if (config.ContainsKey(key))
            {
                return GetFloat(config, key, defaultValue);
            }
        }

        return defaultValue;
    }

    private static int GetInt(Dictionary<string, object> config, string key, int defaultValue)
    {
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return defaultValue;
        }

        return raw switch
        {
            int i => i,
            long l => (int)l,
            float f => (int)f,
            double d => (int)d,
            decimal m => (int)m,
            _ => int.TryParse(raw.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) ? parsed : defaultValue
        };
    }

    private static float GetFloat(Dictionary<string, object> config, string key, float defaultValue)
    {
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return defaultValue;
        }

        return raw switch
        {
            float f => f,
            double d => (float)d,
            decimal m => (float)m,
            int i => i,
            long l => l,
            _ => float.TryParse(raw.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : defaultValue
        };
    }

    private static float[] GetFloatArray(Dictionary<string, object> config, string key, float[] defaultValue)
    {
        if (!config.TryGetValue(key, out var raw) || raw is null)
        {
            return defaultValue;
        }

        if (raw is float[] arr)
        {
            return arr;
        }

        if (raw is IList<object?> list)
        {
            var parsed = list
                .Select(x => x is null
                    ? float.NaN
                    : (float.TryParse(x.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : float.NaN))
                .Where(v => !float.IsNaN(v))
                .ToArray();
            return parsed.Length > 0 ? parsed : defaultValue;
        }

        return defaultValue;
    }
}
