namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// LinearWarmupCosine：线性预热 + 余弦退火学习率调度器。
/// </summary>
public sealed class LinearWarmupCosine : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _minLr;
    private readonly int _warmupEpochs;
    private readonly int _maxEpochs;
    private readonly int _warmupSteps;
    private readonly int _maxSteps;
    public double CurrentLR { get; private set; }

    public LinearWarmupCosine(float initialLr, float minLr, int warmupEpochs, int maxEpochs, int warmupSteps = 0, int maxSteps = 0)
    {
        _initialLr = initialLr;
        _minLr = minLr;
        _warmupEpochs = Math.Max(0, warmupEpochs);
        _maxEpochs = Math.Max(1, maxEpochs);
        _warmupSteps = Math.Max(0, warmupSteps);
        _maxSteps = Math.Max(0, maxSteps);
        CurrentLR = minLr;
    }

    public void Step(int step, int epoch)
    {
        if (_maxSteps > 0)
        {
            var clampedStep = Math.Max(1, Math.Min(step, _maxSteps));
            if (_warmupSteps > 0 && clampedStep <= _warmupSteps)
            {
                var warmupProgress = clampedStep / (float)_warmupSteps;
                CurrentLR = _minLr + (_initialLr - _minLr) * warmupProgress;
                return;
            }

            var denom = Math.Max(1, _maxSteps - _warmupSteps);
            var progress = Math.Clamp((clampedStep - _warmupSteps) / (float)denom, 0f, 1f);
            CurrentLR = _minLr + (_initialLr - _minLr) * (1.0f + Math.Cos(Math.PI * progress)) / 2.0f;
            return;
        }

        if (_warmupEpochs > 0 && epoch < _warmupEpochs)
        {
            CurrentLR = _minLr + (_initialLr - _minLr) * epoch / _warmupEpochs;
            return;
        }

        var epochDenom = Math.Max(1, _maxEpochs - _warmupEpochs);
        var epochProgress = Math.Clamp((epoch - _warmupEpochs) / (float)epochDenom, 0f, 1f);
        var lr = _minLr + (_initialLr - _minLr) * (1.0f + Math.Cos(Math.PI * epochProgress)) / 2.0f;
        CurrentLR = lr;
    }
}
