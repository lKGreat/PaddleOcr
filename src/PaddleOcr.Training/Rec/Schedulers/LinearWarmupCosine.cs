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
    public double CurrentLR { get; private set; }

    public LinearWarmupCosine(float initialLr, float minLr, int warmupEpochs, int maxEpochs)
    {
        _initialLr = initialLr;
        _minLr = minLr;
        _warmupEpochs = warmupEpochs;
        _maxEpochs = maxEpochs;
        CurrentLR = minLr;
    }

    public void Step(int step, int epoch)
    {
        if (epoch < _warmupEpochs)
        {
            // Linear warmup
            CurrentLR = _minLr + (_initialLr - _minLr) * epoch / _warmupEpochs;
        }
        else
        {
            // Cosine annealing
            var progress = (epoch - _warmupEpochs) / (float)(_maxEpochs - _warmupEpochs);
            var lr = _minLr + (_initialLr - _minLr) * (1.0f + Math.Cos(Math.PI * progress)) / 2.0f;
            CurrentLR = lr;
        }
    }
}
