namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// CosineAnnealingDecay：余弦退火学习率调度器。
/// </summary>
public sealed class CosineAnnealingDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _minLr;
    private readonly int _maxEpochs;
    private readonly int _maxSteps;
    public double CurrentLR { get; private set; }

    public CosineAnnealingDecay(float initialLr, float minLr, int maxEpochs, int maxSteps = 0)
    {
        _initialLr = initialLr;
        _minLr = minLr;
        _maxEpochs = Math.Max(1, maxEpochs);
        _maxSteps = Math.Max(0, maxSteps);
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        var progress = _maxSteps > 0
            ? Math.Min(step / (float)_maxSteps, 1.0f)
            : Math.Min(epoch / (float)_maxEpochs, 1.0f);
        var lr = _minLr + (_initialLr - _minLr) * (1.0f + Math.Cos(Math.PI * progress)) / 2.0f;
        CurrentLR = lr;
    }
}
