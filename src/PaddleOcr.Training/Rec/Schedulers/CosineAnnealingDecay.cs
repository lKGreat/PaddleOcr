namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// CosineAnnealingDecay：余弦退火学习率调度器。
/// </summary>
public sealed class CosineAnnealingDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _minLr;
    private readonly int _maxEpochs;
    public double CurrentLR { get; private set; }

    public CosineAnnealingDecay(float initialLr, float minLr, int maxEpochs)
    {
        _initialLr = initialLr;
        _minLr = minLr;
        _maxEpochs = maxEpochs;
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        var progress = Math.Min(epoch / (float)_maxEpochs, 1.0f);
        var lr = _minLr + (_initialLr - _minLr) * (1.0f + Math.Cos(Math.PI * progress)) / 2.0f;
        CurrentLR = lr;
    }
}
