namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// PolynomialDecay：多项式衰减学习率调度器。
/// </summary>
public sealed class PolynomialDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _endLr;
    private readonly int _maxEpochs;
    private readonly float _power;
    public double CurrentLR { get; private set; }

    public PolynomialDecay(float initialLr, float endLr, int maxEpochs, float power = 1.0f)
    {
        _initialLr = initialLr;
        _endLr = endLr;
        _maxEpochs = maxEpochs;
        _power = power;
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        var progress = Math.Min(epoch / (float)_maxEpochs, 1.0f);
        var lr = _endLr + (_initialLr - _endLr) * Math.Pow(1.0 - progress, _power);
        CurrentLR = lr;
    }
}
