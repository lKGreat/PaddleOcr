namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// StepDecay：每隔 stepSize 步将学习率乘以 gamma。
/// 参考: ppocr/optimizer/learning_rate.py - Step
/// </summary>
public sealed class StepDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly int _stepSize;
    private readonly float _gamma;
    public double CurrentLR { get; private set; }

    public StepDecay(float initialLr, int stepSize = 10, float gamma = 0.1f)
    {
        _initialLr = initialLr;
        _stepSize = Math.Max(1, stepSize);
        _gamma = gamma;
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        var numDecays = epoch / _stepSize;
        CurrentLR = _initialLr * Math.Pow(_gamma, numDecays);
    }
}
