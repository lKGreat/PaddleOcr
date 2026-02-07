namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// ExponentialDecay：指数衰减学习率调度器。
/// lr = initial_lr * gamma^epoch
/// 参考: paddle.optimizer.lr.ExponentialDecay
/// </summary>
public sealed class ExponentialDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _gamma;

    public double CurrentLR { get; private set; }

    public ExponentialDecay(float initialLr = 0.001f, float gamma = 0.95f)
    {
        _initialLr = initialLr;
        _gamma = gamma;
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        CurrentLR = _initialLr * Math.Pow(_gamma, epoch);
    }
}
