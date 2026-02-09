namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// 通用 LinearWarmup 包装器：可包装任意 ILRScheduler 实现 warmup。
/// warmup 期间线性从 startLR 上升到内部调度器的 LR，warmup 结束后委托给内部调度器。
/// 参考: ppocr/optimizer/learning_rate.py - LinearWarmup
/// </summary>
public sealed class LinearWarmup : ILRScheduler
{
    private readonly ILRScheduler _inner;
    private readonly int _warmupSteps;
    private readonly double _startLr;
    private readonly double _endLr;

    public double CurrentLR { get; private set; }

    /// <summary>
    /// 创建 LinearWarmup 包装器。
    /// </summary>
    /// <param name="inner">被包装的内部调度器</param>
    /// <param name="warmupSteps">warmup 总步数</param>
    /// <param name="startLr">warmup 起始学习率（默认 0）</param>
    /// <param name="endLr">warmup 结束学习率（等于内部调度器的 initial LR）</param>
    public LinearWarmup(ILRScheduler inner, int warmupSteps, double startLr = 0.0, double endLr = 0.001)
    {
        _inner = inner;
        _warmupSteps = Math.Max(1, warmupSteps);
        _startLr = startLr;
        _endLr = endLr;
        CurrentLR = startLr;
    }

    /// <summary>
    /// 从 warmup_epoch 和 step_each_epoch 创建。
    /// </summary>
    public static LinearWarmup FromEpochs(ILRScheduler inner, int warmupEpochs, int stepsPerEpoch, double startLr = 0.0, double endLr = 0.001)
    {
        return new LinearWarmup(inner, warmupEpochs * stepsPerEpoch, startLr, endLr);
    }

    public void Step(int step, int epoch)
    {
        if (step < _warmupSteps)
        {
            // Linear warmup
            var progress = (double)step / _warmupSteps;
            CurrentLR = _startLr + (_endLr - _startLr) * progress;
        }
        else
        {
            // Delegate to inner scheduler
            _inner.Step(step - _warmupSteps, epoch);
            CurrentLR = _inner.CurrentLR;
        }
    }
}
