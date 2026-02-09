namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// OneCycleDecay：One Cycle 学习率策略。
/// 先从 initial_lr 上升到 max_lr，再下降到 min_lr。
/// 参考: ppocr/optimizer/lr_scheduler.py - OneCycleDecay
/// </summary>
public sealed class OneCycleDecay : ILRScheduler
{
    private readonly float _maxLr;
    private readonly float _minLr;
    private readonly int _totalSteps;
    private readonly float _pctStart;
    public double CurrentLR { get; private set; }

    public OneCycleDecay(float maxLr = 0.001f, float minLr = 0.0f, int totalSteps = 1000, float pctStart = 0.3f)
    {
        _maxLr = maxLr;
        _minLr = minLr;
        _totalSteps = Math.Max(1, totalSteps);
        _pctStart = pctStart;
        CurrentLR = minLr;
    }

    public void Step(int step, int epoch)
    {
        var progress = Math.Min((float)step / _totalSteps, 1.0f);

        if (progress < _pctStart)
        {
            // Phase 1: linear warmup from minLr to maxLr
            var phaseProgress = progress / _pctStart;
            CurrentLR = _minLr + (_maxLr - _minLr) * phaseProgress;
        }
        else
        {
            // Phase 2: cosine annealing from maxLr to minLr
            var phaseProgress = (progress - _pctStart) / (1.0f - _pctStart);
            CurrentLR = _minLr + (_maxLr - _minLr) * (1.0f + Math.Cos(Math.PI * phaseProgress)) / 2.0f;
        }
    }
}
