namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// TwoStepCosineDecay：两阶段余弦退火（用于公式识别）。
/// 第一阶段在 [0, switchEpoch) 从 initialLr 到 midLr，
/// 第二阶段在 [switchEpoch, maxEpochs) 从 midLr 到 minLr。
/// 参考: ppocr/optimizer/lr_scheduler.py - TwoStepCosineDecay
/// </summary>
public sealed class TwoStepCosineDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _midLr;
    private readonly float _minLr;
    private readonly int _switchEpoch;
    private readonly int _maxEpochs;
    public double CurrentLR { get; private set; }

    public TwoStepCosineDecay(float initialLr, float midLr = 0.0001f, float minLr = 0.00001f,
        int switchEpoch = 50, int maxEpochs = 100)
    {
        _initialLr = initialLr;
        _midLr = midLr;
        _minLr = minLr;
        _switchEpoch = Math.Max(1, switchEpoch);
        _maxEpochs = Math.Max(switchEpoch + 1, maxEpochs);
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        if (epoch < _switchEpoch)
        {
            var progress = (float)epoch / _switchEpoch;
            CurrentLR = _midLr + (_initialLr - _midLr) * (1.0f + Math.Cos(Math.PI * progress)) / 2.0f;
        }
        else
        {
            var progress = (float)(epoch - _switchEpoch) / (_maxEpochs - _switchEpoch);
            progress = Math.Min(progress, 1.0f);
            CurrentLR = _minLr + (_midLr - _minLr) * (1.0f + Math.Cos(Math.PI * progress)) / 2.0f;
        }
    }
}
