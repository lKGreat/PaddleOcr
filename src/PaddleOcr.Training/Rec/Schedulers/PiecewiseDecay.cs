namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// PiecewiseDecay：分段衰减学习率调度器。
/// </summary>
public sealed class PiecewiseDecay : ILRScheduler
{
    private readonly float[] _milestones;
    private readonly float[] _values;
    private int _currentEpoch;
    public double CurrentLR { get; private set; }

    public PiecewiseDecay(float[] milestones, float[] values)
    {
        _milestones = milestones;
        _values = values;
        _currentEpoch = 0;
        CurrentLR = values[0];
    }

    public void Step(int step, int epoch)
    {
        _currentEpoch = epoch;
        for (var i = _milestones.Length - 1; i >= 0; i--)
        {
            if (epoch >= _milestones[i])
            {
                CurrentLR = _values[i];
                return;
            }
        }

        CurrentLR = _values[0];
    }
}
