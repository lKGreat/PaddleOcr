namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// MultiStepDecay：在指定的 epoch 里程碑处将学习率乘以 gamma。
/// 参考: ppocr/optimizer/learning_rate.py - MultiStepDecay
/// </summary>
public sealed class MultiStepDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly int[] _milestones;
    private readonly float _gamma;
    public double CurrentLR { get; private set; }

    public MultiStepDecay(float initialLr, int[] milestones, float gamma = 0.1f)
    {
        _initialLr = initialLr;
        _milestones = milestones.OrderBy(m => m).ToArray();
        _gamma = gamma;
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        var numDecays = _milestones.Count(m => epoch >= m);
        CurrentLR = _initialLr * Math.Pow(_gamma, numDecays);
    }
}
