namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// ConstLR：常量学习率（可与 LinearWarmup 组合使用实现 warmup + const）。
/// 参考: ppocr/optimizer/learning_rate.py - Const
/// </summary>
public sealed class ConstLR : ILRScheduler
{
    public double CurrentLR { get; private set; }

    public ConstLR(float learningRate)
    {
        CurrentLR = learningRate;
    }

    public void Step(int step, int epoch)
    {
        // Constant - no change
    }
}
