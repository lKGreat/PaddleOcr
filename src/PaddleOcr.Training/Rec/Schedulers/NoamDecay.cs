namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// NoamDecay：Transformer 风格的学习率调度器（NRTR 使用）。
/// lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
/// 如果提供了 learning_rate，则 lr = learning_rate * min(step^(-0.5), step * warmup_steps^(-1.5))
/// 参考: paddle.optimizer.lr.NoamDecay
/// </summary>
public sealed class NoamDecay : ILRScheduler
{
    private readonly int _dModel;
    private readonly int _warmupSteps;
    private readonly float _learningRate;

    public double CurrentLR { get; private set; }

    /// <param name="dModel">模型维度（用于计算基础学习率）</param>
    /// <param name="warmupSteps">warmup 步数</param>
    /// <param name="learningRate">覆盖学习率（如果 > 0 则忽略 dModel）</param>
    public NoamDecay(int dModel = 512, int warmupSteps = 4000, float learningRate = 0f)
    {
        _dModel = dModel;
        _warmupSteps = Math.Max(1, warmupSteps);
        _learningRate = learningRate;
        CurrentLR = learningRate > 0 ? learningRate : Math.Pow(dModel, -0.5);
    }

    public void Step(int step, int epoch)
    {
        var s = Math.Max(step, 1);
        var scale = Math.Min(Math.Pow(s, -0.5), s * Math.Pow(_warmupSteps, -1.5));

        if (_learningRate > 0)
        {
            CurrentLR = _learningRate * scale;
        }
        else
        {
            CurrentLR = Math.Pow(_dModel, -0.5) * scale;
        }
    }
}
