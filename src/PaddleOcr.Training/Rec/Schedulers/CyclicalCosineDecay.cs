namespace PaddleOcr.Training.Rec.Schedulers;

/// <summary>
/// CyclicalCosineDecay：周期性余弦衰减学习率调度器。
/// 在每个周期内做余弦退火，然后重置。
/// 参考: paddle.optimizer.lr.CyclicCosineDecayLR
/// </summary>
public sealed class CyclicalCosineDecay : ILRScheduler
{
    private readonly float _initialLr;
    private readonly float _minLr;
    private readonly int _cycleLength;
    private readonly float _decayFactor;

    public double CurrentLR { get; private set; }

    /// <param name="initialLr">初始学习率</param>
    /// <param name="minLr">最小学习率</param>
    /// <param name="cycleLength">每个周期的 epoch 数</param>
    /// <param name="decayFactor">每个周期结束后初始 LR 的衰减系数</param>
    public CyclicalCosineDecay(float initialLr = 0.001f, float minLr = 0.0001f, int cycleLength = 50, float decayFactor = 0.5f)
    {
        _initialLr = initialLr;
        _minLr = minLr;
        _cycleLength = Math.Max(1, cycleLength);
        _decayFactor = decayFactor;
        CurrentLR = initialLr;
    }

    public void Step(int step, int epoch)
    {
        var cycle = epoch / _cycleLength;
        var posInCycle = epoch % _cycleLength;
        var progress = (float)posInCycle / _cycleLength;

        // 每个周期的初始 LR 按 decay_factor 衰减
        var cycleInitLr = _initialLr * MathF.Pow(_decayFactor, cycle);
        var lr = _minLr + (cycleInitLr - _minLr) * (1f + MathF.Cos(MathF.PI * progress)) / 2f;
        CurrentLR = lr;
    }
}
