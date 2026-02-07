namespace PaddleOcr.Data;

/// <summary>
/// MultiScaleDataSet：多尺度数据集。
/// </summary>
public sealed class MultiScaleDataSet
{
    private readonly List<(int Height, int Width)> _scales;
    private readonly Random _random;

    public MultiScaleDataSet(List<(int Height, int Width)>? scales = null)
    {
        _scales = scales ?? new List<(int, int)> { (32, 320), (48, 320), (64, 320) };
        _random = new Random();
    }

    /// <summary>
    /// 获取随机尺度。
    /// </summary>
    public (int Height, int Width) GetRandomScale()
    {
        return _scales[_random.Next(_scales.Count)];
    }

    /// <summary>
    /// 获取所有尺度。
    /// </summary>
    public IReadOnlyList<(int Height, int Width)> GetAllScales()
    {
        return _scales;
    }
}
