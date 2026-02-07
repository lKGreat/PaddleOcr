using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Inference.Rec;

/// <summary>
/// Rec 预处理器接口。
/// 将输入图像转换为模型输入张量。
/// </summary>
public interface IRecPreprocessor
{
    /// <summary>
    /// 预处理图像为模型输入张量。
    /// </summary>
    /// <param name="image">输入图像</param>
    /// <param name="targetC">目标通道数</param>
    /// <param name="targetH">目标高度</param>
    /// <param name="targetW">目标宽度</param>
    /// <returns>预处理后的 float 数组和维度信息</returns>
    RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW);
}

/// <summary>
/// 预处理结果，包含张量数据、维度信息和额外元数据。
/// </summary>
public sealed record RecPreprocessResult(
    float[] Data,
    int[] Dims,
    float ValidRatio = 1.0f);
