using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// SPIN 预处理器：SPIN 特定归一化。
/// 输入形状通常为 (3, 32, 100)。
/// </summary>
public sealed class SpinRecPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        using var resized = image.Clone(x => x.Resize(targetW, targetH));

        var channels = targetC == 1 ? 1 : 3;
        var data = new float[channels * targetH * targetW];

        // SPIN 使用标准的 (x - 0.5) / 0.5 归一化
        for (var c = 0; c < channels; c++)
        {
            for (var y = 0; y < targetH; y++)
            {
                for (var x = 0; x < targetW; x++)
                {
                    var pixel = resized[x, y];
                    var value = c switch
                    {
                        0 => pixel.R / 255f,
                        1 => pixel.G / 255f,
                        _ => pixel.B / 255f
                    };
                    data[c * targetH * targetW + y * targetW + x] = (value - 0.5f) / 0.5f;
                }
            }
        }

        var dims = new[] { 1, channels, targetH, targetW };
        return new RecPreprocessResult(data, dims);
    }
}
