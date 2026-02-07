using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// VisionLAN 预处理器：VisionLAN 特定处理。
/// 输入形状通常为 (3, 64, 256)。
/// </summary>
public sealed class VisionLanPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        using var resized = image.Clone(x => x.Resize(targetW, targetH));

        var channels = targetC == 1 ? 1 : 3;
        var data = new float[channels * targetH * targetW];

        // VisionLAN 使用 ImageNet 归一化
        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];

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
                    data[c * targetH * targetW + y * targetW + x] = (value - mean[c]) / std[c];
                }
            }
        }

        var dims = new[] { 1, channels, targetH, targetW };
        return new RecPreprocessResult(data, dims);
    }
}
