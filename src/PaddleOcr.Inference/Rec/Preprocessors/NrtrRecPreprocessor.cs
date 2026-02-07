using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// NRTR/ViTSTR 预处理器：灰度转换 + 特定归一化。
/// 输入形状通常为 (1, 32, 100)。
/// </summary>
public sealed class NrtrRecPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        using var resized = image.Clone(x => x.Resize(targetW, targetH));

        var channels = targetC == 1 ? 1 : 3;
        var data = new float[channels * targetH * targetW];

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var pixel = resized[x, y];
                if (channels == 1)
                {
                    // 灰度转换 + normalize
                    var gray = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255f;
                    data[y * targetW + x] = (gray - 0.5f) / 0.5f;
                }
                else
                {
                    for (var c = 0; c < 3; c++)
                    {
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
        }

        var dims = new[] { 1, channels, targetH, targetW };
        return new RecPreprocessResult(data, dims);
    }
}
