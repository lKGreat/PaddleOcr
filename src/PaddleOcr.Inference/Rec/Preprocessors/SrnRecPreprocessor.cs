using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// SRN 预处理器：灰度 + resize + 编码额外位置信息。
/// 输入形状通常为 (1, 64, 256)。
/// </summary>
public sealed class SrnRecPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        using var resized = image.Clone(x => x.Resize(targetW, targetH));

        // SRN 使用灰度输入
        var data = new float[1 * targetH * targetW];

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var pixel = resized[x, y];
                var gray = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255f;
                data[y * targetW + x] = (gray - 0.5f) / 0.5f;
            }
        }

        var dims = new[] { 1, 1, targetH, targetW };
        return new RecPreprocessResult(data, dims);
    }
}
