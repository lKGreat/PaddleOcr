using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// SAR/RobustScanner 预处理器：保持宽高比 + valid_ratio 计算。
/// 输入形状通常为 (3, 48, 48) 但 SAR 支持更宽的输入。
/// </summary>
public sealed class SarRecPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        var ratio = (float)imgW / imgH;
        var resizedW = (int)MathF.Ceiling(targetH * ratio);
        resizedW = Math.Min(resizedW, targetW);
        resizedW = Math.Max(resizedW, 1);

        var validRatio = (float)resizedW / targetW;

        using var resized = image.Clone(x => x.Resize(resizedW, targetH));

        var channels = targetC == 1 ? 1 : 3;
        var data = new float[channels * targetH * targetW];

        for (var c = 0; c < channels; c++)
        {
            for (var y = 0; y < targetH; y++)
            {
                for (var x = 0; x < targetW; x++)
                {
                    var idx = c * targetH * targetW + y * targetW + x;
                    if (x < resizedW)
                    {
                        var pixel = resized[x, y];
                        var value = c switch
                        {
                            0 => pixel.R / 255f,
                            1 => pixel.G / 255f,
                            _ => pixel.B / 255f
                        };
                        data[idx] = (value - 0.5f) / 0.5f;
                    }
                    else
                    {
                        data[idx] = 0f;
                    }
                }
            }
        }

        var dims = new[] { 1, channels, targetH, targetW };
        return new RecPreprocessResult(data, dims, validRatio);
    }
}
