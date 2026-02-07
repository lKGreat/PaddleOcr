using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// LaTeXOCR 预处理器：特殊归一化 + 动态尺寸。
/// 输入形状通常为 (3, 192, 672)。
/// </summary>
public sealed class LaTeXOcrPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        // 按比例缩放
        var ratio = Math.Min((float)targetH / imgH, (float)targetW / imgW);
        var resizedH = Math.Max(1, (int)(imgH * ratio));
        var resizedW = Math.Max(1, (int)(imgW * ratio));

        // 确保尺寸是 16 的倍数（常见要求）
        resizedH = Math.Max(16, (resizedH + 15) / 16 * 16);
        resizedW = Math.Max(16, (resizedW + 15) / 16 * 16);
        resizedH = Math.Min(resizedH, targetH);
        resizedW = Math.Min(resizedW, targetW);

        using var resized = image.Clone(x => x.Resize(resizedW, resizedH));

        var channels = targetC == 1 ? 1 : 3;
        var data = new float[channels * targetH * targetW];

        // LaTeXOCR 使用 ImageNet 归一化
        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];

        for (var c = 0; c < channels; c++)
        {
            for (var y = 0; y < targetH; y++)
            {
                for (var x = 0; x < targetW; x++)
                {
                    var idx = c * targetH * targetW + y * targetW + x;
                    if (y < resizedH && x < resizedW)
                    {
                        var pixel = resized[x, y];
                        var value = c switch
                        {
                            0 => pixel.R / 255f,
                            1 => pixel.G / 255f,
                            _ => pixel.B / 255f
                        };
                        data[idx] = (value - mean[c]) / std[c];
                    }
                    else
                    {
                        // padding 填充零值（归一化后的 0 对应原始的 mean 值）
                        data[idx] = (0f - mean[c]) / std[c];
                    }
                }
            }
        }

        var dims = new[] { 1, channels, targetH, targetW };
        return new RecPreprocessResult(data, dims);
    }
}
