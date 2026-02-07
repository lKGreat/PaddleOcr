using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// 默认 Rec 预处理器：按宽高比 resize + normalize。
/// 适用于大多数 CTC 系列算法（CRNN, SVTR, SVTR_LCNet, SVTR_HGNet, STARNet 等）。
/// </summary>
public sealed class DefaultRecPreprocessor : IRecPreprocessor
{
    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        // 按宽高比计算 resize 后的宽度
        var ratio = (float)imgW / imgH;
        var resizedW = (int)MathF.Ceiling(targetH * ratio);
        resizedW = Math.Min(resizedW, targetW);
        resizedW = Math.Max(resizedW, 1);

        using var resized = image.Clone(x => x.Resize(resizedW, targetH));

        var channels = targetC == 1 ? 1 : 3;
        var data = new float[channels * targetH * targetW];

        // 填充 padding 区域为 0（归一化后的 -0.5/0.5 对应值）
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
                        float value;
                        if (channels == 1)
                        {
                            value = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255f;
                        }
                        else
                        {
                            value = c switch
                            {
                                0 => pixel.R / 255f,
                                1 => pixel.G / 255f,
                                _ => pixel.B / 255f
                            };
                        }

                        // normalize: (value - 0.5) / 0.5
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
        return new RecPreprocessResult(data, dims);
    }
}
