using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Rec.Preprocessors;

/// <summary>
/// CAN 预处理器：特殊归一化 + image_mask。
/// CAN 模型通常需要反转灰度图像。
/// 输入形状通常为 (1, 256, 256)。
/// </summary>
public sealed class CanRecPreprocessor : IRecPreprocessor
{
    private readonly bool _imageInverse;

    public CanRecPreprocessor(bool imageInverse = true)
    {
        _imageInverse = imageInverse;
    }

    public RecPreprocessResult Process(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        // 按比例缩放，保持长边不超过目标尺寸
        var scale = Math.Min((float)targetH / imgH, (float)targetW / imgW);
        var resizedH = Math.Max(1, (int)(imgH * scale));
        var resizedW = Math.Max(1, (int)(imgW * scale));

        using var resized = image.Clone(x => x.Resize(resizedW, resizedH));

        // CAN 使用灰度输入
        var data = new float[1 * targetH * targetW];

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                if (y < resizedH && x < resizedW)
                {
                    var pixel = resized[x, y];
                    var gray = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255f;
                    if (_imageInverse)
                    {
                        gray = 1f - gray;
                    }

                    // 归一化到 [-1, 1]，与其他预处理器保持一致
                    data[y * targetW + x] = (gray - 0.5f) / 0.5f;
                }
                else
                {
                    // padding 区域：归一化后的零值对应原始的 0.5
                    data[y * targetW + x] = -1f;
                }
            }
        }

        var dims = new[] { 1, 1, targetH, targetW };
        return new RecPreprocessResult(data, dims);
    }
}
