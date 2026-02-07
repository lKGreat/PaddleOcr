using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data;

/// <summary>
/// Rec 训练图像 resize 结果。
/// </summary>
public sealed record RecResizeResult(float[] Data, int Channels, int Height, int Width, float ValidRatio);

/// <summary>
/// Rec 训练 Resize 策略接口。
/// </summary>
public interface IRecTrainingResize
{
    RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW);
}

/// <summary>
/// RecResizeImg：标准 Rec resize。
/// 保持宽高比缩放到 targetH，然后 padding 到 targetW。
/// 归一化: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
/// 参考: ppocr/data/imaug/rec_img_aug.py - RecResizeImg
/// </summary>
public sealed class RecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        // 按目标高度等比缩放
        var ratio = (float)targetH / imgH;
        var resizedW = Math.Min((int)MathF.Ceiling(imgW * ratio), targetW);

        image.Mutate(x => x.Resize(resizedW, targetH));

        var validRatio = Math.Min(1f, (float)resizedW / targetW);
        var data = new float[targetC * targetH * targetW];
        var hw = targetH * targetW;

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                if (x < resizedW)
                {
                    var p = image[x, y];
                    data[idx] = p.R / 127.5f - 1f;
                    data[hw + idx] = p.G / 127.5f - 1f;
                    data[2 * hw + idx] = p.B / 127.5f - 1f;
                }
                else
                {
                    // padding 区域: 归一化后为 0 对应原始 127.5
                    data[idx] = 0f;
                    data[hw + idx] = 0f;
                    data[2 * hw + idx] = 0f;
                }
            }
        }

        return new RecResizeResult(data, targetC, targetH, targetW, validRatio);
    }
}

/// <summary>
/// SARRecResizeImg：SAR 专用 resize。
/// 保持宽高比缩放，计算 valid_ratio。
/// 参考: ppocr/data/imaug/rec_img_aug.py - SARRecResizeImg
/// </summary>
public sealed class SARRecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        // SAR: 保持宽高比缩放到 (targetH, resizedW)
        var ratio = (float)targetH / imgH;
        var resizedW = Math.Max(1, Math.Min((int)MathF.Round(imgW * ratio), targetW));

        image.Mutate(x => x.Resize(resizedW, targetH));

        var validRatio = Math.Min(1f, (float)resizedW / targetW);
        var data = new float[targetC * targetH * targetW];
        var hw = targetH * targetW;

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                if (x < resizedW)
                {
                    var p = image[x, y];
                    data[idx] = p.R / 127.5f - 1f;
                    data[hw + idx] = p.G / 127.5f - 1f;
                    data[2 * hw + idx] = p.B / 127.5f - 1f;
                }
                else
                {
                    data[idx] = 0f;
                    data[hw + idx] = 0f;
                    data[2 * hw + idx] = 0f;
                }
            }
        }

        return new RecResizeResult(data, targetC, targetH, targetW, validRatio);
    }
}

/// <summary>
/// GrayRecResizeImg：灰度 resize（NRTR/ViTSTR 使用）。
/// 转灰度后 resize + padding。
/// 参考: ppocr/data/imaug/rec_img_aug.py - GrayRecResizeImg
/// </summary>
public sealed class GrayRecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        var imgH = image.Height;
        var imgW = image.Width;

        var ratio = (float)targetH / imgH;
        var resizedW = Math.Min((int)MathF.Ceiling(imgW * ratio), targetW);

        // 转灰度
        image.Mutate(x => x.Grayscale().Resize(resizedW, targetH));

        var validRatio = Math.Min(1f, (float)resizedW / targetW);

        // 灰度: 1 channel
        var channels = targetC == 1 ? 1 : targetC;
        var data = new float[channels * targetH * targetW];
        var hw = targetH * targetW;

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                if (x < resizedW)
                {
                    var p = image[x, y];
                    var gray = p.R / 127.5f - 1f; // 灰度后 R=G=B
                    if (channels == 1)
                    {
                        data[idx] = gray;
                    }
                    else
                    {
                        data[idx] = gray;
                        data[hw + idx] = gray;
                        data[2 * hw + idx] = gray;
                    }
                }
                // padding 区域保持 0
            }
        }

        return new RecResizeResult(data, channels, targetH, targetW, validRatio);
    }
}

/// <summary>
/// Resize 策略工厂：根据算法名称创建对应的 resize 策略。
/// </summary>
public static class RecTrainingResizeFactory
{
    public static IRecTrainingResize Create(string algorithmOrHead)
    {
        return algorithmOrHead.ToLowerInvariant() switch
        {
            "sar" or "sarhead" or "robustscanner" or "robustscannerhead" => new SARRecResizeImg(),
            "nrtr" or "nrtrhead" or "vitstr" or "vitstrhead" => new GrayRecResizeImg(),
            _ => new RecResizeImg()
        };
    }
}
