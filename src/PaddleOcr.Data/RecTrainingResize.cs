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
/// SRNRecResizeImg：SRN 专用 resize。
/// 硬 resize 到固定尺寸 (targetH x targetW)，无 padding。
/// 参考: ppocr/data/imaug/rec_img_aug.py - SRNRecResizeImg
/// </summary>
public sealed class SRNRecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        image.Mutate(x => x.Resize(targetW, targetH));
        var data = new float[targetC * targetH * targetW];
        var hw = targetH * targetW;
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                var p = image[x, y];
                data[idx] = p.R / 127.5f - 1f;
                data[hw + idx] = p.G / 127.5f - 1f;
                data[2 * hw + idx] = p.B / 127.5f - 1f;
            }
        }
        return new RecResizeResult(data, targetC, targetH, targetW, 1.0f);
    }
}

/// <summary>
/// PRENResizeImg：PREN 专用硬 resize + (pixel/255 - 0.5) / 0.5 归一化。
/// 参考: ppocr/data/imaug/rec_img_aug.py - PRENResizeImg
/// </summary>
public sealed class PRENResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        image.Mutate(x => x.Resize(targetW, targetH));
        var data = new float[targetC * targetH * targetW];
        var hw = targetH * targetW;
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                var p = image[x, y];
                data[idx] = (p.R / 255f - 0.5f) / 0.5f;
                data[hw + idx] = (p.G / 255f - 0.5f) / 0.5f;
                data[2 * hw + idx] = (p.B / 255f - 0.5f) / 0.5f;
            }
        }
        return new RecResizeResult(data, targetC, targetH, targetW, 1.0f);
    }
}

/// <summary>
/// SPINRecResizeImg：SPIN 专用 resize，转灰度 + 自定义 mean/std 归一化。
/// 参考: ppocr/data/imaug/rec_img_aug.py - SPINRecResizeImg
/// </summary>
public sealed class SPINRecResizeImg : IRecTrainingResize
{
    private readonly float _mean;
    private readonly float _std;

    public SPINRecResizeImg(float mean = 127.5f, float std = 127.5f)
    {
        _mean = mean;
        _std = std;
    }

    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        image.Mutate(x => x.Grayscale().Resize(targetW, targetH));
        var data = new float[1 * targetH * targetW];
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                var p = image[x, y];
                data[idx] = (p.R - _mean) / _std;
            }
        }
        return new RecResizeResult(data, 1, targetH, targetW, 1.0f);
    }
}

/// <summary>
/// ABINetRecResizeImg：ABINet 专用 resize，使用 ImageNet mean/std 归一化。
/// 参考: ppocr/data/imaug/rec_img_aug.py - ABINetRecResizeImg
/// </summary>
public sealed class ABINetRecResizeImg : IRecTrainingResize
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        image.Mutate(x => x.Resize(targetW, targetH));
        var data = new float[targetC * targetH * targetW];
        var hw = targetH * targetW;
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                var p = image[x, y];
                data[idx] = (p.R / 255f - Mean[0]) / Std[0];
                data[hw + idx] = (p.G / 255f - Mean[1]) / Std[1];
                data[2 * hw + idx] = (p.B / 255f - Mean[2]) / Std[2];
            }
        }
        return new RecResizeResult(data, targetC, targetH, targetW, 1.0f);
    }
}

/// <summary>
/// SVTRRecResizeImg：SVTR 专用 resize with padding。
/// 与 RecResizeImg 相同逻辑。
/// 参考: ppocr/data/imaug/rec_img_aug.py - SVTRRecResizeImg
/// </summary>
public sealed class SVTRRecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        // Same as RecResizeImg
        return new RecResizeImg().Resize(image, targetC, targetH, targetW);
    }
}

/// <summary>
/// VLRecResizeImg：VisionLAN 专用 resize。
/// 直接 resize 到目标尺寸（不保持宽高比）。
/// 参考: ppocr/data/imaug/rec_img_aug.py - VLRecResizeImg
/// </summary>
public sealed class VLRecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        image.Mutate(x => x.Resize(targetW, targetH));
        var data = new float[targetC * targetH * targetW];
        var hw = targetH * targetW;
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                var p = image[x, y];
                data[idx] = p.R / 127.5f - 1f;
                data[hw + idx] = p.G / 127.5f - 1f;
                data[2 * hw + idx] = p.B / 127.5f - 1f;
            }
        }
        return new RecResizeResult(data, targetC, targetH, targetW, 1.0f);
    }
}

/// <summary>
/// RFLRecResizeImg：RFL 专用 resize，转灰度 + padding。
/// 参考: ppocr/data/imaug/rec_img_aug.py - RFLRecResizeImg
/// </summary>
public sealed class RFLRecResizeImg : IRecTrainingResize
{
    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        image.Mutate(x => x.Grayscale());
        var imgH = image.Height;
        var imgW = image.Width;
        var ratio = (float)targetH / imgH;
        var resizedW = Math.Min((int)MathF.Ceiling(imgW * ratio), targetW);
        image.Mutate(x => x.Resize(resizedW, targetH));

        var validRatio = Math.Min(1f, (float)resizedW / targetW);
        var data = new float[1 * targetH * targetW];
        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var idx = y * targetW + x;
                data[idx] = x < resizedW ? image[x, y].R / 127.5f - 1f : 0f;
            }
        }
        return new RecResizeResult(data, 1, targetH, targetW, validRatio);
    }
}

/// <summary>
/// RobustScannerRecResizeImg：RobustScanner 专用 resize。
/// 与 SARRecResizeImg 类似，额外添加 word_positions。
/// 参考: ppocr/data/imaug/rec_img_aug.py - RobustScannerRecResizeImg
/// </summary>
public sealed class RobustScannerRecResizeImg : IRecTrainingResize
{
    private readonly float _widthDownsampleRatio;

    public RobustScannerRecResizeImg(float widthDownsampleRatio = 0.25f)
    {
        _widthDownsampleRatio = widthDownsampleRatio;
    }

    public RecResizeResult Resize(Image<Rgb24> image, int targetC, int targetH, int targetW)
    {
        // Use SAR resize logic
        return new SARRecResizeImg().Resize(image, targetC, targetH, targetW);
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
            "sar" or "sarhead" => new SARRecResizeImg(),
            "robustscanner" or "robustscannerhead" => new RobustScannerRecResizeImg(),
            "nrtr" or "nrtrhead" or "vitstr" or "vitstrhead" => new GrayRecResizeImg(),
            "srn" or "srnhead" => new SRNRecResizeImg(),
            "pren" or "prenhead" => new PRENResizeImg(),
            "spin" or "spinattentionhead" => new SPINRecResizeImg(),
            "abinet" or "abinethead" => new ABINetRecResizeImg(),
            "svtr" or "svtrnet" or "svtrv2" or "repsvtr" => new SVTRRecResizeImg(),
            "vl" or "vlhead" or "visionlan" => new VLRecResizeImg(),
            "rfl" or "rflhead" => new RFLRecResizeImg(),
            _ => new RecResizeImg()
        };
    }
}
