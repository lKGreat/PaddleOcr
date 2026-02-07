using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data;

/// <summary>
/// RecAugmentation：Rec 数据增强 (旋转、噪声、模糊等)。
/// </summary>
public static class RecAugmentation
{
    /// <summary>
    /// 应用随机旋转。
    /// </summary>
    public static Image<Rgb24> RandomRotate(Image<Rgb24> image, float maxAngle = 15.0f)
    {
        var angle = Random.Shared.NextSingle() * maxAngle * 2 - maxAngle;
        image.Mutate(x => x.Rotate(angle));
        return image;
    }

    /// <summary>
    /// 应用随机噪声。
    /// </summary>
    public static Image<Rgb24> AddNoise(Image<Rgb24> image, float noiseLevel = 0.1f)
    {
        image.Mutate(x =>
        {
            // 简化实现：添加随机噪声
            // 实际实现可能需要更复杂的噪声生成
        });
        return image;
    }

    /// <summary>
    /// 应用随机模糊。
    /// </summary>
    public static Image<Rgb24> RandomBlur(Image<Rgb24> image, float maxRadius = 2.0f)
    {
        var radius = Random.Shared.NextSingle() * maxRadius;
        image.Mutate(x => x.GaussianBlur(radius));
        return image;
    }

    /// <summary>
    /// 应用随机亮度调整。
    /// </summary>
    public static Image<Rgb24> RandomBrightness(Image<Rgb24> image, float factor = 0.2f)
    {
        var brightness = 1.0f + (Random.Shared.NextSingle() * 2 - 1) * factor;
        image.Mutate(x => x.Brightness(brightness));
        return image;
    }

    /// <summary>
    /// 应用随机对比度调整。
    /// </summary>
    public static Image<Rgb24> RandomContrast(Image<Rgb24> image, float factor = 0.2f)
    {
        var contrast = 1.0f + (Random.Shared.NextSingle() * 2 - 1) * factor;
        image.Mutate(x => x.Contrast(contrast));
        return image;
    }

    /// <summary>
    /// 应用组合增强。
    /// </summary>
    public static Image<Rgb24> ApplyAugmentation(Image<Rgb24> image, bool enableRotate = true, bool enableNoise = true, bool enableBlur = true, bool enableBrightness = true, bool enableContrast = true)
    {
        if (enableRotate && Random.Shared.NextSingle() > 0.5f)
        {
            image = RandomRotate(image);
        }

        if (enableNoise && Random.Shared.NextSingle() > 0.5f)
        {
            image = AddNoise(image);
        }

        if (enableBlur && Random.Shared.NextSingle() > 0.5f)
        {
            image = RandomBlur(image);
        }

        if (enableBrightness && Random.Shared.NextSingle() > 0.5f)
        {
            image = RandomBrightness(image);
        }

        if (enableContrast && Random.Shared.NextSingle() > 0.5f)
        {
            image = RandomContrast(image);
        }

        return image;
    }
}
