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
    /// 应用高斯噪声。
    /// </summary>
    public static Image<Rgb24> AddNoise(Image<Rgb24> image, float noiseLevel = 0.1f)
    {
        var rng = Random.Shared;
        var maxNoise = (int)(noiseLevel * 255);
        image.Mutate(ctx =>
        {
            // 使用 ProcessPixelRowsAsVector4 进行像素级操作
        });

        // 直接在像素上添加高斯噪声
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                // Box-Muller 近似高斯噪声
                var u1 = rng.NextSingle();
                var u2 = rng.NextSingle();
                var gaussianNoise = MathF.Sqrt(-2f * MathF.Log(Math.Max(u1, 1e-10f))) * MathF.Cos(2f * MathF.PI * u2);
                var noise = (int)(gaussianNoise * maxNoise);

                var r = Math.Clamp(pixel.R + noise, 0, 255);
                var g = Math.Clamp(pixel.G + noise, 0, 255);
                var b = Math.Clamp(pixel.B + noise, 0, 255);
                image[x, y] = new Rgb24((byte)r, (byte)g, (byte)b);
            }
        }

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
