using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// SVTRRecAug：SVTR-specific recognition augmentation.
/// 使用 geometry + deterioration + color jitter pipeline.
/// 参考: ppocr/data/imaug/rec_img_aug.py - SVTRRecAug
/// </summary>
public sealed class SVTRRecAug
{
    private readonly float _geometryP;
    private readonly float _deteriorationP;
    private readonly float _colorjitterP;

    public SVTRRecAug(float geometryP = 0.5f, float deteriorationP = 0.25f, float colorjitterP = 0.25f)
    {
        _geometryP = geometryP;
        _deteriorationP = deteriorationP;
        _colorjitterP = colorjitterP;
    }

    public Image<Rgb24> Apply(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;

        // Geometry transforms
        if (rng.NextSingle() < _geometryP)
        {
            var choice = rng.Next(3);
            image = choice switch
            {
                0 => ApplyRotation(image, rng),
                1 => ApplyPerspective(image, rng),
                _ => ApplyAffine(image, rng)
            };
        }

        // Deterioration transforms
        if (rng.NextSingle() < _deteriorationP)
        {
            var choice = rng.Next(3);
            image = choice switch
            {
                0 => ApplyGaussianBlur(image, rng),
                1 => ApplyMotionBlur(image, rng),
                _ => ApplyGaussianNoise(image, rng)
            };
        }

        // Color jitter
        if (rng.NextSingle() < _colorjitterP)
        {
            image = ApplyColorJitter(image, rng);
        }

        return image;
    }

    private static Image<Rgb24> ApplyRotation(Image<Rgb24> image, Random rng)
    {
        var angle = (rng.NextSingle() * 2 - 1) * 5f; // ±5 degrees
        image.Mutate(ctx => ctx.Rotate(angle));
        return image;
    }

    private static Image<Rgb24> ApplyPerspective(Image<Rgb24> image, Random rng)
    {
        // Simplified: use slight affine transform as perspective approximation
        return ApplyAffine(image, rng);
    }

    private static Image<Rgb24> ApplyAffine(Image<Rgb24> image, Random rng)
    {
        // Slight skew
        var angle = (rng.NextSingle() * 2 - 1) * 3f;
        image.Mutate(ctx => ctx.Skew(angle, 0));
        return image;
    }

    private static Image<Rgb24> ApplyGaussianBlur(Image<Rgb24> image, Random rng)
    {
        var sigma = rng.NextSingle() * 2.0f + 0.5f;
        image.Mutate(ctx => ctx.GaussianBlur(sigma));
        return image;
    }

    private static Image<Rgb24> ApplyMotionBlur(Image<Rgb24> image, Random rng)
    {
        // Approximate motion blur with gaussian blur
        var sigma = rng.NextSingle() * 1.5f + 0.5f;
        image.Mutate(ctx => ctx.GaussianBlur(sigma));
        return image;
    }

    private static Image<Rgb24> ApplyGaussianNoise(Image<Rgb24> image, Random rng)
    {
        var noiseLevel = rng.NextSingle() * 20;
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < row.Length; x++)
                {
                    var noise = (int)(rng.NextSingle() * 2 - 1) * noiseLevel;
                    row[x] = new Rgb24(
                        (byte)Math.Clamp(row[x].R + noise, 0, 255),
                        (byte)Math.Clamp(row[x].G + noise, 0, 255),
                        (byte)Math.Clamp(row[x].B + noise, 0, 255));
                }
            }
        });
        return image;
    }

    private static Image<Rgb24> ApplyColorJitter(Image<Rgb24> image, Random rng)
    {
        var brightness = 1.0f + (rng.NextSingle() * 0.4f - 0.2f);
        image.Mutate(ctx => ctx.Brightness(brightness));
        var contrast = 1.0f + (rng.NextSingle() * 0.4f - 0.2f);
        image.Mutate(ctx => ctx.Contrast(contrast));
        return image;
    }
}

/// <summary>
/// ABINetRecAug：ABINet-specific recognition augmentation.
/// 与 SVTRRecAug 结构相同，使用 CV-style geometry/deterioration/colorjitter。
/// 参考: ppocr/data/imaug/rec_img_aug.py - ABINetRecAug
/// </summary>
public sealed class ABINetRecAug
{
    private readonly SVTRRecAug _inner;

    public ABINetRecAug(float geometryP = 0.5f, float deteriorationP = 0.25f, float colorjitterP = 0.25f)
    {
        _inner = new SVTRRecAug(geometryP, deteriorationP, colorjitterP);
    }

    public Image<Rgb24> Apply(Image<Rgb24> image, Random? rng = null)
    {
        return _inner.Apply(image, rng);
    }
}

/// <summary>
/// ParseQRecAug：ParseQ-specific recognition augmentation.
/// 与 SVTRRecAug 结构相同。
/// 参考: ppocr/data/imaug/rec_img_aug.py - ParseQRecAug
/// </summary>
public sealed class ParseQRecAug
{
    private readonly SVTRRecAug _inner;

    public ParseQRecAug(float geometryP = 0.5f, float deteriorationP = 0.25f, float colorjitterP = 0.25f)
    {
        _inner = new SVTRRecAug(geometryP, deteriorationP, colorjitterP);
    }

    public Image<Rgb24> Apply(Image<Rgb24> image, Random? rng = null)
    {
        return _inner.Apply(image, rng);
    }
}
