using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// Base data augmentation for recognition and classification training.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/rec_img_aug.py BaseDataAugmentation.
/// 
/// Applies 6 augmentation operations with configurable probabilities:
/// - Random crop
/// - Gaussian blur
/// - HSV augmentation
/// - Color jitter
/// - Gaussian noise
/// - Color reverse (255 - pixel)
/// </summary>
public sealed class BaseDataAugmentation
{
    private readonly float _cropProb;
    private readonly float _reverseProb;
    private readonly float _noiseProb;
    private readonly float _jitterProb;
    private readonly float _blurProb;
    private readonly float _hsvAugProb;

    /// <summary>
    /// Creates a new BaseDataAugmentation instance with configurable probabilities.
    /// Default probabilities match Python PaddleOCR: 0.4 for all operations.
    /// </summary>
    public BaseDataAugmentation(
        float cropProb = 0.4f,
        float reverseProb = 0.4f,
        float noiseProb = 0.4f,
        float jitterProb = 0.4f,
        float blurProb = 0.4f,
        float hsvAugProb = 0.4f)
    {
        _cropProb = cropProb;
        _reverseProb = reverseProb;
        _noiseProb = noiseProb;
        _jitterProb = jitterProb;
        _blurProb = blurProb;
        _hsvAugProb = hsvAugProb;
    }

    /// <summary>
    /// Apply base data augmentation to an image.
    /// Each operation is applied independently with its configured probability.
    /// </summary>
    public Image<Rgb24> Apply(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;

        // 1. Random crop
        if (rng.NextSingle() < _cropProb && image.Height >= 20 && image.Width >= 20)
        {
            image = RandomCrop(image, rng);
        }

        // 2. Gaussian blur (kernel=5, sigma=1)
        if (rng.NextSingle() < _blurProb)
        {
            image.Mutate(x => x.GaussianBlur(1f));
        }

        // 3. HSV augmentation
        if (rng.NextSingle() < _hsvAugProb)
        {
            image = HsvAugment(image, rng);
        }

        // 4. Color jitter (brightness + contrast + saturation)
        if (rng.NextSingle() < _jitterProb)
        {
            image = ColorJitter(image, rng);
        }

        // 5. Gaussian noise
        if (rng.NextSingle() < _noiseProb)
        {
            image = AddGaussianNoise(image, rng);
        }

        // 6. Color reverse (255 - pixel)
        if (rng.NextSingle() < _reverseProb)
        {
            image.Mutate(x => x.Invert());
        }

        return image;
    }

    /// <summary>
    /// Random crop (preserving text area).
    /// Reference: ppocr/data/imaug/rec_img_aug.py random_crop
    /// </summary>
    private static Image<Rgb24> RandomCrop(Image<Rgb24> image, Random rng)
    {
        var w = image.Width;
        var h = image.Height;
        var cropRatio = 0.1f;
        var cropW = (int)(w * cropRatio * rng.NextSingle());
        var cropH = (int)(h * cropRatio * rng.NextSingle());
        if (cropW >= w / 2 || cropH >= h / 2) return image;

        var x1 = rng.Next(0, Math.Max(1, cropW));
        var y1 = rng.Next(0, Math.Max(1, cropH));
        var newW = Math.Max(1, w - cropW);
        var newH = Math.Max(1, h - cropH);
        if (x1 + newW > w) newW = w - x1;
        if (y1 + newH > h) newH = h - y1;
        if (newW <= 0 || newH <= 0) return image;

        image.Mutate(ctx => ctx.Crop(new Rectangle(x1, y1, newW, newH)));
        return image;
    }

    /// <summary>
    /// HSV augmentation: randomly shift hue, scale saturation and value.
    /// Reference: ppocr/data/imaug/rec_img_aug.py hsv_aug
    /// </summary>
    private static Image<Rgb24> HsvAugment(Image<Rgb24> image, Random rng)
    {
        var hShift = (rng.NextSingle() * 2 - 1) * 0.1f * 360f;
        var sScale = 1f + (rng.NextSingle() * 2 - 1) * 0.3f;
        var vScale = 1f + (rng.NextSingle() * 2 - 1) * 0.3f;
        image.Mutate(x => x.Hue(hShift).Saturate(sScale).Brightness(vScale));
        return image;
    }

    /// <summary>
    /// Color jitter: random brightness, contrast, saturation.
    /// Reference: ppocr/data/imaug/rec_img_aug.py color_jitter
    /// </summary>
    private static Image<Rgb24> ColorJitter(Image<Rgb24> image, Random rng)
    {
        var brightness = 1.0f + (rng.NextSingle() * 2 - 1) * 0.3f;
        var contrast = 1.0f + (rng.NextSingle() * 2 - 1) * 0.3f;
        var saturation = 1.0f + (rng.NextSingle() * 2 - 1) * 0.3f;
        image.Mutate(x => x.Brightness(brightness).Contrast(contrast).Saturate(saturation));
        return image;
    }

    /// <summary>
    /// Add Gaussian noise.
    /// Reference: ppocr/data/imaug/rec_img_aug.py gaussian_noise
    /// </summary>
    private static Image<Rgb24> AddGaussianNoise(Image<Rgb24> image, Random rng)
    {
        var sigma = 10; // noise standard deviation
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                // Box-Muller transform for Gaussian noise
                var u1 = rng.NextSingle();
                var u2 = rng.NextSingle();
                var gaussian = MathF.Sqrt(-2f * MathF.Log(Math.Max(u1, 1e-10f))) * MathF.Cos(2f * MathF.PI * u2);
                var noise = (int)(gaussian * sigma);

                var r = (byte)Math.Clamp(pixel.R + noise, 0, 255);
                var g = (byte)Math.Clamp(pixel.G + noise, 0, 255);
                var b = (byte)Math.Clamp(pixel.B + noise, 0, 255);
                image[x, y] = new Rgb24(r, g, b);
            }
        }
        return image;
    }
}
