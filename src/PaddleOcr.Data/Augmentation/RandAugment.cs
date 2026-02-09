using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// RandAugment implementation for classification training.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/randaugment.py RandAugment.
/// 
/// Randomly selects N operations from 14 available and applies them with magnitude M.
/// Operations: shearX, shearY, translateX, translateY, rotate, color, posterize,
///             solarize, contrast, sharpness, brightness, autocontrast, equalize, invert
/// </summary>
public sealed class RandAugment
{
    private readonly float _prob;
    private readonly int _numLayers;
    private readonly int _magnitude;

    private delegate Image<Rgb24> AugOperation(Image<Rgb24> image, float magnitude, Random rng);
    private readonly AugOperation[] _operations;

    /// <summary>
    /// Creates a new RandAugment instance.
    /// </summary>
    /// <param name="prob">Probability of applying RandAugment. Default: 0.5</param>
    /// <param name="numLayers">Number of operations to apply. Default: 2</param>
    /// <param name="magnitude">Magnitude of operations (0-10). Default: 5</param>
    public RandAugment(float prob = 0.5f, int numLayers = 2, int magnitude = 5)
    {
        _prob = prob;
        _numLayers = numLayers;
        _magnitude = magnitude;

        _operations = new AugOperation[]
        {
            ShearX,
            ShearY,
            TranslateX,
            TranslateY,
            Rotate,
            ColorOp,
            Posterize,
            Solarize,
            ContrastOp,
            Sharpness,
            Brightness,
            AutoContrast,
            Equalize,
            Invert
        };
    }

    /// <summary>
    /// Apply RandAugment: randomly select and apply N operations.
    /// </summary>
    public Image<Rgb24> Apply(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;

        if (rng.NextSingle() > _prob)
        {
            return image;
        }

        var magnitude = (float)_magnitude / 10f; // Normalize to [0, 1]

        for (var i = 0; i < _numLayers; i++)
        {
            var opIdx = rng.Next(_operations.Length);
            image = _operations[opIdx](image, magnitude, rng);
        }

        return image;
    }

    // ---- 14 Operations (matching Python PaddleOCR) ----

    private static Image<Rgb24> ShearX(Image<Rgb24> image, float magnitude, Random rng)
    {
        var shear = (rng.NextSingle() * 2 - 1) * magnitude * 0.3f; // max 30% shear at mag=10
        var builder = new AffineTransformBuilder().AppendSkewDegrees(shear * 45f, 0);
        image.Mutate(x => x.Transform(builder));
        return image;
    }

    private static Image<Rgb24> ShearY(Image<Rgb24> image, float magnitude, Random rng)
    {
        var shear = (rng.NextSingle() * 2 - 1) * magnitude * 0.3f;
        var builder = new AffineTransformBuilder().AppendSkewDegrees(0, shear * 45f);
        image.Mutate(x => x.Transform(builder));
        return image;
    }

    private static Image<Rgb24> TranslateX(Image<Rgb24> image, float magnitude, Random rng)
    {
        var maxPixels = (int)(image.Width * magnitude * 0.3f);
        var dx = rng.Next(-maxPixels, maxPixels + 1);
        var builder = new AffineTransformBuilder().AppendTranslation(new System.Numerics.Vector2(dx, 0));
        image.Mutate(x => x.Transform(builder));
        return image;
    }

    private static Image<Rgb24> TranslateY(Image<Rgb24> image, float magnitude, Random rng)
    {
        var maxPixels = (int)(image.Height * magnitude * 0.3f);
        var dy = rng.Next(-maxPixels, maxPixels + 1);
        var builder = new AffineTransformBuilder().AppendTranslation(new System.Numerics.Vector2(0, dy));
        image.Mutate(x => x.Transform(builder));
        return image;
    }

    private static Image<Rgb24> Rotate(Image<Rgb24> image, float magnitude, Random rng)
    {
        var maxAngle = magnitude * 30f; // max 30 degrees at mag=10
        var angle = (rng.NextSingle() * 2 - 1) * maxAngle;
        image.Mutate(x => x.Rotate(angle));
        return image;
    }

    private static Image<Rgb24> ColorOp(Image<Rgb24> image, float magnitude, Random rng)
    {
        // Blend with grayscale version
        var factor = magnitude * (rng.NextSingle() * 2 - 1);
        var saturation = 1f + factor;
        image.Mutate(x => x.Saturate(Math.Max(0, saturation)));
        return image;
    }

    private static Image<Rgb24> Posterize(Image<Rgb24> image, float magnitude, Random rng)
    {
        // Reduce bits per channel (4-8 bits)
        var bits = Math.Max(1, 8 - (int)(magnitude * 4));
        var mask = (byte)(0xFF << (8 - bits));
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                image[x, y] = new Rgb24(
                    (byte)(pixel.R & mask),
                    (byte)(pixel.G & mask),
                    (byte)(pixel.B & mask));
            }
        }
        return image;
    }

    private static Image<Rgb24> Solarize(Image<Rgb24> image, float magnitude, Random rng)
    {
        // Invert pixels above threshold
        var threshold = (byte)(255 * (1 - magnitude));
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                image[x, y] = new Rgb24(
                    pixel.R >= threshold ? (byte)(255 - pixel.R) : pixel.R,
                    pixel.G >= threshold ? (byte)(255 - pixel.G) : pixel.G,
                    pixel.B >= threshold ? (byte)(255 - pixel.B) : pixel.B);
            }
        }
        return image;
    }

    private static Image<Rgb24> ContrastOp(Image<Rgb24> image, float magnitude, Random rng)
    {
        var factor = 1f + magnitude * (rng.NextSingle() * 2 - 1);
        image.Mutate(x => x.Contrast(Math.Max(0, factor)));
        return image;
    }

    private static Image<Rgb24> Sharpness(Image<Rgb24> image, float magnitude, Random rng)
    {
        var factor = 1f + magnitude * (rng.NextSingle() * 2 - 1);
        if (factor > 1f)
        {
            image.Mutate(x => x.GaussianSharpen(factor));
        }
        else if (factor < 1f)
        {
            image.Mutate(x => x.GaussianBlur(2f - factor));
        }
        return image;
    }

    private static Image<Rgb24> Brightness(Image<Rgb24> image, float magnitude, Random rng)
    {
        var factor = 1f + magnitude * (rng.NextSingle() * 2 - 1);
        image.Mutate(x => x.Brightness(Math.Max(0, factor)));
        return image;
    }

    private static Image<Rgb24> AutoContrast(Image<Rgb24> image, float magnitude, Random rng)
    {
        // Stretch histogram to full range per channel
        byte minR = 255, maxR = 0;
        byte minG = 255, maxG = 0;
        byte minB = 255, maxB = 0;
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                if (p.R < minR) minR = p.R;
                if (p.R > maxR) maxR = p.R;
                if (p.G < minG) minG = p.G;
                if (p.G > maxG) maxG = p.G;
                if (p.B < minB) minB = p.B;
                if (p.B > maxB) maxB = p.B;
            }
        }

        var rangeR = maxR - minR;
        var rangeG = maxG - minG;
        var rangeB = maxB - minB;
        if (rangeR == 0 && rangeG == 0 && rangeB == 0) return image;

        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                var r = rangeR > 0 ? (byte)Math.Clamp((p.R - minR) * 255 / rangeR, 0, 255) : p.R;
                var g = rangeG > 0 ? (byte)Math.Clamp((p.G - minG) * 255 / rangeG, 0, 255) : p.G;
                var b = rangeB > 0 ? (byte)Math.Clamp((p.B - minB) * 255 / rangeB, 0, 255) : p.B;
                image[x, y] = new Rgb24(r, g, b);
            }
        }
        return image;
    }

    private static Image<Rgb24> Equalize(Image<Rgb24> image, float magnitude, Random rng)
    {
        // Histogram equalization per channel
        var histR = new int[256];
        var histG = new int[256];
        var histB = new int[256];
        var totalPixels = image.Width * image.Height;

        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                histR[p.R]++;
                histG[p.G]++;
                histB[p.B]++;
            }
        }

        var lutR = BuildEqualizeLut(histR, totalPixels);
        var lutG = BuildEqualizeLut(histG, totalPixels);
        var lutB = BuildEqualizeLut(histB, totalPixels);

        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var p = image[x, y];
                image[x, y] = new Rgb24(lutR[p.R], lutG[p.G], lutB[p.B]);
            }
        }
        return image;
    }

    private static byte[] BuildEqualizeLut(int[] histogram, int totalPixels)
    {
        var lut = new byte[256];
        var cumSum = 0;
        for (var i = 0; i < 256; i++)
        {
            cumSum += histogram[i];
            lut[i] = (byte)Math.Clamp(cumSum * 255 / totalPixels, 0, 255);
        }
        return lut;
    }

    private static Image<Rgb24> Invert(Image<Rgb24> image, float magnitude, Random rng)
    {
        image.Mutate(x => x.Invert());
        return image;
    }
}
