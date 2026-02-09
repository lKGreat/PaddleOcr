using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// Recognition concatenation augmentation.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/rec_img_aug.py RecConAug.
/// 
/// Concatenates multiple text images horizontally to create longer text sequences.
/// This increases text diversity and helps training generalize to longer texts.
/// </summary>
public sealed class RecConAug
{
    private readonly float _prob;
    private readonly int _extDataNum;
    private readonly int _imageH;
    private readonly int _imageW;
    private readonly int _maxTextLength;

    /// <summary>
    /// Creates a new RecConAug instance.
    /// </summary>
    /// <param name="prob">Probability of applying augmentation. Default: 0.5</param>
    /// <param name="extDataNum">Number of extra images to concatenate. Default: 2</param>
    /// <param name="imageH">Target image height. Default: 48</param>
    /// <param name="imageW">Target image width. Default: 320</param>
    /// <param name="maxTextLength">Maximum text length. Default: 25</param>
    public RecConAug(
        float prob = 0.5f,
        int extDataNum = 2,
        int imageH = 48,
        int imageW = 320,
        int maxTextLength = 25)
    {
        _prob = prob;
        _extDataNum = extDataNum;
        _imageH = imageH;
        _imageW = imageW;
        _maxTextLength = maxTextLength;
    }

    /// <summary>
    /// Apply concatenation augmentation.
    /// Concatenates the source image with external images horizontally.
    /// </summary>
    /// <param name="srcImage">Source image.</param>
    /// <param name="srcText">Source text label.</param>
    /// <param name="extImages">External images to concatenate.</param>
    /// <param name="extTexts">External text labels.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>Tuple of (concatenated image, concatenated text), or original if skipped.</returns>
    public (Image<Rgb24> Image, string Text) Apply(
        Image<Rgb24> srcImage,
        string srcText,
        IReadOnlyList<Image<Rgb24>> extImages,
        IReadOnlyList<string> extTexts,
        Random rng)
    {
        // Check probability
        if (rng.NextSingle() > _prob)
        {
            return (srcImage, srcText);
        }

        // Resize source to target height, maintaining aspect ratio
        var images = new List<Image<Rgb24>>();
        var texts = new List<string> { srcText };

        var srcResized = ResizeToHeight(srcImage, _imageH);
        images.Add(srcResized);

        // Add external images
        var numToConcat = Math.Min(_extDataNum, extImages.Count);
        for (var i = 0; i < numToConcat; i++)
        {
            var extResized = ResizeToHeight(extImages[i].Clone(), _imageH);
            images.Add(extResized);
            texts.Add(extTexts[i]);
        }

        // Concatenate text
        var concatText = string.Concat(texts);

        // Check text length constraint
        if (concatText.Length > _maxTextLength)
        {
            // Trim and only use what fits
            concatText = concatText[.._maxTextLength];
        }

        // Concatenate images horizontally
        var totalWidth = images.Sum(img => img.Width);
        var targetW = Math.Min(totalWidth, _imageW);

        var result = new Image<Rgb24>(totalWidth, _imageH, new Rgb24(0, 0, 0));
        var offsetX = 0;
        foreach (var img in images)
        {
            result.Mutate(ctx => ctx.DrawImage(img, new Point(offsetX, 0), 1f));
            offsetX += img.Width;
            if (img != srcResized) // Don't dispose the original
            {
                img.Dispose();
            }
        }

        // Resize to target width if wider
        if (result.Width > _imageW)
        {
            result.Mutate(x => x.Resize(_imageW, _imageH));
        }

        // Dispose the resized source if it was cloned
        if (srcResized != srcImage)
        {
            srcResized.Dispose();
        }

        return (result, concatText);
    }

    /// <summary>
    /// Resize image to target height while maintaining aspect ratio.
    /// </summary>
    private static Image<Rgb24> ResizeToHeight(Image<Rgb24> image, int targetH)
    {
        if (image.Height == targetH)
        {
            return image;
        }

        var scale = (float)targetH / image.Height;
        var newW = Math.Max(1, (int)(image.Width * scale));
        var result = image.Clone(x => x.Resize(newW, targetH));
        return result;
    }
}
