using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// Polygon-aware random crop for detection training.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/east_process.py EastRandomCropData.
/// 
/// Crops a random region from the image while trying to keep text regions.
/// Coordinates of polygons are adjusted relative to the crop region.
/// Polygons outside the crop area are filtered.
/// </summary>
public sealed class EastRandomCropData
{
    private readonly int _cropWidth;
    private readonly int _cropHeight;
    private readonly int _maxTries;
    private readonly bool _keepRatio;
    private readonly float _minCropSideRatio;

    /// <summary>
    /// Creates a new EastRandomCropData instance.
    /// </summary>
    /// <param name="cropWidth">Target crop width. Default: 640</param>
    /// <param name="cropHeight">Target crop height. Default: 640</param>
    /// <param name="maxTries">Maximum random crop attempts. Default: 50</param>
    /// <param name="keepRatio">Whether to keep aspect ratio. Default: true</param>
    /// <param name="minCropSideRatio">Minimum crop side ratio relative to image. Default: 0.1</param>
    public EastRandomCropData(
        int cropWidth = 640,
        int cropHeight = 640,
        int maxTries = 50,
        bool keepRatio = true,
        float minCropSideRatio = 0.1f)
    {
        _cropWidth = cropWidth;
        _cropHeight = cropHeight;
        _maxTries = maxTries;
        _keepRatio = keepRatio;
        _minCropSideRatio = minCropSideRatio;
    }

    /// <summary>
    /// Apply random crop to detection data.
    /// Returns the cropped data with adjusted polygon coordinates.
    /// </summary>
    public DetAugData? Apply(DetAugData data, Random rng)
    {
        var image = data.Image;
        var polys = data.Polys;
        var texts = data.Texts;
        var ignoreTags = data.IgnoreTags;

        if (image.Width <= 0 || image.Height <= 0)
        {
            return null;
        }

        var h = image.Height;
        var w = image.Width;

        // If image is smaller than crop size, pad it
        if (h < _cropHeight || w < _cropWidth)
        {
            var padH = Math.Max(_cropHeight - h, 0);
            var padW = Math.Max(_cropWidth - w, 0);
            var newH = h + padH;
            var newW = w + padW;
            var padded = new Image<Rgb24>(newW, newH, new Rgb24(0, 0, 0));
            padded.Mutate(ctx => ctx.DrawImage(image, new Point(0, 0), 1f));
            image.Dispose();
            image = padded;
            h = newH;
            w = newW;
        }

        // Find text region bounds
        var allX = new List<float>();
        var allY = new List<float>();
        foreach (var poly in polys)
        {
            foreach (var pt in poly)
            {
                allX.Add(pt.X);
                allY.Add(pt.Y);
            }
        }

        // Try random crops
        int cropX = 0, cropY = 0;
        var found = false;

        for (var t = 0; t < _maxTries; t++)
        {
            if (_keepRatio)
            {
                // Random crop maintaining target aspect ratio
                cropX = rng.Next(0, Math.Max(1, w - _cropWidth + 1));
                cropY = rng.Next(0, Math.Max(1, h - _cropHeight + 1));
            }
            else
            {
                var cropW = Math.Max((int)(_minCropSideRatio * w), _cropWidth);
                var cropH = Math.Max((int)(_minCropSideRatio * h), _cropHeight);
                cropW = Math.Min(cropW, w);
                cropH = Math.Min(cropH, h);
                cropX = rng.Next(0, Math.Max(1, w - cropW + 1));
                cropY = rng.Next(0, Math.Max(1, h - cropH + 1));
            }

            // Check if at least one non-ignored polygon is in the crop area
            var hasText = false;
            for (var i = 0; i < polys.Length; i++)
            {
                if (ignoreTags[i]) continue;
                if (IsPolyInCrop(polys[i], cropX, cropY, _cropWidth, _cropHeight))
                {
                    hasText = true;
                    break;
                }
            }

            if (hasText || polys.Length == 0)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            // Fall back to center crop
            cropX = Math.Max(0, (w - _cropWidth) / 2);
            cropY = Math.Max(0, (h - _cropHeight) / 2);
        }

        // Perform the crop
        var cropRect = new Rectangle(cropX, cropY, 
            Math.Min(_cropWidth, w - cropX), 
            Math.Min(_cropHeight, h - cropY));
        image.Mutate(x => x.Crop(cropRect));

        // Resize to target if needed
        if (image.Width != _cropWidth || image.Height != _cropHeight)
        {
            var scaleX = (float)_cropWidth / image.Width;
            var scaleY = (float)_cropHeight / image.Height;
            image.Mutate(x => x.Resize(_cropWidth, _cropHeight));

            // Adjust polygon coordinates
            var newPolys = new List<PointF[]>();
            var newTexts = new List<string>();
            var newIgnoreTags = new List<bool>();
            for (var i = 0; i < polys.Length; i++)
            {
                var adjusted = AdjustPolyToCrop(polys[i], cropX, cropY, scaleX, scaleY, _cropWidth, _cropHeight);
                if (adjusted is not null)
                {
                    newPolys.Add(adjusted);
                    newTexts.Add(texts[i]);
                    newIgnoreTags.Add(ignoreTags[i]);
                }
            }

            return new DetAugData
            {
                Image = image,
                Polys = newPolys.ToArray(),
                Texts = newTexts.ToArray(),
                IgnoreTags = newIgnoreTags.ToArray()
            };
        }
        else
        {
            // Adjust polygon coordinates (no resize needed)
            var newPolys = new List<PointF[]>();
            var newTexts = new List<string>();
            var newIgnoreTags = new List<bool>();
            for (var i = 0; i < polys.Length; i++)
            {
                var adjusted = AdjustPolyToCrop(polys[i], cropX, cropY, 1f, 1f, _cropWidth, _cropHeight);
                if (adjusted is not null)
                {
                    newPolys.Add(adjusted);
                    newTexts.Add(texts[i]);
                    newIgnoreTags.Add(ignoreTags[i]);
                }
            }

            return new DetAugData
            {
                Image = image,
                Polys = newPolys.ToArray(),
                Texts = newTexts.ToArray(),
                IgnoreTags = newIgnoreTags.ToArray()
            };
        }
    }

    /// <summary>
    /// Check if any point of the polygon falls within the crop region.
    /// </summary>
    private static bool IsPolyInCrop(PointF[] poly, int cropX, int cropY, int cropW, int cropH)
    {
        foreach (var pt in poly)
        {
            if (pt.X >= cropX && pt.X <= cropX + cropW &&
                pt.Y >= cropY && pt.Y <= cropY + cropH)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Adjust polygon coordinates relative to crop region and apply scaling.
    /// Returns null if polygon is completely outside the crop area.
    /// </summary>
    private static PointF[]? AdjustPolyToCrop(PointF[] poly, int cropX, int cropY, float scaleX, float scaleY, int cropW, int cropH)
    {
        var adjusted = new PointF[poly.Length];
        var anyInside = false;
        for (var j = 0; j < poly.Length; j++)
        {
            var nx = (poly[j].X - cropX) * scaleX;
            var ny = (poly[j].Y - cropY) * scaleY;
            // Clip to crop bounds
            nx = Math.Clamp(nx, 0, cropW - 1);
            ny = Math.Clamp(ny, 0, cropH - 1);
            adjusted[j] = new PointF(nx, ny);

            if (poly[j].X >= cropX && poly[j].X <= cropX + cropW &&
                poly[j].Y >= cropY && poly[j].Y <= cropY + cropH)
            {
                anyInside = true;
            }
        }

        return anyInside ? adjusted : null;
    }
}
