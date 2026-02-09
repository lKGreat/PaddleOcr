using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// Copy-Paste augmentation for detection training.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/copy_paste.py CopyPaste.
/// 
/// Extracts text regions from external data and pastes them onto the source image.
/// This helps increase text density and diversity during training.
/// </summary>
public sealed class CopyPaste
{
    private readonly float _objectsPasteRatio;
    private readonly bool _limitPaste;
    private readonly int _maxPaste;

    /// <summary>
    /// Creates a new CopyPaste instance.
    /// </summary>
    /// <param name="objectsPasteRatio">Ratio of polygons to paste. Default: 0.2</param>
    /// <param name="limitPaste">Whether to limit paste to avoid overlap. Default: true</param>
    /// <param name="maxPaste">Maximum number of regions to paste. Default: 30</param>
    public CopyPaste(
        float objectsPasteRatio = 0.2f,
        bool limitPaste = true,
        int maxPaste = 30)
    {
        _objectsPasteRatio = objectsPasteRatio;
        _limitPaste = limitPaste;
        _maxPaste = maxPaste;
    }

    /// <summary>
    /// Apply copy-paste augmentation: paste text regions from extData onto srcData.
    /// </summary>
    public DetAugData Apply(DetAugData srcData, DetAugData extData, Random rng)
    {
        var srcImage = srcData.Image;
        var srcPolys = new List<PointF[]>(srcData.Polys);
        var srcTexts = new List<string>(srcData.Texts);
        var srcIgnoreTags = new List<bool>(srcData.IgnoreTags);

        var extImage = extData.Image;
        var extPolys = extData.Polys;
        var extTexts = extData.Texts;
        var extIgnoreTags = extData.IgnoreTags;

        // Select polygons to paste (prefer non-ignored ones)
        var candidates = new List<int>();
        for (var i = 0; i < extIgnoreTags.Length; i++)
        {
            if (!extIgnoreTags[i])
            {
                candidates.Add(i);
            }
        }

        if (candidates.Count == 0)
        {
            return srcData;
        }

        var selectNum = Math.Max(1, Math.Min((int)(_objectsPasteRatio * extPolys.Length), _maxPaste));
        selectNum = Math.Min(selectNum, candidates.Count);

        // Shuffle and select
        for (var i = candidates.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
        }

        var srcH = srcImage.Height;
        var srcW = srcImage.Width;

        for (var s = 0; s < selectNum; s++)
        {
            var idx = candidates[s];
            var poly = extPolys[idx];

            // Get bounding box of polygon in ext image
            var minX = float.MaxValue;
            var minY = float.MaxValue;
            var maxX = float.MinValue;
            var maxY = float.MinValue;
            foreach (var pt in poly)
            {
                minX = Math.Min(minX, pt.X);
                minY = Math.Min(minY, pt.Y);
                maxX = Math.Max(maxX, pt.X);
                maxY = Math.Max(maxY, pt.Y);
            }

            var boxW = (int)(maxX - minX);
            var boxH = (int)(maxY - minY);
            if (boxW < 2 || boxH < 2) continue;

            // Clip bounds to ext image
            var cx1 = Math.Max(0, (int)minX);
            var cy1 = Math.Max(0, (int)minY);
            var cx2 = Math.Min(extImage.Width, cx1 + boxW);
            var cy2 = Math.Min(extImage.Height, cy1 + boxH);
            if (cx2 <= cx1 || cy2 <= cy1) continue;

            // Extract the crop from ext image
            using var crop = extImage.Clone(x => x.Crop(new Rectangle(cx1, cy1, cx2 - cx1, cy2 - cy1)));

            // Find paste location in src image
            var pasteX = rng.Next(0, Math.Max(1, srcW - crop.Width));
            var pasteY = rng.Next(0, Math.Max(1, srcH - crop.Height));

            // Check overlap with existing polygons if limit_paste
            if (_limitPaste)
            {
                var pasteRect = new RectangleF(pasteX, pasteY, crop.Width, crop.Height);
                var hasOverlap = false;
                foreach (var existingPoly in srcPolys)
                {
                    if (PolyOverlapsRect(existingPoly, pasteRect))
                    {
                        hasOverlap = true;
                        break;
                    }
                }
                if (hasOverlap) continue;
            }

            // Paste the crop onto src image
            srcImage.Mutate(ctx => ctx.DrawImage(crop, new Point(pasteX, pasteY), 1f));

            // Add new polygon with adjusted coordinates
            var newPoly = new PointF[poly.Length];
            for (var j = 0; j < poly.Length; j++)
            {
                newPoly[j] = new PointF(
                    Math.Clamp(poly[j].X - minX + pasteX, 0, srcW - 1),
                    Math.Clamp(poly[j].Y - minY + pasteY, 0, srcH - 1));
            }

            srcPolys.Add(newPoly);
            srcTexts.Add(extTexts[idx]);
            srcIgnoreTags.Add(extIgnoreTags[idx]);
        }

        return new DetAugData
        {
            Image = srcImage,
            Polys = srcPolys.ToArray(),
            Texts = srcTexts.ToArray(),
            IgnoreTags = srcIgnoreTags.ToArray()
        };
    }

    /// <summary>
    /// Check if a polygon bounding box overlaps with a rectangle.
    /// </summary>
    private static bool PolyOverlapsRect(PointF[] poly, RectangleF rect)
    {
        var minX = float.MaxValue;
        var minY = float.MaxValue;
        var maxX = float.MinValue;
        var maxY = float.MinValue;
        foreach (var pt in poly)
        {
            minX = Math.Min(minX, pt.X);
            minY = Math.Min(minY, pt.Y);
            maxX = Math.Max(maxX, pt.X);
            maxY = Math.Max(maxY, pt.Y);
        }

        var polyRect = new RectangleF(minX, minY, maxX - minX, maxY - minY);
        return polyRect.IntersectsWith(rect);
    }
}
