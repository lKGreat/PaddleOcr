using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// Detection augmentation matching Python PaddleOCR ppocr/data/imaug/iaa_augment.py IaaAugment.
/// Applies: Fliplr (horizontal flip), Affine (rotation), Resize (scale).
/// Transforms both image and polygon annotations consistently.
/// </summary>
public sealed class IaaAugment
{
    private readonly float _flipProb;
    private readonly float _rotateMin;
    private readonly float _rotateMax;
    private readonly float _scaleMin;
    private readonly float _scaleMax;

    public IaaAugment(
        float flipProb = 0.5f,
        float rotateMin = -10f,
        float rotateMax = 10f,
        float scaleMin = 0.5f,
        float scaleMax = 3.0f)
    {
        _flipProb = flipProb;
        _rotateMin = rotateMin;
        _rotateMax = rotateMax;
        _scaleMin = scaleMin;
        _scaleMax = scaleMax;
    }

    /// <summary>
    /// Apply augmentation to image and polygons.
    /// Returns null if augmentation fails.
    /// </summary>
    public DetAugData? Apply(DetAugData data, Random rng)
    {
        var image = data.Image;
        var polys = data.Polys;

        // 1. Fliplr (horizontal flip)
        if (rng.NextSingle() < _flipProb)
        {
            image.Mutate(x => x.Flip(FlipMode.Horizontal));
            var w = image.Width;
            for (var i = 0; i < polys.Length; i++)
            {
                for (var j = 0; j < polys[i].Length; j++)
                {
                    polys[i][j] = new PointF(w - polys[i][j].X, polys[i][j].Y);
                }
            }
        }

        // 2. Affine rotation
        var angle = _rotateMin + rng.NextSingle() * (_rotateMax - _rotateMin);
        if (MathF.Abs(angle) > 0.1f)
        {
            var cx = image.Width / 2f;
            var cy = image.Height / 2f;
            var rad = angle * MathF.PI / 180f;
            var cos = MathF.Cos(rad);
            var sin = MathF.Sin(rad);

            image.Mutate(x => x.Rotate(angle));

            // Transform polygon points using rotation matrix
            var newCx = image.Width / 2f;
            var newCy = image.Height / 2f;
            for (var i = 0; i < polys.Length; i++)
            {
                for (var j = 0; j < polys[i].Length; j++)
                {
                    var px = polys[i][j].X - cx;
                    var py = polys[i][j].Y - cy;
                    var nx = px * cos - py * sin + newCx;
                    var ny = px * sin + py * cos + newCy;
                    polys[i][j] = new PointF(nx, ny);
                }
            }
        }

        // 3. Resize (random scale)
        var scale = _scaleMin + rng.NextSingle() * (_scaleMax - _scaleMin);
        if (MathF.Abs(scale - 1f) > 0.01f)
        {
            var newW = Math.Max(1, (int)(image.Width * scale));
            var newH = Math.Max(1, (int)(image.Height * scale));
            image.Mutate(x => x.Resize(newW, newH));

            for (var i = 0; i < polys.Length; i++)
            {
                for (var j = 0; j < polys[i].Length; j++)
                {
                    polys[i][j] = new PointF(polys[i][j].X * scale, polys[i][j].Y * scale);
                }
            }
        }

        return data with { Image = image, Polys = polys };
    }
}

/// <summary>
/// Data container for detection augmentation pipeline.
/// Matches Python PaddleOCR data dictionary format.
/// </summary>
public record DetAugData
{
    public required Image<Rgb24> Image { get; init; }
    /// <summary>Polygons: [N][M] points, each polygon has M (x,y) points.</summary>
    public required PointF[][] Polys { get; init; }
    /// <summary>Text transcriptions per polygon.</summary>
    public required string[] Texts { get; init; }
    /// <summary>Ignore tags per polygon (true = ignore, e.g. "###").</summary>
    public required bool[] IgnoreTags { get; init; }
}
