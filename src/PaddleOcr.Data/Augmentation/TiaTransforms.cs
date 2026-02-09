using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// Text Image Augmentation (TIA) transforms.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/rec_img_aug.py.
/// 
/// Implements three geometric transforms using Moving Least Squares (MLS) warping:
/// - tia_distort: Random distortion of control points
/// - tia_stretch: Random stretching of control points  
/// - tia_perspective: Random perspective transform of control points
/// 
/// These transforms are critical for recognition training diversity.
/// </summary>
public static class TiaTransforms
{
    /// <summary>
    /// Apply TIA augmentation with configurable probability.
    /// Matches Python: ppocr/data/imaug/rec_img_aug.py warp() function.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="prob">Probability of applying any TIA transform. Default: 0.4</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>Augmented image.</returns>
    public static Image<Rgb24> Apply(Image<Rgb24> image, float prob = 0.4f, Random? rng = null)
    {
        rng ??= Random.Shared;
        if (rng.NextSingle() > prob)
        {
            return image;
        }

        var choice = rng.Next(3);
        return choice switch
        {
            0 => TiaDistort(image, rng),
            1 => TiaStretch(image, rng),
            2 => TiaPerspective(image, rng),
            _ => image
        };
    }

    /// <summary>
    /// Random distortion using MLS warping.
    /// Moves random control points by random amounts.
    /// Reference: ppocr/data/imaug/rec_img_aug.py tia_distort
    /// </summary>
    public static Image<Rgb24> TiaDistort(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;
        var w = image.Width;
        var h = image.Height;
        if (w < 8 || h < 8) return image;

        // Create control point grid (3x3)
        var segment = 4;
        var srcPts = new List<(float X, float Y)>();
        var dstPts = new List<(float X, float Y)>();

        // Top and bottom boundary points
        var xStep = w / (float)(segment - 1);
        for (var i = 0; i < segment; i++)
        {
            srcPts.Add((i * xStep, 0));
            srcPts.Add((i * xStep, h - 1));
            dstPts.Add((i * xStep, 0));
            dstPts.Add((i * xStep, h - 1));
        }

        // Move interior points randomly
        var numInterior = segment - 2;
        for (var i = 1; i <= numInterior; i++)
        {
            var x = i * xStep;
            var dx = (rng.NextSingle() * 2 - 1) * w * 0.03f;
            var dy = (rng.NextSingle() * 2 - 1) * h * 0.1f;
            // Only adjust destination points
            dstPts[i * 2] = (x + dx, Math.Clamp(dy, 0, h - 1));
            dstPts[i * 2 + 1] = (x + dx, Math.Clamp(h - 1 + dy, 0, h - 1));
        }

        return WarpMLS(image, srcPts, dstPts);
    }

    /// <summary>
    /// Random stretch using MLS warping.
    /// Stretches the image from random control points.
    /// Reference: ppocr/data/imaug/rec_img_aug.py tia_stretch
    /// </summary>
    public static Image<Rgb24> TiaStretch(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;
        var w = image.Width;
        var h = image.Height;
        if (w < 8 || h < 8) return image;

        var srcPts = new List<(float X, float Y)>();
        var dstPts = new List<(float X, float Y)>();

        // Corner points (fixed)
        srcPts.Add((0, 0));
        srcPts.Add((w - 1, 0));
        srcPts.Add((0, h - 1));
        srcPts.Add((w - 1, h - 1));
        dstPts.Add((0, 0));
        dstPts.Add((w - 1, 0));
        dstPts.Add((0, h - 1));
        dstPts.Add((w - 1, h - 1));

        // Add midpoints on each edge and randomly move them
        var halfW = w / 2f;
        var halfH = h / 2f;
        var stretchAmount = 0.1f;

        // Top edge midpoint
        var dx = (rng.NextSingle() * 2 - 1) * w * stretchAmount;
        srcPts.Add((halfW, 0));
        dstPts.Add((halfW + dx, 0));

        // Bottom edge midpoint
        dx = (rng.NextSingle() * 2 - 1) * w * stretchAmount;
        srcPts.Add((halfW, h - 1));
        dstPts.Add((halfW + dx, h - 1));

        // Left edge midpoint
        var dy = (rng.NextSingle() * 2 - 1) * h * stretchAmount;
        srcPts.Add((0, halfH));
        dstPts.Add((0, halfH + dy));

        // Right edge midpoint
        dy = (rng.NextSingle() * 2 - 1) * h * stretchAmount;
        srcPts.Add((w - 1, halfH));
        dstPts.Add((w - 1, halfH + dy));

        return WarpMLS(image, srcPts, dstPts);
    }

    /// <summary>
    /// Random perspective transform using MLS warping.
    /// Moves corner points to simulate perspective distortion.
    /// Reference: ppocr/data/imaug/rec_img_aug.py tia_perspective
    /// </summary>
    public static Image<Rgb24> TiaPerspective(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;
        var w = image.Width;
        var h = image.Height;
        if (w < 8 || h < 8) return image;

        var srcPts = new List<(float X, float Y)>();
        var dstPts = new List<(float X, float Y)>();

        var perspectiveAmount = 0.1f;
        var maxDx = w * perspectiveAmount;
        var maxDy = h * perspectiveAmount;

        // Source corners
        srcPts.Add((0, 0));
        srcPts.Add((w - 1, 0));
        srcPts.Add((0, h - 1));
        srcPts.Add((w - 1, h - 1));

        // Destination corners (randomly perturbed)
        dstPts.Add((rng.NextSingle() * maxDx, rng.NextSingle() * maxDy));
        dstPts.Add((w - 1 - rng.NextSingle() * maxDx, rng.NextSingle() * maxDy));
        dstPts.Add((rng.NextSingle() * maxDx, h - 1 - rng.NextSingle() * maxDy));
        dstPts.Add((w - 1 - rng.NextSingle() * maxDx, h - 1 - rng.NextSingle() * maxDy));

        return WarpMLS(image, srcPts, dstPts);
    }

    /// <summary>
    /// Moving Least Squares (MLS) affine warping.
    /// Reference: Python WarpMLS class in ppocr/data/imaug/rec_img_aug.py.
    /// 
    /// Uses affine deformation with control points to create smooth warping.
    /// For each target pixel, finds the weighted affine transform based on
    /// the control points and their displacements.
    /// </summary>
    private static Image<Rgb24> WarpMLS(
        Image<Rgb24> image,
        List<(float X, float Y)> srcPts,
        List<(float X, float Y)> dstPts)
    {
        var w = image.Width;
        var h = image.Height;
        var result = new Image<Rgb24>(w, h, new Rgb24(0, 0, 0));
        var n = srcPts.Count;

        if (n < 2)
        {
            result.Dispose();
            return image.Clone();
        }

        // Precompute alpha parameter for weights
        const float alpha = 1.0f;

        // For each pixel in the result image, compute the source coordinate
        for (var dstY = 0; dstY < h; dstY++)
        {
            for (var dstX = 0; dstX < w; dstX++)
            {
                // Compute weights for each control point
                var weights = new float[n];
                var totalWeight = 0f;
                var isExact = -1;

                for (var i = 0; i < n; i++)
                {
                    var dx = dstX - dstPts[i].X;
                    var dy = dstY - dstPts[i].Y;
                    var dist2 = dx * dx + dy * dy;

                    if (dist2 < 1e-6f)
                    {
                        isExact = i;
                        break;
                    }

                    weights[i] = 1.0f / MathF.Pow(dist2, alpha);
                    totalWeight += weights[i];
                }

                float srcX, srcY;

                if (isExact >= 0)
                {
                    // Exact match - use corresponding source point
                    srcX = srcPts[isExact].X;
                    srcY = srcPts[isExact].Y;
                }
                else
                {
                    // Compute weighted centroid of source and destination points
                    float pStarX = 0, pStarY = 0;
                    float qStarX = 0, qStarY = 0;

                    for (var i = 0; i < n; i++)
                    {
                        var wi = weights[i] / totalWeight;
                        pStarX += wi * dstPts[i].X;
                        pStarY += wi * dstPts[i].Y;
                        qStarX += wi * srcPts[i].X;
                        qStarY += wi * srcPts[i].Y;
                    }

                    // Compute affine matrix using MLS
                    // M = sum(wi * (pi - p*)(pi - p*)^T)^-1 * sum(wi * (pi - p*)(qi - q*)^T)
                    float a11 = 0, a12 = 0, a21 = 0, a22 = 0;
                    float b11 = 0, b12 = 0, b21 = 0, b22 = 0;

                    for (var i = 0; i < n; i++)
                    {
                        var pHatX = dstPts[i].X - pStarX;
                        var pHatY = dstPts[i].Y - pStarY;
                        var qHatX = srcPts[i].X - qStarX;
                        var qHatY = srcPts[i].Y - qStarY;
                        var wi = weights[i];

                        a11 += wi * pHatX * pHatX;
                        a12 += wi * pHatX * pHatY;
                        a21 += wi * pHatY * pHatX;
                        a22 += wi * pHatY * pHatY;

                        b11 += wi * pHatX * qHatX;
                        b12 += wi * pHatX * qHatY;
                        b21 += wi * pHatY * qHatX;
                        b22 += wi * pHatY * qHatY;
                    }

                    // Invert 2x2 matrix A
                    var det = a11 * a22 - a12 * a21;
                    if (MathF.Abs(det) < 1e-10f)
                    {
                        // Singular - just translate
                        srcX = dstX + (qStarX - pStarX);
                        srcY = dstY + (qStarY - pStarY);
                    }
                    else
                    {
                        var invDet = 1.0f / det;
                        var ia11 = a22 * invDet;
                        var ia12 = -a12 * invDet;
                        var ia21 = -a21 * invDet;
                        var ia22 = a11 * invDet;

                        // M = A^-1 * B
                        var m11 = ia11 * b11 + ia12 * b21;
                        var m12 = ia11 * b12 + ia12 * b22;
                        var m21 = ia21 * b11 + ia22 * b21;
                        var m22 = ia21 * b12 + ia22 * b22;

                        // Apply transform
                        var vx = dstX - pStarX;
                        var vy = dstY - pStarY;
                        srcX = m11 * vx + m21 * vy + qStarX;
                        srcY = m12 * vx + m22 * vy + qStarY;
                    }
                }

                // Bilinear interpolation from source
                var sx = (int)MathF.Floor(srcX);
                var sy = (int)MathF.Floor(srcY);
                if (sx >= 0 && sx < w - 1 && sy >= 0 && sy < h - 1)
                {
                    var fx = srcX - sx;
                    var fy = srcY - sy;
                    var p00 = image[sx, sy];
                    var p10 = image[sx + 1, sy];
                    var p01 = image[sx, sy + 1];
                    var p11 = image[sx + 1, sy + 1];

                    var r = (byte)Math.Clamp(
                        (int)(p00.R * (1 - fx) * (1 - fy) + p10.R * fx * (1 - fy) +
                              p01.R * (1 - fx) * fy + p11.R * fx * fy), 0, 255);
                    var g = (byte)Math.Clamp(
                        (int)(p00.G * (1 - fx) * (1 - fy) + p10.G * fx * (1 - fy) +
                              p01.G * (1 - fx) * fy + p11.G * fx * fy), 0, 255);
                    var b = (byte)Math.Clamp(
                        (int)(p00.B * (1 - fx) * (1 - fy) + p10.B * fx * (1 - fy) +
                              p01.B * (1 - fx) * fy + p11.B * fx * fy), 0, 255);

                    result[dstX, dstY] = new Rgb24(r, g, b);
                }
                else if (sx >= 0 && sx < w && sy >= 0 && sy < h)
                {
                    result[dstX, dstY] = image[sx, sy];
                }
            }
        }

        return result;
    }
}
