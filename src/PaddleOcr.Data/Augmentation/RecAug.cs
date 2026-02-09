using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Data.Augmentation;

/// <summary>
/// RecAug：Full recognition augmentation combining TIA transforms + BaseDataAugmentation.
/// 1:1 port of Python PaddleOCR ppocr/data/imaug/rec_img_aug.py RecAug.
/// 
/// Applies TIA transforms (distort, stretch, perspective) with tia_prob,
/// then BaseDataAugmentation operations.
/// </summary>
public sealed class RecAug
{
    private readonly float _tiaProb;
    private readonly BaseDataAugmentation _baseAug;

    public RecAug(
        float tiaProb = 0.4f,
        float cropProb = 0.4f,
        float reverseProb = 0.4f,
        float noiseProb = 0.4f,
        float jitterProb = 0.4f,
        float blurProb = 0.4f,
        float hsvAugProb = 0.4f)
    {
        _tiaProb = tiaProb;
        _baseAug = new BaseDataAugmentation(cropProb, reverseProb, noiseProb, jitterProb, blurProb, hsvAugProb);
    }

    /// <summary>
    /// Apply full RecAug augmentation.
    /// </summary>
    public Image<Rgb24> Apply(Image<Rgb24> image, Random? rng = null)
    {
        rng ??= Random.Shared;

        // Apply TIA transforms
        if (rng.NextSingle() < _tiaProb)
        {
            var tiaChoice = rng.Next(3);
            image = tiaChoice switch
            {
                0 => TiaTransforms.TiaDistort(image, rng),
                1 => TiaTransforms.TiaStretch(image, rng),
                _ => TiaTransforms.TiaPerspective(image, rng)
            };
        }

        // Apply base augmentation
        image = _baseAug.Apply(image, rng);
        return image;
    }
}
