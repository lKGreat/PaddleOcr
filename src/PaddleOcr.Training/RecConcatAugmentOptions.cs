namespace PaddleOcr.Training;

/// <summary>
/// Options for RecConAug-style concatenation augmentation.
/// </summary>
internal sealed record RecConcatAugmentOptions(bool Enabled, float Prob, int ExtDataNum, float MaxWhRatio, int ImageHeight)
{
    public static RecConcatAugmentOptions Disabled { get; } = new(false, 0f, 0, 0f, 0);

    public static RecConcatAugmentOptions FromConfig(TrainingConfigView cfg, int imageH, int imageW)
    {
        var hasTransform = cfg.HasTransform("Train.dataset.transforms", "RecConAug");
        var enabled = cfg.GetConfigBool("Train.dataset.concat_aug.enable", false) || hasTransform;
        if (!enabled)
        {
            return Disabled;
        }

        var prob = cfg.GetTransformFloat("RecConAug", "prob", cfg.GetConfigFloat("Train.dataset.concat_aug.prob", 0.5f));
        if (prob <= 0f)
        {
            prob = 0.5f;
        }

        var extDataNum = cfg.GetTransformInt("RecConAug", "ext_data_num", cfg.GetConfigInt("Train.dataset.concat_aug.ext_data_num", 1));
        if (extDataNum <= 0)
        {
            extDataNum = 1;
        }

        var imageShape = cfg.GetTransformIntArray("RecConAug", "image_shape");
        if (imageShape.Length >= 2 && imageShape[0] > 0 && imageShape[1] > 0)
        {
            // PaddleOCR RecConAug uses image_shape=[H,W,C], max_wh_ratio = W / H.
            imageH = imageShape[0];
            imageW = imageShape[1];
        }

        var defaultRatio = imageW <= 0 || imageH <= 0 ? 6.67f : (float)imageW / imageH;
        var maxWhRatio = defaultRatio;
        if (maxWhRatio <= 0f)
        {
            maxWhRatio = defaultRatio;
        }

        return new RecConcatAugmentOptions(true, prob, extDataNum, maxWhRatio, Math.Max(1, imageH));
    }
}
