namespace PaddleOcr.Training;

/// <summary>
/// Options for RecConAug-style concatenation augmentation (PP-OCRv5).
/// </summary>
internal sealed record RecConcatAugmentOptions(bool Enabled, float Prob, int ExtDataNum, float MaxWhRatio)
{
    public static RecConcatAugmentOptions Disabled { get; } = new(false, 0f, 0, 0f);

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

        var extDataNum = cfg.GetTransformInt("RecConAug", "ext_data_num", cfg.GetConfigInt("Train.dataset.concat_aug.ext_data_num", 2));
        if (extDataNum <= 0)
        {
            extDataNum = 1;
        }

        var defaultRatio = imageW <= 0 || imageH <= 0 ? 6.67f : (float)imageW / imageH;
        var maxWhRatio = cfg.GetTransformFloat("RecConAug", "max_wh_ratio", cfg.GetConfigFloat("Train.dataset.concat_aug.max_wh_ratio", defaultRatio));
        if (maxWhRatio <= 0f)
        {
            maxWhRatio = defaultRatio;
        }

        return new RecConcatAugmentOptions(true, prob, extDataNum, maxWhRatio);
    }
}

