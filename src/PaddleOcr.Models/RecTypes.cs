namespace PaddleOcr.Models;

/// <summary>
/// 识别推理结果。
/// </summary>
public sealed record RecResult(string Text, float Score);

/// <summary>
/// 识别评估指标。
/// </summary>
public sealed record RecEvalMetrics(
    float Accuracy,
    float CharacterAccuracy,
    float AvgEditDistance);

/// <summary>
/// 识别推理选项。
/// </summary>
public sealed record RecInferenceOptions(
    string ImageDir,
    string RecModelPath,
    string OutputDir,
    string? RecCharDictPath,
    bool UseSpaceChar,
    float DropScore,
    RecAlgorithm RecAlgorithm,
    string RecImageShape,
    int RecBatchNum,
    int MaxTextLength,
    bool RecImageInverse,
    bool ReturnWordBox)
{
    /// <summary>
    /// 解析图像形状字符串为 (C, H, W)。
    /// </summary>
    public (int C, int H, int W) ParseImageShape()
    {
        var parts = RecImageShape.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length == 3 &&
            int.TryParse(parts[0], out var c) &&
            int.TryParse(parts[1], out var h) &&
            int.TryParse(parts[2], out var w))
        {
            return (c, h, w);
        }

        return RecAlgorithm.GetDefaultImageShape();
    }
}
