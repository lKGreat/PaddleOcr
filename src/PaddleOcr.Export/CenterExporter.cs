using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Training;
using PaddleOcr.Training.Rec;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Export;

/// <summary>
/// CenterExporter：从训练数据提取字符特征中心。
/// </summary>
public sealed class CenterExporter
{
    private readonly ILogger _logger;

    public CenterExporter(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// 导出字符特征中心。
    /// </summary>
    public string ExportCenter(
        RecModel model,
        IEnumerable<(float[] Images, long[] Labels, int Batch)> datasetBatches,
        int height,
        int width,
        int maxTextLength,
        IReadOnlyDictionary<char, int> charToId,
        IReadOnlyList<char> vocab,
        string outputPath,
        int batchSize = 32,
        Device? device = null)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? Directory.GetCurrentDirectory());

        device ??= cuda.is_available() ? CUDA : CPU;
        model.to(device);
        model.eval();

        // 按字符聚合特征
        var charFeatures = new Dictionary<int, List<float[]>>();
        var charCounts = new Dictionary<int, int>();

        using var noGrad = torch.no_grad();
        var rng = new Random(7);

        _logger.LogInformation("Extracting features for center export...");

        foreach (var (images, labels, batch) in datasetBatches)
        {
            using var x = torch.tensor(images, dtype: ScalarType.Float32)
                .reshape(batch, 3, height, width)
                .to(device);

            // 获取特征（假设模型支持 return_feats）
            var features = ExtractFeatures(model, x);

            // 处理每个样本
            var labelsFlat = labels;
            for (var i = 0; i < batch; i++)
            {
                var labelSeq = labelsFlat.Skip(i * maxTextLength).Take(maxTextLength).ToArray();
                var feat = features[i]; // [feature_dim]

                // 按字符聚合特征
                for (var j = 0; j < labelSeq.Length; j++)
                {
                    var charId = (int)labelSeq[j];
                    if (charId <= 0 || charId > vocab.Count)
                    {
                        continue;
                    }

                    if (!charFeatures.ContainsKey(charId))
                    {
                        charFeatures[charId] = new List<float[]>();
                        charCounts[charId] = 0;
                    }

                    // 提取对应位置的特征（简化：使用全局特征）
                    charFeatures[charId].Add(feat);
                    charCounts[charId]++;
                }
            }
        }

        // 计算每个字符的特征中心
        var centers = new Dictionary<int, float[]>();
        foreach (var (charId, feats) in charFeatures)
        {
            if (feats.Count == 0)
            {
                continue;
            }

            var dim = feats[0].Length;
            var center = new float[dim];
            foreach (var feat in feats)
            {
                for (var i = 0; i < dim; i++)
                {
                    center[i] += feat[i];
                }
            }

            for (var i = 0; i < dim; i++)
            {
                center[i] /= feats.Count;
            }

            centers[charId] = center;
        }

        // 保存中心
        var centerData = new CenterData
        {
            VocabSize = vocab.Count,
            FeatureDim = centers.Values.FirstOrDefault()?.Length ?? 0,
            Centers = centers,
            CharCounts = charCounts,
            GeneratedAtUtc = DateTime.UtcNow
        };

        var json = JsonSerializer.Serialize(centerData, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(outputPath, json);

        _logger.LogInformation("Exported center data: {Path}, chars={Chars}, dim={Dim}", outputPath, centers.Count, centerData.FeatureDim);
        return outputPath;
    }

    private float[][] ExtractFeatures(RecModel model, Tensor input)
    {
        var predictions = model.ForwardDict(input);

        // 优先使用 "features" 键（如果模型 head 显式输出特征）
        if (predictions.TryGetValue("features", out var features))
        {
            return TensorToBatchedVectors(features, input.shape[0]);
        }

        // 尝试使用 "predict" 之前的中间表示
        // 对于 CTC 类型的 head，predict 是 [B, T, V]，取时间维度平均得到 [B, V] 作为全局特征
        if (predictions.TryGetValue("predict", out var logits))
        {
            // [B, T, V] -> 对 T 维做平均 -> [B, V]（全局语义特征向量）
            if (logits.dim() == 3)
            {
                using var avgOverTime = logits.mean(new long[] { 1 }, keepdim: false);
                return TensorToBatchedVectors(avgOverTime, input.shape[0]);
            }

            // [B, V] 已是二维
            if (logits.dim() == 2)
            {
                return TensorToBatchedVectors(logits, input.shape[0]);
            }
        }

        // 如果有 "visual" 输出（VisionLAN 等），使用它
        if (predictions.TryGetValue("visual", out var visual))
        {
            if (visual.dim() == 3)
            {
                using var avgVisual = visual.mean(new long[] { 1 }, keepdim: false);
                return TensorToBatchedVectors(avgVisual, input.shape[0]);
            }

            return TensorToBatchedVectors(visual, input.shape[0]);
        }

        // 最终回退：取第一个输出
        var firstOutput = predictions.Values.First();
        if (firstOutput.dim() == 3)
        {
            using var avg = firstOutput.mean(new long[] { 1 }, keepdim: false);
            return TensorToBatchedVectors(avg, input.shape[0]);
        }

        return TensorToBatchedVectors(firstOutput, input.shape[0]);
    }

    private static float[][] TensorToBatchedVectors(Tensor tensor, long batchSize)
    {
        var data = tensor.cpu().data<float>().ToArray();
        var featDim = (int)(data.Length / batchSize);
        var result = new float[batchSize][];
        for (var i = 0; i < batchSize; i++)
        {
            result[i] = new float[featDim];
            Array.Copy(data, i * featDim, result[i], 0, featDim);
        }

        return result;
    }
}

/// <summary>
/// 中心数据格式。
/// </summary>
public sealed class CenterData
{
    public int VocabSize { get; set; }
    public int FeatureDim { get; set; }
    public Dictionary<int, float[]> Centers { get; set; } = new();
    public Dictionary<int, int> CharCounts { get; set; } = new();
    public DateTime GeneratedAtUtc { get; set; }
}
