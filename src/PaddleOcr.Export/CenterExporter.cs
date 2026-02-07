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
        // 简化实现：从模型的中间层提取特征
        // 实际实现需要根据具体模型架构调整
        var predictions = model.ForwardDict(input);
        if (predictions.TryGetValue("features", out var features))
        {
            var featData = features.cpu().data<float>().ToArray();
            var batchSize = input.shape[0];
            var featDim = featData.Length / batchSize;
            var result = new float[batchSize][];
            for (var i = 0; i < batchSize; i++)
            {
                result[i] = new float[featDim];
                Array.Copy(featData, i * featDim, result[i], 0, featDim);
            }

            return result;
        }

        // 回退：使用预测输出的平均值
        var logits = predictions["predict"];
        using var avg = logits.mean(new long[] { 1 }, keepdim: false); // [B, T, V] -> [B, V]
        var avgData = avg.cpu().data<float>().ToArray();
        var inputBatchSize = input.shape[0];
        var vocabSize = avgData.Length / inputBatchSize;
        var fallback = new float[inputBatchSize][];
        for (var i = 0; i < inputBatchSize; i++)
        {
            fallback[i] = new float[vocabSize];
            Array.Copy(avgData, i * vocabSize, fallback[i], 0, vocabSize);
        }

        return fallback;
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
