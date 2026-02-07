using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Training.Rec;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Export;

/// <summary>
/// OnnxModelExporter：从 TorchSharp 训练模型直接导出 ONNX。
/// </summary>
public sealed class OnnxModelExporter
{
    private readonly ILogger _logger;

    public OnnxModelExporter(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// 从 TorchSharp 模型导出 ONNX。
    /// </summary>
    public string ExportOnnx(
        RecModel model,
        string outputPath,
        int batchSize = 1,
        int channels = 3,
        int height = 48,
        int width = 320,
        bool dynamicBatch = false,
        bool dynamicSequence = false)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? Directory.GetCurrentDirectory());

        model.eval();
        using var noGrad = torch.no_grad();

        // 创建示例输入
        var inputShape = dynamicBatch
            ? new long[] { -1, channels, height, width }
            : new long[] { batchSize, channels, height, width };

        using var dummyInput = torch.randn(inputShape, dtype: ScalarType.Float32);

        try
        {
            // 导出 ONNX（TorchSharp 的 ONNX 导出 API）
            // 注意：TorchSharp 的 ONNX 导出功能可能有限
            // 这里使用简化实现，实际使用时需要根据 TorchSharp 版本调整
            // 暂时保存为 TorchScript 格式，需要手动转换为 ONNX
            model.save(outputPath.Replace(".onnx", ".pt"));

            _logger.LogWarning("ONNX export not fully supported. Saved TorchScript model to {Path}. Manual conversion to ONNX may be required.", outputPath.Replace(".onnx", ".pt"));
            _logger.LogInformation("Exported model (TorchScript): {Path}", outputPath.Replace(".onnx", ".pt"));
            return outputPath.Replace(".onnx", ".pt");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to export ONNX model");
            throw;
        }
    }

    /// <summary>
    /// 从配置导出 ONNX。
    /// </summary>
    public string ExportOnnxFromConfig(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);

        // 加载模型
        var ckpt = ResolveCheckpoint(cfg);
        if (!File.Exists(ckpt))
        {
            throw new FileNotFoundException($"checkpoint not found: {ckpt}");
        }

        // 构建模型（需要从配置读取架构信息）
        var numClasses = EstimateNumClasses(cfg);
        var model = BuildModelFromConfig(cfg, numClasses);
        model.load(ckpt);

        var dev = cuda.is_available() ? CUDA : CPU;
        model.to(dev);

        // 解析图像尺寸
        var (c, h, w) = ParseImageShape(cfg);

        var outputPath = Path.Combine(cfg.SaveInferenceDir, "inference.onnx");
        ExportOnnx(model, outputPath, batchSize: 1, channels: c, height: h, width: w, dynamicBatch: true, dynamicSequence: true);

        // 生成 manifest
        var io = GetOnnxIoMetadata(outputPath);
        var manifest = BuildManifest(cfg, "onnx", DateTime.UtcNow, "inference.onnx", ckpt, null, null, io.Inputs, io.Outputs);
        WriteManifest(cfg.SaveInferenceDir, manifest);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);

        model.Dispose();
        return outputPath;
    }

    private RecModel BuildModelFromConfig(ExportConfigView cfg, int numClasses)
    {
        // 从配置读取架构信息
        var backboneName = GetConfigString(cfg, "Architecture.Backbone.name", "MobileNetV1Enhance");
        var neckName = GetConfigString(cfg, "Architecture.Neck.name", "SequenceEncoder");
        var headName = GetConfigString(cfg, "Architecture.Head.name", "CTCHead");
        var hiddenSize = GetConfigInt(cfg, "Architecture.Head.hidden_size", 48);
        var maxLen = GetConfigInt(cfg, "Global.max_text_length", 25);
        var inChannels = GetConfigInt(cfg, "Architecture.in_channels", 3);

        return RecModelBuilder.Build(backboneName, neckName, headName, numClasses, inChannels, hiddenSize, maxLen);
    }

    private int EstimateNumClasses(ExportConfigView cfg)
    {
        // 从字典文件估算类别数
        if (!string.IsNullOrWhiteSpace(cfg.RecCharDictPath) && File.Exists(cfg.RecCharDictPath))
        {
            var lines = File.ReadAllLines(cfg.RecCharDictPath);
            return lines.Length + 1; // +1 for blank/PAD
        }

        // 默认值
        return 6625; // 常见的中文字典大小
    }

    private (int C, int H, int W) ParseImageShape(ExportConfigView cfg)
    {
        // 尝试从配置读取图像尺寸
        var shape = GetConfigIntList(cfg, "Train.dataset.transforms", "RecResizeImg", "image_shape");
        if (shape.Count >= 3)
        {
            return (shape[0], shape[1], shape[2]);
        }

        return (3, 48, 320);
    }

    private string GetConfigString(ExportConfigView cfg, string path, string fallback)
    {
        // 简化实现：从配置根字典读取
        return fallback; // TODO: 实现配置读取
    }

    private int GetConfigInt(ExportConfigView cfg, string path, int fallback)
    {
        return fallback; // TODO: 实现配置读取
    }

    private List<int> GetConfigIntList(ExportConfigView cfg, string transformsPath, string opName, string field)
    {
        return [3, 48, 320]; // TODO: 实现配置读取
    }

    private static string ResolveCheckpoint(ExportConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints))
        {
            return cfg.Checkpoints;
        }

        var best = Path.Combine(cfg.SaveModelDir, "best.pt");
        if (File.Exists(best))
        {
            return best;
        }

        var latest = Path.Combine(cfg.SaveModelDir, "latest.pt");
        if (File.Exists(latest))
        {
            return latest;
        }

        return best;
    }

    private static (IReadOnlyList<ExportTensorInfo> Inputs, IReadOnlyList<ExportTensorInfo> Outputs) GetOnnxIoMetadata(string onnxPath)
    {
        using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(onnxPath);
        var inputs = session.InputMetadata
            .Select(kv => new ExportTensorInfo(kv.Key, kv.Value.Dimensions.ToArray()))
            .ToList();
        var outputs = session.OutputMetadata
            .Select(kv => new ExportTensorInfo(kv.Key, kv.Value.Dimensions.ToArray()))
            .ToList();
        return (inputs, outputs);
    }

    private static ExportManifest BuildManifest(
        ExportConfigView cfg,
        string format,
        DateTime createdAtUtc,
        string artifactFile,
        string? checkpoint,
        string? source,
        string? sourceDirectory,
        IReadOnlyList<ExportTensorInfo>? onnxInputs,
        IReadOnlyList<ExportTensorInfo>? onnxOutputs)
    {
        return new ExportManifest(
            SchemaVersion: "1.0",
            Format: format,
            ModelType: cfg.ModelType,
            CreatedAtUtc: createdAtUtc,
            ArtifactFile: artifactFile,
            Checkpoint: checkpoint,
            Source: source,
            SourceDirectory: sourceDirectory,
            LabelList: cfg.LabelList,
            RecCharDictPath: cfg.RecCharDictPath,
            ClsImageShape: cfg.ClsImageShape,
            DetInputSize: cfg.DetInputSize,
            Compatibility: new ExportCompatibility("1.x", "native", true),
            OnnxInputs: onnxInputs ?? [],
            OnnxOutputs: onnxOutputs ?? []);
    }

    private static void WriteManifest(string dir, object manifest)
    {
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(dir, "manifest.json"), json);
    }

    private void ValidateManifestOrThrow(string dir)
    {
        var manifestPath = Path.Combine(dir, "manifest.json");
        if (!File.Exists(manifestPath))
        {
            throw new InvalidOperationException("manifest.json not found");
        }

        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize<ExportManifest>(json);
            if (manifest is null)
            {
                throw new InvalidOperationException("manifest parse failed");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"manifest validation failed: {ex.Message}");
        }
    }
}
