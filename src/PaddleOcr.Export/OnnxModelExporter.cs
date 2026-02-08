using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Training.Rec;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Export;

/// <summary>
/// OnnxModelExporter：从 TorchSharp 训练模型导出推理模型（纯 C# 实现，不依赖 Python）。
/// 导出 TorchScript 格式 (.pt)，同时生成包含输入输出元数据的 manifest。
/// </summary>
public sealed class OnnxModelExporter
{
    private readonly ILogger _logger;

    public OnnxModelExporter(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// 从 TorchSharp 模型导出推理模型。
    /// 使用 TorchScript 格式保存，通过前向传播记录输入输出张量元数据。
    /// </summary>
    /// <returns>导出结果，包含模型路径和输入输出元数据。</returns>
    public ExportResult Export(
        RecModel model,
        string outputDir,
        int batchSize = 1,
        int channels = 3,
        int height = 48,
        int width = 320,
        bool dynamicBatch = false)
    {
        Directory.CreateDirectory(outputDir);

        model.eval();
        using var noGrad = torch.no_grad();

        // 使用实际 batchSize 创建示例输入，前向传播获取输出形状
        using var dummyInput = torch.randn(batchSize, channels, height, width, dtype: ScalarType.Float32);
        using var output = model.forward(dummyInput);
        var outputShape = output.shape.Select(s => (int)s).ToArray();

        // 保存 TorchScript 模型
        var modelFileName = "inference.pt";
        var modelPath = Path.Combine(outputDir, modelFileName);
        model.save(modelPath);
        _logger.LogInformation("Exported TorchScript model: {Path}", modelPath);

        // 构建输入输出元数据
        var inputDims = dynamicBatch
            ? new[] { -1, channels, height, width }
            : new[] { batchSize, channels, height, width };
        var outputDims = dynamicBatch
            ? new[] { -1 }.Concat(outputShape.Skip(1)).ToArray()
            : outputShape;

        var inputs = new List<ExportTensorInfo> { new("input", inputDims) };
        var outputs = new List<ExportTensorInfo> { new("output", outputDims) };

        _logger.LogInformation(
            "Model input: [{InputShape}], output: [{OutputShape}]",
            string.Join(", ", inputDims),
            string.Join(", ", outputDims));

        return new ExportResult(modelPath, modelFileName, inputs, outputs);
    }

    /// <summary>
    /// 从配置导出推理模型。
    /// </summary>
    public string ExportFromConfig(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);

        // 加载模型
        var ckpt = ResolveCheckpoint(cfg);
        if (!File.Exists(ckpt))
        {
            throw new FileNotFoundException($"checkpoint not found: {ckpt}");
        }

        // 构建模型（从配置读取架构信息）
        var numClasses = EstimateNumClasses(cfg);
        var model = BuildModelFromConfig(cfg, numClasses);
        model.load(ckpt);

        var dev = cuda.is_available() ? CUDA : CPU;
        model.to(dev);

        // 解析图像尺寸
        var (c, h, w) = ParseImageShape(cfg);

        var result = Export(model, cfg.SaveInferenceDir, batchSize: 1, channels: c, height: h, width: w, dynamicBatch: true);

        // 如果已有 ONNX 文件，尝试读取其元数据；否则使用前向传播得到的元数据
        var onnxPath = Path.Combine(cfg.SaveInferenceDir, "inference.onnx");
        var (manifestInputs, manifestOutputs) = File.Exists(onnxPath)
            ? ReadOnnxIoMetadata(onnxPath)
            : (result.Inputs, result.Outputs);

        // 生成 manifest
        var manifest = BuildManifest(cfg, "torchscript", DateTime.UtcNow, result.ModelFileName, ckpt, null, null, manifestInputs, manifestOutputs);
        WriteManifest(cfg.SaveInferenceDir, manifest);
        ValidateManifestOrThrow(cfg.SaveInferenceDir);

        // 复制字典文件到推理目录
        CopyDictIfNeeded(cfg);

        model.Dispose();
        return result.ModelPath;
    }

    /// <summary>
    /// 向后兼容的旧入口（内部重定向到 ExportFromConfig）。
    /// </summary>
    public string ExportOnnxFromConfig(ExportConfigView cfg) => ExportFromConfig(cfg);

    private RecModel BuildModelFromConfig(ExportConfigView cfg, int numClasses)
    {
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
        if (!string.IsNullOrWhiteSpace(cfg.RecCharDictPath) && File.Exists(cfg.RecCharDictPath))
        {
            var lines = File.ReadAllLines(cfg.RecCharDictPath);
            return lines.Length + 1; // +1 for blank/PAD
        }

        return 6625;
    }

    private (int C, int H, int W) ParseImageShape(ExportConfigView cfg)
    {
        var shape = GetConfigIntList(cfg, "Train.dataset.transforms", "RecResizeImg", "image_shape");
        if (shape.Count >= 3)
        {
            return (shape[0], shape[1], shape[2]);
        }

        return (3, 48, 320);
    }

    private static string GetConfigString(ExportConfigView cfg, string path, string fallback)
    {
        return cfg.GetByPathPublic(path)?.ToString() ?? fallback;
    }

    private static int GetConfigInt(ExportConfigView cfg, string path, int fallback)
    {
        var raw = cfg.GetByPathPublic(path);
        if (raw is null)
        {
            return fallback;
        }

        return int.TryParse(raw.ToString(), out var v) ? v : fallback;
    }

    private static List<int> GetConfigIntList(ExportConfigView cfg, string transformsPath, string opName, string field)
    {
        var transforms = cfg.GetByPathPublic(transformsPath);
        if (transforms is List<object?> list)
        {
            foreach (var item in list)
            {
                if (item is Dictionary<string, object?> op && op.TryGetValue(opName, out var cfgObj) &&
                    cfgObj is Dictionary<string, object?> opCfg && opCfg.TryGetValue(field, out var fieldObj) &&
                    fieldObj is List<object?> fieldList)
                {
                    return fieldList
                        .Where(x => x is not null)
                        .Select(x => int.TryParse(x!.ToString(), out var v) ? v : 0)
                        .ToList();
                }
            }
        }

        return [3, 48, 320];
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

    private static (IReadOnlyList<ExportTensorInfo> Inputs, IReadOnlyList<ExportTensorInfo> Outputs) ReadOnnxIoMetadata(string onnxPath)
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

    private void CopyDictIfNeeded(ExportConfigView cfg)
    {
        if (string.IsNullOrWhiteSpace(cfg.RecCharDictPath) || !File.Exists(cfg.RecCharDictPath))
        {
            return;
        }

        var dest = Path.Combine(cfg.SaveInferenceDir, Path.GetFileName(cfg.RecCharDictPath));
        if (!File.Exists(dest))
        {
            File.Copy(cfg.RecCharDictPath, dest);
            _logger.LogInformation("Copied dict to {Path}", dest);
        }
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
            OnnxOutputs: onnxOutputs ?? [],
            StaticEquivalence: "compatible",
            ConversionChain: "torch_onnx");
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
        catch (Exception ex) when (ex is not InvalidOperationException)
        {
            throw new InvalidOperationException($"manifest validation failed: {ex.Message}");
        }
    }
}

/// <summary>
/// 模型导出结果。
/// </summary>
public sealed record ExportResult(
    string ModelPath,
    string ModelFileName,
    IReadOnlyList<ExportTensorInfo> Inputs,
    IReadOnlyList<ExportTensorInfo> Outputs);
