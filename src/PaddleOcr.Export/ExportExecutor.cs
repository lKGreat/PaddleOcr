using Microsoft.Extensions.Logging;
using PaddleOcr.Core.Cli;
using PaddleOcr.Training;
using PaddleOcr.Training.Rec;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Export;

public sealed class ExportExecutor : ICommandExecutor
{
    private static readonly HashSet<string> Supported = new(StringComparer.OrdinalIgnoreCase)
    {
        "export",
        "export-onnx",
        "export-center",
        "convert:json2pdmodel",
        "convert:check-json-model"
    };

    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!Supported.Contains(subCommand))
        {
            return Task.FromResult(CommandResult.Fail($"Unsupported export command: {subCommand}"));
        }

        if (!subCommand.Equals("convert:json2pdmodel", StringComparison.OrdinalIgnoreCase) &&
            string.IsNullOrWhiteSpace(context.ConfigPath))
        {
            return Task.FromResult(CommandResult.Fail($"{subCommand} requires -c/--config"));
        }

        try
        {
            var exporter = new NativeExporter(context.Logger);
            if (subCommand.Equals("convert:json2pdmodel", StringComparison.OrdinalIgnoreCase))
            {
                if (!context.Options.TryGetValue("--json_model_dir", out var jsonDir) ||
                    !context.Options.TryGetValue("--output_dir", out var outputDir))
                {
                    return Task.FromResult(CommandResult.Fail("convert json2pdmodel requires --json_model_dir and --output_dir"));
                }

                if (!exporter.ValidateJsonModelDir(jsonDir, out var precheckMessage))
                {
                    return Task.FromResult(CommandResult.Fail($"convert json2pdmodel precheck failed: {precheckMessage}"));
                }

                exporter.ConvertJsonToPdmodel(jsonDir, outputDir);
                return Task.FromResult(CommandResult.Ok($"convert:json2pdmodel completed. output={outputDir}"));
            }

            if (subCommand.Equals("convert:check-json-model", StringComparison.OrdinalIgnoreCase))
            {
                if (!context.Options.TryGetValue("--json_model_dir", out var jsonDir))
                {
                    return Task.FromResult(CommandResult.Fail("convert check-json-model requires --json_model_dir"));
                }

                var ok = exporter.ValidateJsonModelDir(jsonDir, out var message);
                return Task.FromResult(ok
                    ? CommandResult.Ok($"convert:check-json-model passed: {message}")
                    : CommandResult.Fail($"convert:check-json-model failed: {message}"));
            }

            var cfg = new ExportConfigView(context.Config, context.ConfigPath!);
            if (subCommand.Equals("export-onnx", StringComparison.OrdinalIgnoreCase))
            {
                try
                {
                    var onnxExporter = new OnnxModelExporter(context.Logger);
                    var outFile = onnxExporter.ExportFromConfig(cfg);
                    return Task.FromResult(CommandResult.Ok($"export-onnx completed. output={outFile}"));
                }
                catch
                {
                    var outFile = exporter.ExportOnnx(cfg);
                    return Task.FromResult(CommandResult.Ok($"export-onnx completed. output={outFile}"));
                }
            }

            if (subCommand.Equals("export-center", StringComparison.OrdinalIgnoreCase))
            {
                var result = RunExportCenter(cfg, context.Logger);
                return Task.FromResult(result);
            }

            var native = exporter.ExportNative(cfg);
            return Task.FromResult(CommandResult.Ok($"{subCommand} completed. output={native}"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(CommandResult.Fail($"{subCommand} failed: {ex.Message}"));
        }
    }

    /// <summary>
    /// 执行 export-center 命令：加载模型 -> 遍历训练数据 -> 提取特征 -> 聚合中心 -> 保存。
    /// </summary>
    private static CommandResult RunExportCenter(ExportConfigView cfg, ILogger logger)
    {
        // 读取字典
        var dictPath = cfg.RecCharDictPath;
        if (string.IsNullOrWhiteSpace(dictPath) || !File.Exists(dictPath))
        {
            return CommandResult.Fail("export-center requires a valid rec_char_dict_path in config");
        }

        var (charToId, vocab) = SimpleRecDataset.LoadDictionary(dictPath, useSpaceChar: true);

        // 读取训练数据配置
        var trainLabelFile = cfg.GetByPathPublic("Train.dataset.label_file_list");
        string labelFilePath;
        if (trainLabelFile is List<object?> labelList && labelList.Count > 0 && labelList[0] is not null)
        {
            labelFilePath = labelList[0]!.ToString() ?? string.Empty;
        }
        else
        {
            return CommandResult.Fail("export-center requires Train.dataset.label_file_list in config");
        }

        var dataDir = cfg.GetByPathPublic("Train.dataset.data_dir")?.ToString() ?? ".";
        var maxTextLength = int.TryParse(cfg.GetByPathPublic("Global.max_text_length")?.ToString(), out var mtl) ? mtl : 25;

        // 解析图像尺寸
        var recShapeObj = cfg.GetByPathPublic("Global.rec_image_shape");
        int h = 48, w = 320;
        if (recShapeObj is string shapeStr)
        {
            var parts = shapeStr.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length >= 3 && int.TryParse(parts[1], out var ph) && int.TryParse(parts[2], out var pw))
            {
                h = ph;
                w = pw;
            }
        }

        // 构建模型
        var backboneName = cfg.GetByPathPublic("Architecture.Backbone.name")?.ToString() ?? "MobileNetV1Enhance";
        var neckName = cfg.GetByPathPublic("Architecture.Neck.name")?.ToString() ?? "SequenceEncoder";
        var headName = cfg.GetByPathPublic("Architecture.Head.name")?.ToString() ?? "CTCHead";
        var hiddenSize = int.TryParse(cfg.GetByPathPublic("Architecture.Head.hidden_size")?.ToString(), out var hs) ? hs : 48;
        var inChannels = int.TryParse(cfg.GetByPathPublic("Architecture.in_channels")?.ToString(), out var ic) ? ic : 3;
        var numClasses = vocab.Count + 1;

        var model = RecModelBuilder.Build(backboneName, neckName, headName, numClasses, inChannels, hiddenSize, maxTextLength);

        // 加载 checkpoint
        var ckpt = cfg.Checkpoints;
        if (string.IsNullOrWhiteSpace(ckpt))
        {
            var best = Path.Combine(cfg.SaveModelDir, "best.pt");
            ckpt = File.Exists(best) ? best : Path.Combine(cfg.SaveModelDir, "latest.pt");
        }

        if (!File.Exists(ckpt))
        {
            model.Dispose();
            return CommandResult.Fail($"export-center: checkpoint not found: {ckpt}");
        }

        model.load(ckpt);
        var dev = cuda.is_available() ? CUDA : CPU;
        model.to(dev);

        // 加载训练数据集
        var trainSet = new SimpleRecDataset(labelFilePath, dataDir, h, w, maxTextLength, charToId);

        // 导出中心
        var centerExporter = new CenterExporter(logger);
        var outputPath = Path.Combine(cfg.SaveInferenceDir, "train_center.json");
        var batches = trainSet.GetBatches(32, shuffle: false, new Random(7));
        centerExporter.ExportCenter(model, batches, h, w, maxTextLength, charToId, vocab, outputPath, device: dev);

        model.Dispose();
        return CommandResult.Ok($"export-center completed. output={outputPath}");
    }
}
