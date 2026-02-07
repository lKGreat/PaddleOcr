using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace PaddleOcr.Export;

public sealed class NativeExporter
{
    private readonly ILogger _logger;

    public NativeExporter(ILogger logger)
    {
        _logger = logger;
    }

    public string ExportNative(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);
        var ckpt = ResolveCheckpoint(cfg);
        if (!File.Exists(ckpt))
        {
            throw new FileNotFoundException($"checkpoint not found: {ckpt}");
        }

        var target = Path.Combine(cfg.SaveInferenceDir, "model.pt");
        File.Copy(ckpt, target, overwrite: true);
        WriteManifest(cfg.SaveInferenceDir, new
        {
            format = "torchsharp-native",
            model_type = cfg.ModelType,
            checkpoint = ckpt,
            exported_at = DateTime.UtcNow
        });
        _logger.LogInformation("Exported native model: {Path}", target);
        return target;
    }

    public string ExportOnnx(ExportConfigView cfg)
    {
        Directory.CreateDirectory(cfg.SaveInferenceDir);
        var onnx = ResolveOnnxSource(cfg);
        if (onnx is null || !File.Exists(onnx))
        {
            throw new FileNotFoundException("No ONNX source found. Set Global.checkpoints or Global.pretrained_model to an .onnx file.");
        }

        var target = Path.Combine(cfg.SaveInferenceDir, "inference.onnx");
        File.Copy(onnx, target, overwrite: true);
        WriteManifest(cfg.SaveInferenceDir, new
        {
            format = "onnx",
            model_type = cfg.ModelType,
            source = onnx,
            exported_at = DateTime.UtcNow
        });
        _logger.LogInformation("Exported ONNX model: {Path}", target);
        return target;
    }

    public string ConvertJsonToPdmodel(string jsonModelDir, string outputDir)
    {
        if (!Directory.Exists(jsonModelDir))
        {
            throw new DirectoryNotFoundException($"json model dir not found: {jsonModelDir}");
        }

        Directory.CreateDirectory(outputDir);
        var srcJson = Path.Combine(jsonModelDir, "inference.json");
        var srcParams = Path.Combine(jsonModelDir, "inference.pdiparams");
        if (!File.Exists(srcJson) || !File.Exists(srcParams))
        {
            throw new FileNotFoundException("Expect inference.json and inference.pdiparams in json model dir.");
        }

        var dstModel = Path.Combine(outputDir, "inference.pdmodel");
        var dstParams = Path.Combine(outputDir, "inference.pdiparams");
        File.Copy(srcJson, dstModel, overwrite: true);
        File.Copy(srcParams, dstParams, overwrite: true);

        WriteManifest(outputDir, new
        {
            format = "pdmodel-shim",
            source = jsonModelDir,
            converted_at = DateTime.UtcNow
        });
        _logger.LogInformation("Converted json model dir to pdmodel shim: {Dir}", outputDir);
        return dstModel;
    }

    private static void WriteManifest(string dir, object manifest)
    {
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(dir, "manifest.json"), json);
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

    private static string? ResolveOnnxSource(ExportConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints) &&
            cfg.Checkpoints.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            return cfg.Checkpoints;
        }

        if (!string.IsNullOrWhiteSpace(cfg.PretrainedModel))
        {
            if (cfg.PretrainedModel.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                return cfg.PretrainedModel;
            }

            var candidate = Path.Combine(cfg.PretrainedModel, "inference.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }
}
