using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Core.Cli;
using PaddleOcr.Export;
using System.Text.Json;

namespace PaddleOcr.Tests;

public sealed class ExportTests
{
    [Fact]
    public void ExportNative_Should_Create_Paddle_Infer_Files_And_Write_Manifest()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        var cfgPath = Path.Combine(root, "cfg.yml");
        File.WriteAllText(cfgPath, "dummy: 1");

        var modelDir = Path.Combine(root, "out");
        Directory.CreateDirectory(modelDir);
        File.WriteAllText(Path.Combine(modelDir, "best.pt"), "abc");

        var cfg = new ExportConfigView(new Dictionary<string, object?>
        {
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "cls" },
            ["Global"] = new Dictionary<string, object?>
            {
                ["save_model_dir"] = modelDir,
                ["save_inference_dir"] = Path.Combine(root, "infer")
            }
        }, cfgPath);

        var exporter = new NativeExporter(NullLogger.Instance);
        var target = exporter.ExportNative(cfg);
        var inferDir = Path.GetDirectoryName(target)!;

        File.Exists(target).Should().BeTrue();
        File.Exists(Path.Combine(inferDir, "inference.json")).Should().BeTrue();
        File.Exists(Path.Combine(inferDir, "inference.pdiparams")).Should().BeTrue();
        File.Exists(Path.Combine(inferDir, "inference.yml")).Should().BeTrue();
        File.Exists(Path.Combine(inferDir, "manifest.json")).Should().BeTrue();

        var manifestJson = File.ReadAllText(Path.Combine(inferDir, "manifest.json"));
        using var doc = JsonDocument.Parse(manifestJson);
        doc.RootElement.GetProperty("SchemaVersion").GetString().Should().Be("1.0");
        doc.RootElement.GetProperty("Format").GetString().Should().Be("paddle-infer-shim");
        doc.RootElement.GetProperty("ArtifactFile").GetString().Should().Be("inference.pdiparams");
        doc.RootElement.GetProperty("Compatibility").GetProperty("ManifestSemVer").GetString().Should().Be("1.x");
        doc.RootElement.GetProperty("OnnxInputs").ValueKind.Should().Be(JsonValueKind.Array);
        doc.RootElement.GetProperty("OnnxOutputs").ValueKind.Should().Be(JsonValueKind.Array);
    }

    [Fact]
    public void ValidateJsonModelDir_Should_Report_Missing_Files()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        var exporter = new NativeExporter(NullLogger.Instance);

        var ok = exporter.ValidateJsonModelDir(root, out var message);

        ok.Should().BeFalse();
        message.Should().Contain("missing inference.json");
    }

    [Fact]
    public void ExportPaddleStatic_Should_Fail_Without_Paddle_Source_In_Strict_Mode()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        var cfgPath = Path.Combine(root, "cfg.yml");
        File.WriteAllText(cfgPath, "dummy: 1");

        var cfg = new ExportConfigView(new Dictionary<string, object?>
        {
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "rec" },
            ["Global"] = new Dictionary<string, object?>
            {
                ["save_model_dir"] = Path.Combine(root, "out"),
                ["save_inference_dir"] = Path.Combine(root, "infer"),
                ["checkpoints"] = Path.Combine(root, "best.pt")
            }
        }, cfgPath);

        var exporter = new NativeExporter(NullLogger.Instance);
        var act = () => exporter.ExportPaddleStatic(cfg);

        act.Should().Throw<FileNotFoundException>()
            .WithMessage("*No Paddle static source found*");
    }

    [Fact]
    public void ExportPaddleStatic_Should_Reject_Onnx_Source_In_Strict_Mode()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        var cfgPath = Path.Combine(root, "cfg.yml");
        File.WriteAllText(cfgPath, "dummy: 1");
        var onnxPath = Path.Combine(root, "m.onnx");
        File.WriteAllText(onnxPath, "onnx");

        var cfg = new ExportConfigView(new Dictionary<string, object?>
        {
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "rec" },
            ["Global"] = new Dictionary<string, object?>
            {
                ["save_model_dir"] = Path.Combine(root, "out"),
                ["save_inference_dir"] = Path.Combine(root, "infer"),
                ["checkpoints"] = onnxPath
            }
        }, cfgPath);

        var exporter = new NativeExporter(NullLogger.Instance);
        var act = () => exporter.ExportPaddleStatic(cfg, "strict", "onnx");

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*does not support onnx->paddle static conversion yet*");
    }

    [Fact]
    public void ValidateManifestFile_Should_Fail_When_Compatibility_Missing()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(
            Path.Combine(root, "manifest.json"),
            """
            {
              "SchemaVersion": "1.0",
              "Format": "onnx",
              "ModelType": "det",
              "CreatedAtUtc": "2026-02-07T00:00:00Z",
              "ArtifactFile": "inference.onnx",
              "OnnxInputs": [],
              "OnnxOutputs": []
            }
            """);

        var exporter = new NativeExporter(NullLogger.Instance);
        var ok = exporter.ValidateManifestFile(root, out var message);

        ok.Should().BeFalse();
        message.ToLowerInvariant().Should().Contain("compatibility");
    }

    [Fact]
    public void ValidateManifestFile_Should_Fail_When_ManifestSemVer_Not_1x()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(
            Path.Combine(root, "manifest.json"),
            """
            {
              "SchemaVersion": "1.0",
              "Format": "onnx",
              "ModelType": "det",
              "CreatedAtUtc": "2026-02-07T00:00:00Z",
              "ArtifactFile": "inference.onnx",
              "Compatibility": { "ManifestSemVer": "2.0", "Runtime": "native", "BackwardCompatible": false },
              "OnnxInputs": [],
              "OnnxOutputs": []
            }
            """);

        var exporter = new NativeExporter(NullLogger.Instance);
        var ok = exporter.ValidateManifestFile(root, out var message);

        ok.Should().BeFalse();
        message.Should().Contain("unsupported manifest compatibility");
    }

    [Fact]
    public void ExportPaddleStatic_Should_Copy_From_Paddle_Source_Dir_In_Strict_Mode()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        var cfgPath = Path.Combine(root, "cfg.yml");
        File.WriteAllText(cfgPath, "dummy: 1");

        var sourceDir = Path.Combine(root, "paddle_src");
        Directory.CreateDirectory(sourceDir);
        File.WriteAllText(Path.Combine(sourceDir, "inference.json"), "{\"v\":1}");
        File.WriteAllText(Path.Combine(sourceDir, "inference.pdiparams"), "abc");
        File.WriteAllText(Path.Combine(sourceDir, "inference.yml"), "Global:\n  model_name: test\n");

        var cfg = new ExportConfigView(new Dictionary<string, object?>
        {
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "rec" },
            ["Global"] = new Dictionary<string, object?>
            {
                ["save_model_dir"] = Path.Combine(root, "out"),
                ["save_inference_dir"] = Path.Combine(root, "infer"),
                ["pretrained_model"] = sourceDir,
                ["character_dict_path"] = Path.Combine(root, "dict.txt")
            }
        }, cfgPath);

        File.WriteAllText(Path.Combine(root, "dict.txt"), "a\nb\n");
        var exporter = new NativeExporter(NullLogger.Instance);
        var target = exporter.ExportPaddleStatic(cfg, "strict", "paddle");

        target.Should().EndWith("inference.json");
        var inferDir = Path.GetDirectoryName(target)!;
        File.ReadAllText(Path.Combine(inferDir, "inference.json")).Should().Be("{\"v\":1}");
        File.ReadAllText(Path.Combine(inferDir, "inference.pdiparams")).Should().Be("abc");
        File.ReadAllText(Path.Combine(inferDir, "inference.yml")).Should().Contain("model_name: test");
    }

    [Fact]
    public async Task ConvertCheckJsonModel_Should_Not_Require_ConfigPath()
    {
        var root = Path.Combine(Path.GetTempPath(), "pocr_export_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "inference.json"), "{}");
        File.WriteAllBytes(Path.Combine(root, "inference.pdiparams"), [1, 2, 3]);

        var executor = new ExportExecutor();
        var context = new PaddleOcr.Core.Cli.ExecutionContext(
            NullLogger.Instance,
            ["convert", "check-json-model"],
            null,
            new Dictionary<string, object?>(),
            new Dictionary<string, string> { ["--json_model_dir"] = root },
            []);

        var result = await executor.ExecuteAsync("convert:check-json-model", context);
        result.Success.Should().BeTrue();
    }
}
