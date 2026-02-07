using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Export;
using System.Text.Json;

namespace PaddleOcr.Tests;

public sealed class ExportTests
{
    [Fact]
    public void ExportNative_Should_Copy_Checkpoint_And_Write_Manifest()
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
                ["save_model_dir"] = "./out",
                ["save_inference_dir"] = "./infer"
            }
        }, cfgPath);

        var exporter = new NativeExporter(NullLogger.Instance);
        var target = exporter.ExportNative(cfg);

        File.Exists(target).Should().BeTrue();
        File.Exists(Path.Combine(root, "infer", "manifest.json")).Should().BeTrue();

        var manifestJson = File.ReadAllText(Path.Combine(root, "infer", "manifest.json"));
        using var doc = JsonDocument.Parse(manifestJson);
        doc.RootElement.GetProperty("SchemaVersion").GetString().Should().Be("1.0");
        doc.RootElement.GetProperty("Format").GetString().Should().Be("torchsharp-native");
        doc.RootElement.GetProperty("ArtifactFile").GetString().Should().Be("model.pt");
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
}
