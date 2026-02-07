using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Export;

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
    }
}

