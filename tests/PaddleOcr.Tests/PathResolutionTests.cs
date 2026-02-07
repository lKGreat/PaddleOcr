using FluentAssertions;
using PaddleOcr.Export;

namespace PaddleOcr.Tests;

public sealed class PathResolutionTests
{
    [Fact]
    public void ExportConfigView_Should_Keep_Rooted_Path_Intact()
    {
        var cfgPath = Path.Combine(Path.GetTempPath(), "pocr_cfg_" + Guid.NewGuid().ToString("N") + ".yml");
        File.WriteAllText(cfgPath, "dummy: 1");
        var rooted = Path.Combine(Path.GetTempPath(), "model_dir");

        var cfg = new ExportConfigView(new Dictionary<string, object?>
        {
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "cls" },
            ["Global"] = new Dictionary<string, object?>
            {
                ["save_model_dir"] = rooted,
                ["save_inference_dir"] = rooted
            }
        }, cfgPath);

        cfg.SaveModelDir.Should().Be(rooted);
        cfg.SaveInferenceDir.Should().Be(rooted);
    }
}

