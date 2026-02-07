using FluentAssertions;
using PaddleOcr.Config;

namespace PaddleOcr.Tests;

public sealed class ConfigTests
{
    [Fact]
    public void OverrideParser_Should_Parse_Scalar_Values()
    {
        var parsed = OverrideParser.Parse([
            "Global.use_gpu=false",
            "Global.epoch_num=200",
            "Metric.name='acc'"
        ]);

        parsed["Global.use_gpu"].Should().Be(false);
        parsed["Global.epoch_num"].Should().Be(200);
        parsed["Metric.name"].Should().Be("acc");
    }

    [Fact]
    public void ConfigMerger_Should_Merge_Dotted_Path()
    {
        var cfg = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["Global"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["epoch_num"] = 10
            }
        };

        ConfigMerger.MergeInPlace(cfg, new Dictionary<string, object?>
        {
            ["Global.epoch_num"] = 20,
            ["Global.save_model_dir"] = "./output"
        });

        var global = (IDictionary<string, object?>)cfg["Global"]!;
        global["epoch_num"].Should().Be(20);
        global["save_model_dir"].Should().Be("./output");
    }

    [Fact]
    public void ConfigValidator_Diff_Should_Report_Changed_Field()
    {
        var a = new Dictionary<string, object?>
        {
            ["Global"] = new Dictionary<string, object?> { ["epoch_num"] = 1 },
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "cls" }
        };
        var b = new Dictionary<string, object?>
        {
            ["Global"] = new Dictionary<string, object?> { ["epoch_num"] = 2 },
            ["Architecture"] = new Dictionary<string, object?> { ["model_type"] = "cls" }
        };

        var diffs = ConfigValidator.Diff(a, b);
        diffs.Should().ContainSingle(x => x.Contains("Global.epoch_num"));
    }
}
