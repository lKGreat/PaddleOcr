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
    public void ConfigMerger_Should_Merge_List_Index_Path()
    {
        var cfg = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["Train"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["dataset"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["label_file_list"] = new List<object?> { "a.txt" }
                }
            }
        };

        ConfigMerger.MergeInPlace(cfg, new Dictionary<string, object?>
        {
            ["Train.dataset.label_file_list[0]"] = "train.txt",
            ["Eval.dataset.label_file_list[0]"] = "eval.txt"
        });

        var train = (IDictionary<string, object?>)cfg["Train"]!;
        var trainDataset = (IDictionary<string, object?>)train["dataset"]!;
        var trainList = (System.Collections.IList)trainDataset["label_file_list"]!;
        trainList[0].Should().Be("train.txt");

        var eval = (IDictionary<string, object?>)cfg["Eval"]!;
        var evalDataset = (IDictionary<string, object?>)eval["dataset"]!;
        var evalList = (System.Collections.IList)evalDataset["label_file_list"]!;
        evalList[0].Should().Be("eval.txt");
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
