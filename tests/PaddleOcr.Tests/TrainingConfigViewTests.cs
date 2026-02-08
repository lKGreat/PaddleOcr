using FluentAssertions;
namespace PaddleOcr.Tests;

public sealed class TrainingConfigViewTests
{
    [Fact]
    public void CtcInputLengthMode_Should_Default_To_Full()
    {
        var cfg = CreateConfig(new Dictionary<string, object?>());
        GetString(cfg, "CtcInputLengthMode").Should().Be("full");
        GetBool(cfg, "UseValidRatioForCtcInputLength").Should().BeFalse();
    }

    [Fact]
    public void CtcInputLengthMode_Should_Parse_ValidRatio()
    {
        var cfg = CreateConfig(new Dictionary<string, object?>
        {
            ["Global"] = new Dictionary<string, object?> { ["ctc_input_length_mode"] = "valid_ratio" }
        });

        GetString(cfg, "CtcInputLengthMode").Should().Be("valid_ratio");
        GetBool(cfg, "UseValidRatioForCtcInputLength").Should().BeTrue();
    }

    [Fact]
    public void MultiScaleSampler_Should_Provide_Widths_And_Heights()
    {
        var cfg = CreateConfig(new Dictionary<string, object?>
        {
            ["Train"] = new Dictionary<string, object?>
            {
                ["dataset"] = new Dictionary<string, object?>
                {
                    ["name"] = "MultiScaleDataSet",
                    ["ds_width"] = false
                },
                ["sampler"] = new Dictionary<string, object?>
                {
                    ["scales"] = new List<object?>
                    {
                        new List<object?> { 320, 32 },
                        new List<object?> { 320, 48 },
                        new List<object?> { 640, 48 }
                    }
                }
            }
        });

        GetIntArray(cfg, "MultiScaleWidths").Should().BeEquivalentTo([320, 640]);
        GetIntArray(cfg, "MultiScaleHeights").Should().BeEquivalentTo([32, 48]);
    }

    [Fact]
    public void TrainRatioList_Should_Parse_Scalar_And_Delimiter()
    {
        var cfg = CreateConfig(new Dictionary<string, object?>
        {
            ["Train"] = new Dictionary<string, object?>
            {
                ["dataset"] = new Dictionary<string, object?>
                {
                    ["ratio_list"] = 0.25,
                    ["delimiter"] = ","
                }
            }
        });

        GetFloatArray(cfg, "TrainRatioList").Should().Equal(0.25f);
        GetString(cfg, "TrainDelimiter").Should().Be(",");
    }

    [Fact]
    public void TeacherBackend_Should_Parse_And_Default()
    {
        var def = CreateConfig(new Dictionary<string, object?>());
        GetString(def, "TeacherBackend").Should().Be("paddle");
        GetBool(def, "TeacherUseGpu").Should().BeTrue();
        GetInt(def, "TeacherGpuDeviceId").Should().Be(0);
        GetInt(def, "TeacherGpuMemMb").Should().Be(1024);
    }

    [Fact]
    public void CharsetCoverageThreshold_Should_Parse()
    {
        var cfg = CreateConfig(new Dictionary<string, object?>
        {
            ["Global"] = new Dictionary<string, object?>
            {
                ["charset_coverage_fail_fast"] = false,
                ["charset_max_unknown_ratio"] = 0.2
            }
        });

        GetBool(cfg, "CharsetCoverageFailFast").Should().BeFalse();
        GetFloat(cfg, "CharsetMaxUnknownRatio").Should().BeApproximately(0.2f, 1e-6f);
    }

    private static object CreateConfig(Dictionary<string, object?> overrides)
    {
        var root = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["Global"] = new Dictionary<string, object?>(StringComparer.Ordinal),
            ["Train"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["dataset"] = new Dictionary<string, object?>(StringComparer.Ordinal),
                ["loader"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            },
            ["Eval"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["dataset"] = new Dictionary<string, object?>(StringComparer.Ordinal),
                ["loader"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            },
            ["Architecture"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["model_type"] = "rec"
            },
            ["Optimizer"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["lr"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            }
        };

        Merge(root, overrides);
        var asm = typeof(PaddleOcr.Training.TrainingExecutor).Assembly;
        var type = asm.GetType("PaddleOcr.Training.TrainingConfigView", throwOnError: true)!;
        return Activator.CreateInstance(type, root, Path.Combine(Path.GetTempPath(), "cfg.yml"))!;
    }

    private static void Merge(IDictionary<string, object?> target, IDictionary<string, object?> source)
    {
        foreach (var (key, value) in source)
        {
            if (value is IDictionary<string, object?> sourceMap &&
                target.TryGetValue(key, out var existing) &&
                existing is IDictionary<string, object?> targetMap)
            {
                Merge(targetMap, sourceMap);
            }
            else
            {
                target[key] = value;
            }
        }
    }

    private static string GetString(object cfg, string propertyName)
    {
        return (string)(cfg.GetType().GetProperty(propertyName)!.GetValue(cfg) ?? string.Empty);
    }

    private static bool GetBool(object cfg, string propertyName)
    {
        return (bool)(cfg.GetType().GetProperty(propertyName)!.GetValue(cfg) ?? false);
    }

    private static int[] GetIntArray(object cfg, string propertyName)
    {
        var value = cfg.GetType().GetProperty(propertyName)!.GetValue(cfg);
        return value switch
        {
            int[] arr => arr,
            IEnumerable<int> seq => seq.ToArray(),
            _ => []
        };
    }

    private static float[] GetFloatArray(object cfg, string propertyName)
    {
        var value = cfg.GetType().GetProperty(propertyName)!.GetValue(cfg);
        return value switch
        {
            float[] arr => arr,
            IEnumerable<float> seq => seq.ToArray(),
            _ => []
        };
    }

    private static int GetInt(object cfg, string propertyName)
    {
        return (int)(cfg.GetType().GetProperty(propertyName)!.GetValue(cfg) ?? 0);
    }

    private static float GetFloat(object cfg, string propertyName)
    {
        return (float)(cfg.GetType().GetProperty(propertyName)!.GetValue(cfg) ?? 0f);
    }
}
