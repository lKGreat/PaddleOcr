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

    [Fact]
    public void TrainLoader_And_Sampler_Should_Parse_Official_Fields()
    {
        var cfg = CreateConfig(new Dictionary<string, object?>
        {
            ["Train"] = new Dictionary<string, object?>
            {
                ["dataset"] = new Dictionary<string, object?>
                {
                    ["name"] = "MultiScaleDataSet",
                    ["ds_width"] = true,
                    ["ext_op_transform_idx"] = 1,
                    ["transforms"] = new List<object?>
                    {
                        new Dictionary<string, object?>
                        {
                            ["RecConAug"] = new Dictionary<string, object?>
                            {
                                ["image_shape"] = new List<object?> { 32, 224, 3 }
                            }
                        }
                    }
                },
                ["loader"] = new Dictionary<string, object?>
                {
                    ["shuffle"] = false,
                    ["drop_last"] = true,
                    ["num_workers"] = 6
                },
                ["sampler"] = new Dictionary<string, object?>
                {
                    ["name"] = "MultiScaleSampler",
                    ["scales"] = new List<object?>
                    {
                        new List<object?> { 320, 32 },
                        new List<object?> { 640, 48 }
                    },
                    ["first_bs"] = 192,
                    ["fix_bs"] = false,
                    ["divided_factor"] = new List<object?> { 8, 16 }
                }
            },
            ["Eval"] = new Dictionary<string, object?>
            {
                ["loader"] = new Dictionary<string, object?>
                {
                    ["shuffle"] = true,
                    ["drop_last"] = true,
                    ["num_workers"] = 2
                }
            }
        });

        GetBool(cfg, "TrainShuffle").Should().BeFalse();
        GetBool(cfg, "TrainDropLast").Should().BeTrue();
        GetInt(cfg, "TrainNumWorkers").Should().Be(6);
        GetBool(cfg, "EvalShuffle").Should().BeTrue();
        GetBool(cfg, "EvalDropLast").Should().BeTrue();
        GetInt(cfg, "EvalNumWorkers").Should().Be(2);
        GetBool(cfg, "TrainDsWidth").Should().BeTrue();
        GetInt(cfg, "TrainExtOpTransformIdx").Should().Be(1);
        GetBool(cfg, "UseTrainMultiScaleSampler").Should().BeTrue();
        GetString(cfg, "TrainSamplerName").Should().Be("MultiScaleSampler");
        GetInt(cfg, "TrainSamplerFirstBatchSize").Should().Be(192);
        GetBool(cfg, "TrainSamplerFixBatchSize").Should().BeFalse();
        GetIntArray(cfg, "TrainSamplerDividedFactor").Should().Equal(8, 16);
        GetValueTuple3Int(cfg, "RecImageShape").Should().Be((3, 32, 224));
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

    private static (int C, int H, int W) GetValueTuple3Int(object cfg, string propertyName)
    {
        var value = cfg.GetType().GetProperty(propertyName)!.GetValue(cfg);
        return value switch
        {
            ValueTuple<int, int, int> tuple => tuple,
            _ => (0, 0, 0)
        };
    }
}
