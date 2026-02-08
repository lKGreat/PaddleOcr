namespace PaddleOcr.Training;

internal sealed class TrainingConfigView
{
    private readonly IReadOnlyDictionary<string, object?> _root;
    private readonly string _configDir;

    public TrainingConfigView(IReadOnlyDictionary<string, object?> root, string configPath)
    {
        _root = root;
        _configDir = Path.GetDirectoryName(Path.GetFullPath(configPath)) ?? Directory.GetCurrentDirectory();
    }

    public string ModelType => GetString("Architecture.model_type", "cls");
    public int EpochNum => GetInt("Global.epoch_num", 1);
    public int BatchSize => GetInt("Train.loader.batch_size_per_card", 16);
    public int EvalBatchSize => GetInt("Eval.loader.batch_size_per_card", BatchSize);
    public int PrintBatchStep => Math.Max(1, GetInt("Global.print_batch_step", 10));
    public int LogSmoothWindow => Math.Max(1, GetInt("Global.log_smooth_window", 20));
    public int SaveEpochStep => Math.Max(1, GetInt("Global.save_epoch_step", 1));
    public bool SaveBatchModel => GetBool("Global.save_batch_model", true);
    public (int Start, int Interval) EvalBatchStep => ParseEvalBatchStep();
    public bool CalMetricDuringTrain => GetBool("Global.cal_metric_during_train", true);
    public int CalcEpochInterval => Math.Max(1, GetInt("Global.calc_epoch_interval", 1));
    public float LearningRate => GetFloat("Optimizer.lr.learning_rate", 1e-3f);
    public int LrDecayStep => GetInt("Optimizer.lr.step_size", 0);
    public float LrDecayGamma => GetFloat("Optimizer.lr.gamma", 0.1f);
    public float GradClipNorm => GetFloat("Optimizer.grad_clip_norm", 0f);
    public int EarlyStopPatience => GetInt("Global.early_stop_patience", 0);
    public bool ResumeTraining => GetBool("Global.resume_training", true);
    public int Seed => GetInt("Global.seed", 1024);
    public bool Deterministic => GetBool("Global.deterministic", true);
    public string DeviceRaw => (GetStringOrNull("Global.device") ?? string.Empty).Trim().ToLowerInvariant();
    public bool UseGpu => GetBool("Global.use_gpu", false);
    public string Device => ResolveDeviceString();
    public bool UseAmp => ResolveUseAmp();
    public float MinImproveDelta => GetFloat("Global.min_improve_delta", 1e-4f);
    public bool NanGuard => GetBool("Global.nan_guard", true);
    public string SaveModelDir => ResolvePath(GetString("Global.save_model_dir", "./output/cls"));
    public string? Checkpoints => ResolvePathOrNull(GetStringOrNull("Global.checkpoints"));
    public string? PretrainedModel => ResolvePathOrNull(GetStringOrNull("Global.pretrained_model"));
    public string? TeacherModelDir => ResolvePathOrNull(GetStringOrNull("Global.teacher_model_dir"));
    public string? TeacherPaddleLibDir => ResolvePathOrNull(GetStringOrNull("Global.teacher_paddle_lib_dir"));
    public string? TeacherOnnxModelPath => ResolvePathOrNull(GetStringOrNull("Global.teacher_onnx_model_path"));
    public string TeacherBackend => ResolveTeacherBackend();
    public bool TeacherUseGpu => GetBool("Global.teacher_use_gpu", true);
    public int TeacherGpuDeviceId => Math.Max(0, GetInt("Global.teacher_gpu_device_id", 0));
    public int TeacherGpuMemMb => Math.Max(64, GetInt("Global.teacher_gpu_mem_mb", 1024));
    public float DistillWeight => Clamp(GetFloat("Global.distill_weight", 0f), 0f, 1f);
    public float DistillTemperature => Math.Max(1e-3f, GetFloat("Global.distill_temperature", 1f));
    public bool StrictTeacherStudent => GetBool("Global.strict_teacher_student", true);
    public string CtcInputLengthMode => ResolveCtcInputLengthMode();
    public bool UseValidRatioForCtcInputLength => CtcInputLengthMode == "valid_ratio";
    public bool CharsetCoverageFailFast => GetBool("Global.charset_coverage_fail_fast", true);
    public float CharsetMaxUnknownRatio => Clamp(GetFloat("Global.charset_max_unknown_ratio", 0.05f), 0f, 1f);

    public string TrainLabelFile => ResolvePath(GetFirstString("Train.dataset.label_file_list"));
    public string EvalLabelFile => ResolvePath(GetFirstString("Eval.dataset.label_file_list"));
    public IReadOnlyList<string> TrainLabelFiles => GetStringList("Train.dataset.label_file_list").Select(ResolvePath).ToList();
    public IReadOnlyList<string> EvalLabelFiles => GetStringList("Eval.dataset.label_file_list").Select(ResolvePath).ToList();
    public string TrainDelimiter => GetString("Train.dataset.delimiter", "\t");
    public string EvalDelimiter => GetString("Eval.dataset.delimiter", TrainDelimiter);
    public IReadOnlyList<float> TrainRatioList => ParseRatioList("Train.dataset.ratio_list");
    public string DataDir => ResolvePath(GetString("Train.dataset.data_dir", "."));
    public string EvalDataDir => ResolvePath(GetString("Eval.dataset.data_dir", DataDir));
    public string InvalidSamplePolicy => GetString("Train.dataset.invalid_sample_policy", "skip").Trim().ToLowerInvariant();
    public int MinValidSamples => GetInt("Train.dataset.min_valid_samples", 1);
    public float DetShrinkRatio => Clamp(GetFloatAny(["Train.dataset.det_shrink_ratio", "Loss.det_shrink_ratio"], 0.4f), 0.05f, 0.95f);
    public float DetThreshMin => Clamp(GetFloatAny(["Train.dataset.det_thresh_min", "Loss.det_thresh_min"], 0.3f), 0f, 1f);
    public float DetThreshMax => Clamp(GetFloatAny(["Train.dataset.det_thresh_max", "Loss.det_thresh_max"], 0.7f), 0f, 1f);
    public float DetShrinkLossWeight => Math.Max(0f, GetFloatAny(["Loss.det_shrink_loss_weight", "Loss.alpha"], 1f));
    public float DetThresholdLossWeight => Math.Max(0f, GetFloatAny(["Loss.det_threshold_loss_weight", "Loss.beta"], 0.5f));
    public float DetEvalIouThresh => Clamp(GetFloatAny(["Global.det_eval_iou_thresh", "Metric.det_eval_iou_thresh"], 0.5f), 0f, 1f);

    public (int C, int H, int W) ImageShape => ParseImageShape();
    public int DetInputSize => GetInt("Train.dataset.transforms.ResizeTextImg.size", 640);
    public (int C, int H, int W) RecImageShape => ParseRecImageShape();
    public string? RecCharDictPath => ResolvePathOrNull(GetStringOrNull("Global.character_dict_path"));
    public int MaxTextLength => GetInt("Global.max_text_length", 25);
    public bool UseSpaceChar => GetBool("Global.use_space_char", true);

    private (int C, int H, int W) ParseImageShape()
    {
        var shape = GetIntListFromTransforms("Train.dataset.transforms", "ClsResizeImg", "image_shape");
        if (shape.Count >= 3)
        {
            return (shape[0], shape[1], shape[2]);
        }

        return (3, 48, 192);
    }

    private (int C, int H, int W) ParseRecImageShape()
    {
        var shape = GetIntListFromTransforms("Train.dataset.transforms", "RecResizeImg", "image_shape");
        if (shape.Count >= 3)
        {
            return (shape[0], shape[1], shape[2]);
        }

        return (3, 48, 320);
    }

    private string ResolvePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        if (Path.IsPathRooted(path))
        {
            return path;
        }

        var fromCwd = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), path));
        if (File.Exists(fromCwd) || Directory.Exists(fromCwd))
        {
            return fromCwd;
        }

        var fromConfig = Path.GetFullPath(Path.Combine(_configDir, path));
        if (File.Exists(fromConfig) || Directory.Exists(fromConfig))
        {
            return fromConfig;
        }

        return fromCwd;
    }

    private string? ResolvePathOrNull(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        var resolved = ResolvePath(path);
        if (File.Exists(resolved) || Directory.Exists(resolved))
        {
            return resolved;
        }

        var fileName = Path.GetFileName(path);
        if (!string.IsNullOrWhiteSpace(fileName))
        {
            var dictCandidate = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "assets", "dicts", fileName));
            if (File.Exists(dictCandidate))
            {
                return dictCandidate;
            }
        }

        return resolved;
    }

    private string GetFirstString(string path)
    {
        var values = GetStringList(path);
        return values.Count == 0 ? string.Empty : values[0];
    }

    private List<string> GetStringList(string path)
    {
        var raw = GetByPath(path);
        if (raw is IList<object?> list)
        {
            return list
                .Where(x => x is not null)
                .Select(x => x!.ToString() ?? string.Empty)
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .ToList();
        }

        var scalar = raw?.ToString();
        if (!string.IsNullOrWhiteSpace(scalar))
        {
            return [scalar];
        }

        return [];
    }

    private List<int> GetIntListFromTransforms(string transformsPath, string opName, string field)
    {
        if (GetByPath(transformsPath) is not List<object?> transforms)
        {
            return [];
        }

        foreach (var item in transforms)
        {
            if (item is not Dictionary<string, object?> op)
            {
                continue;
            }

            if (!op.TryGetValue(opName, out var opVal))
            {
                continue;
            }

            if (opVal is Dictionary<string, object?> opCfg && opCfg.TryGetValue(field, out var listObj) && listObj is List<object?> list)
            {
                return list
                    .Where(x => x is not null)
                    .Select(x => int.TryParse(x!.ToString(), out var v) ? v : 0)
                    .ToList();
            }
        }

        return [];
    }

    public bool HasTransform(string transformsPath, string opName)
    {
        if (GetByPath(transformsPath) is not List<object?> transforms)
        {
            return false;
        }

        foreach (var item in transforms)
        {
            if (item is Dictionary<string, object?> op && op.ContainsKey(opName))
            {
                return true;
            }
        }

        return false;
    }

    private object? GetByPath(string path)
    {
        var parts = path.Split('.', StringSplitOptions.RemoveEmptyEntries);
        object? cur = _root;
        foreach (var p in parts)
        {
            if (cur is IReadOnlyDictionary<string, object?> rd && rd.TryGetValue(p, out var v))
            {
                cur = v;
                continue;
            }

            if (cur is Dictionary<string, object?> d && d.TryGetValue(p, out var dv))
            {
                cur = dv;
                continue;
            }

            return null;
        }

        return cur;
    }

    private string GetString(string path, string fallback)
    {
        return GetByPath(path)?.ToString() ?? fallback;
    }

    private string? GetStringOrNull(string path)
    {
        return GetByPath(path)?.ToString();
    }

    private int GetInt(string path, int fallback)
    {
        return int.TryParse(GetByPath(path)?.ToString(), out var v) ? v : fallback;
    }

    private float GetFloat(string path, float fallback)
    {
        return float.TryParse(GetByPath(path)?.ToString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : fallback;
    }

    private float GetFloatAny(IReadOnlyList<string> paths, float fallback)
    {
        foreach (var path in paths)
        {
            if (float.TryParse(GetByPath(path)?.ToString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var value))
            {
                return value;
            }
        }

        return fallback;
    }

    private bool GetBool(string path, bool fallback)
    {
        return bool.TryParse(GetByPath(path)?.ToString(), out var v) ? v : fallback;
    }

    private string ResolveDeviceString()
    {
        if (!string.IsNullOrWhiteSpace(DeviceRaw))
        {
            return DeviceRaw;
        }

        return UseGpu ? "cuda" : "cpu";
    }

    private bool ResolveUseAmp()
    {
        var raw = GetStringOrNull("Global.use_amp");
        if (!string.IsNullOrWhiteSpace(raw) && bool.TryParse(raw, out var explicitFlag))
        {
            return explicitFlag;
        }

        return Device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase);
    }

    private string ResolveCtcInputLengthMode()
    {
        var raw = (GetStringOrNull("Global.ctc_input_length_mode") ?? string.Empty).Trim().ToLowerInvariant();
        return raw switch
        {
            "valid_ratio" => "valid_ratio",
            "full" => "full",
            _ => "full"
        };
    }

    private string ResolveTeacherBackend()
    {
        var raw = (GetStringOrNull("Global.teacher_backend") ?? "paddle").Trim().ToLowerInvariant();
        return raw switch
        {
            "paddle" => "paddle",
            "torch" => "torch",
            "onnx" => "onnx",
            "auto" => "auto",
            _ => "paddle"
        };
    }

    private static float Clamp(float value, float min, float max)
    {
        return Math.Clamp(value, min, max);
    }

    // 扩展方法：用于 ConfigDrivenRecTrainer
    public string GetArchitectureString(string path, string fallback)
    {
        return GetString($"Architecture.{path}", fallback);
    }

    public int GetArchitectureInt(string path, int fallback)
    {
        return GetInt($"Architecture.{path}", fallback);
    }

    public string GetOptimizerString(string path, string fallback)
    {
        return GetString($"Optimizer.{path}", fallback);
    }

    public float GetOptimizerFloat(string path, float fallback)
    {
        return GetFloat($"Optimizer.{path}", fallback);
    }

    public Dictionary<string, object> GetOptimizerLrConfig()
    {
        var lrObj = GetByPath("Optimizer.lr");
        if (lrObj is Dictionary<string, object?> dict)
        {
            return dict.ToDictionary(kv => kv.Key, kv => kv.Value ?? (object)string.Empty);
        }
        if (lrObj is IReadOnlyDictionary<string, object?> roDict)
        {
            return roDict.ToDictionary(kv => kv.Key, kv => kv.Value ?? (object)string.Empty);
        }

        return new Dictionary<string, object>();
    }

    public string GetLossString(string path, string fallback)
    {
        return GetString($"Loss.{path}", fallback);
    }

    public Dictionary<string, object> GetLossConfig()
    {
        var lossObj = GetByPath("Loss");
        if (lossObj is Dictionary<string, object?> dict)
        {
            return dict.ToDictionary(kv => kv.Key, kv => kv.Value ?? (object)string.Empty);
        }
        if (lossObj is IReadOnlyDictionary<string, object?> roDict)
        {
            return roDict.ToDictionary(kv => kv.Key, kv => kv.Value ?? (object)string.Empty);
        }

        return new Dictionary<string, object>();
    }

    public bool HasArchitectureConfig()
    {
        var arch = GetByPath("Architecture");
        return arch is Dictionary<string, object?> dict &&
               (dict.ContainsKey("Backbone") || dict.ContainsKey("Head"));
    }

    // 公共方法：用于 ConfigDrivenRecTrainer
    public int GetConfigInt(string path, int fallback)
    {
        return GetInt(path, fallback);
    }

    public bool GetConfigBool(string path, bool fallback)
    {
        return GetBool(path, fallback);
    }

    /// <summary>
    /// 是否使用 MultiScaleDataSet。
    /// </summary>
    public bool UseMultiScale
    {
        get
        {
            return GetString("Train.dataset.name", "")
                .Equals("MultiScaleDataSet", StringComparison.OrdinalIgnoreCase);
        }
    }

    /// <summary>
    /// MultiScale 候选宽度列表。
    /// 从配置 Train.dataset.ds_width 读取，默认 [320, 256, 192, 128, 96, 64]。
    /// </summary>
    public int[] MultiScaleWidths
    {
        get
        {
            if (TryGetSamplerScales(out var scales) && scales.Length > 0)
            {
                var widths = scales
                    .Select(s => s.Width)
                    .Where(w => w > 0)
                    .Distinct()
                    .ToArray();
                if (widths.Length > 0)
                {
                    return widths;
                }
            }

            var raw = GetByPath("Train.dataset.ds_width");
            if (raw is bool b && !b)
            {
                return [RecImageShape.W];
            }
            if (raw is IList<object?> list)
            {
                var parsed = list
                    .Select(x => int.TryParse(x?.ToString(), out var v) ? v : 0)
                    .Where(v => v > 0)
                    .ToArray();
                return parsed.Length == 0 ? [RecImageShape.W] : parsed;
            }
            return [320, 256, 192, 128, 96, 64];
        }
    }

    public int[] MultiScaleHeights
    {
        get
        {
            if (TryGetSamplerScales(out var scales) && scales.Length > 0)
            {
                var heights = scales
                    .Select(s => s.Height)
                    .Where(h => h > 0)
                    .Distinct()
                    .ToArray();
                if (heights.Length > 0)
                {
                    return heights;
                }
            }

            return [RecImageShape.H];
        }
    }

    public string GetConfigString(string path, string fallback)
    {
        return GetString(path, fallback);
    }

    public object? GetByPathPublic(string path)
    {
        return GetByPath(path);
    }

    private (int Start, int Interval) ParseEvalBatchStep()
    {
        var raw = GetByPath("Global.eval_batch_step");
        if (raw is IList<object?> list && list.Count >= 2 &&
            int.TryParse(list[0]?.ToString(), out var start) &&
            int.TryParse(list[1]?.ToString(), out var interval))
        {
            return (Math.Max(0, start), Math.Max(1, interval));
        }

        return (0, 2000);
    }

    private IReadOnlyList<float> ParseRatioList(string path)
    {
        var raw = GetByPath(path);
        if (raw is null)
        {
            return [1f];
        }

        if (raw is IList<object?> list)
        {
            var parsed = list
                .Select(ParseFloatOrNaN)
                .Where(v => !float.IsNaN(v))
                .ToArray();
            return parsed.Length == 0 ? [1f] : parsed;
        }

        var scalar = ParseFloatOrNaN(raw);
        return float.IsNaN(scalar) ? [1f] : [scalar];
    }

    private bool TryGetSamplerScales(out (int Width, int Height)[] scales)
    {
        scales = [];
        var raw = GetByPath("Train.sampler.scales");
        if (raw is not IList<object?> list || list.Count == 0)
        {
            return false;
        }

        var parsed = new List<(int Width, int Height)>();
        foreach (var item in list)
        {
            if (item is not IList<object?> shape || shape.Count < 2)
            {
                continue;
            }

            if (!int.TryParse(shape[0]?.ToString(), out var w) || !int.TryParse(shape[1]?.ToString(), out var h))
            {
                continue;
            }

            if (w > 0 && h > 0)
            {
                parsed.Add((w, h));
            }
        }

        scales = parsed.ToArray();
        return scales.Length > 0;
    }

    private static float ParseFloatOrNaN(object? raw)
    {
        if (raw is null)
        {
            return float.NaN;
        }

        if (raw is float f)
        {
            return f;
        }

        if (raw is double d)
        {
            return (float)d;
        }

        if (raw is int i)
        {
            return i;
        }

        return float.TryParse(
            raw.ToString(),
            System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture,
            out var parsed)
            ? parsed
            : float.NaN;
    }
}
