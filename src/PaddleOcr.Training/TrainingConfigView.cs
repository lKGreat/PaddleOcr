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
    public float LearningRate => GetFloat("Optimizer.lr.learning_rate", 1e-3f);
    public int LrDecayStep => GetInt("Optimizer.lr.step_size", 0);
    public float LrDecayGamma => GetFloat("Optimizer.lr.gamma", 0.1f);
    public float GradClipNorm => GetFloat("Optimizer.grad_clip_norm", 0f);
    public int EarlyStopPatience => GetInt("Global.early_stop_patience", 0);
    public bool ResumeTraining => GetBool("Global.resume_training", true);
    public int Seed => GetInt("Global.seed", 1024);
    public bool Deterministic => GetBool("Global.deterministic", true);
    public string Device => GetString("Global.device", "cpu").Trim().ToLowerInvariant();
    public float MinImproveDelta => GetFloat("Global.min_improve_delta", 1e-4f);
    public bool NanGuard => GetBool("Global.nan_guard", true);
    public string SaveModelDir => ResolvePath(GetString("Global.save_model_dir", "./output/cls"));
    public string? Checkpoints => ResolvePathOrNull(GetStringOrNull("Global.checkpoints"));

    public string TrainLabelFile => ResolvePath(GetFirstString("Train.dataset.label_file_list"));
    public string EvalLabelFile => ResolvePath(GetFirstString("Eval.dataset.label_file_list"));
    public string DataDir => ResolvePath(GetString("Train.dataset.data_dir", "."));
    public string EvalDataDir => ResolvePath(GetString("Eval.dataset.data_dir", DataDir));
    public string InvalidSamplePolicy => GetString("Train.dataset.invalid_sample_policy", "skip").Trim().ToLowerInvariant();
    public int MinValidSamples => GetInt("Train.dataset.min_valid_samples", 1);

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
        return Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(_configDir, path));
    }

    private string? ResolvePathOrNull(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        return ResolvePath(path);
    }

    private string GetFirstString(string path)
    {
        if (GetByPath(path) is List<object?> list && list.Count > 0 && list[0] is not null)
        {
            return list[0]!.ToString() ?? string.Empty;
        }

        return string.Empty;
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

    private bool GetBool(string path, bool fallback)
    {
        return bool.TryParse(GetByPath(path)?.ToString(), out var v) ? v : fallback;
    }
}
