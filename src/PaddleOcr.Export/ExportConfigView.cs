namespace PaddleOcr.Export;

public sealed class ExportConfigView
{
    private readonly IReadOnlyDictionary<string, object?> _root;
    private readonly string _configDir;

    public ExportConfigView(IReadOnlyDictionary<string, object?> root, string configPath)
    {
        _root = root;
        _configDir = Path.GetDirectoryName(Path.GetFullPath(configPath)) ?? Directory.GetCurrentDirectory();
    }

    public string ModelType => GetString("Architecture.model_type", "cls");
    public string SaveModelDir => ResolvePath(GetString("Global.save_model_dir", "./output"));
    public string SaveInferenceDir => ResolvePath(GetString("Global.save_inference_dir", "./inference"));
    public string? Checkpoints => ResolvePathOrNull(GetStringOrNull("Global.checkpoints"));
    public string? PretrainedModel => ResolvePathOrNull(GetStringOrNull("Global.pretrained_model"));
    public IReadOnlyList<string> LabelList => GetStringList("Global.label_list");
    public string? RecCharDictPath => ResolvePathOrNull(
        GetStringOrNull("Global.rec_char_dict_path") ??
        GetStringOrNull("Global.character_dict_path"));
    public int DetInputSize => GetDetInputSize();
    public IReadOnlyList<int> ClsImageShape => GetClsImageShape();

    /// <summary>
    /// 按点分路径读取配置值（公开访问）。
    /// </summary>
    public object? GetByPathPublic(string path) => GetByPath(path);

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

    private string GetString(string path, string fallback) => GetByPath(path)?.ToString() ?? fallback;
    private string? GetStringOrNull(string path) => GetByPath(path)?.ToString();
    private int GetInt(string path, int fallback) => int.TryParse(GetByPath(path)?.ToString(), out var v) ? v : fallback;

    private IReadOnlyList<string> GetStringList(string path)
    {
        if (GetByPath(path) is List<object?> list)
        {
            return list.Where(x => x is not null).Select(x => x!.ToString() ?? string.Empty).ToList();
        }

        return [];
    }

    private int GetDetInputSize()
    {
        if (GetByPath("Train.dataset.transforms") is not List<object?> transforms)
        {
            return 640;
        }

        foreach (var item in transforms)
        {
            if (item is Dictionary<string, object?> op &&
                op.TryGetValue("ResizeTextImg", out var cfgObj) &&
                cfgObj is Dictionary<string, object?> cfg)
            {
                return int.TryParse(cfg.GetValueOrDefault("size")?.ToString(), out var v) ? v : 640;
            }
        }

        return 640;
    }

    private IReadOnlyList<int> GetClsImageShape()
    {
        if (GetByPath("Train.dataset.transforms") is not List<object?> transforms)
        {
            return [3, 48, 192];
        }

        foreach (var item in transforms)
        {
            if (item is Dictionary<string, object?> op &&
                op.TryGetValue("ClsResizeImg", out var cfgObj) &&
                cfgObj is Dictionary<string, object?> cfg &&
                cfg.TryGetValue("image_shape", out var listObj) &&
                listObj is List<object?> list)
            {
                return list.Where(x => x is not null).Select(x => int.TryParse(x!.ToString(), out var v) ? v : 0).ToList();
            }
        }

        return [3, 48, 192];
    }

    private string ResolvePath(string path)
    {
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

        return ResolvePath(path);
    }
}
