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
    public string? RecCharDictPath => ResolvePathOrNull(GetStringOrNull("Global.rec_char_dict_path"));

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

    private IReadOnlyList<string> GetStringList(string path)
    {
        if (GetByPath(path) is List<object?> list)
        {
            return list.Where(x => x is not null).Select(x => x!.ToString() ?? string.Empty).ToList();
        }

        return [];
    }

    private string ResolvePath(string path) => Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(_configDir, path));

    private string? ResolvePathOrNull(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        return ResolvePath(path);
    }
}
