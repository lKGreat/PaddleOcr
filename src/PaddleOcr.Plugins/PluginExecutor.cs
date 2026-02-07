using System.Text.Json;
using System.Text.Json.Serialization;
using PaddleOcr.Core.Cli;

namespace PaddleOcr.Plugins;

public sealed class PluginExecutor : ICommandExecutor
{
    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!subCommand.Equals("validate-package", StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(CommandResult.Fail("plugin supports: validate-package"));
        }

        var packageDir = context.Options.TryGetValue("--package_dir", out var path) && !string.IsNullOrWhiteSpace(path)
            ? path
            : Directory.GetCurrentDirectory();

        var result = PluginPackageValidator.Validate(packageDir);
        if (!result.Valid)
        {
            return Task.FromResult(CommandResult.Fail("plugin validate-package failed:\n" + string.Join('\n', result.Errors)));
        }

        return Task.FromResult(CommandResult.Ok($"plugin validate-package passed: {packageDir} ({result.Manifest!.Name}@{result.Manifest.Version})"));
    }
}

public static class PluginPackageValidator
{
    public static PluginValidationResult Validate(string packageDir)
    {
        var errors = new List<string>();
        if (!Directory.Exists(packageDir))
        {
            errors.Add($"package directory not found: {packageDir}");
            return new PluginValidationResult(false, null, errors);
        }

        var manifestPath = Path.Combine(packageDir, "plugin.json");
        if (!File.Exists(manifestPath))
        {
            errors.Add($"missing manifest: {manifestPath}");
            return new PluginValidationResult(false, null, errors);
        }

        PluginManifest? manifest;
        try
        {
            var json = File.ReadAllText(manifestPath);
            manifest = JsonSerializer.Deserialize<PluginManifest>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
        }
        catch (Exception ex)
        {
            errors.Add($"manifest parse error: {ex.Message}");
            return new PluginValidationResult(false, null, errors);
        }

        if (manifest is null)
        {
            errors.Add("manifest parse error: empty manifest");
            return new PluginValidationResult(false, null, errors);
        }

        ValidateRequired(manifest.SchemaVersion, "schema_version", errors);
        ValidateRequired(manifest.Name, "name", errors);
        ValidateRequired(manifest.Version, "version", errors);
        ValidateRequired(manifest.Type, "type", errors);
        ValidateRequired(manifest.EntryAssembly, "entry_assembly", errors);
        ValidateRequired(manifest.EntryType, "entry_type", errors);

        var allowedTypes = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "preprocess",
            "postprocess",
            "metric"
        };
        if (!string.IsNullOrWhiteSpace(manifest.Type) && !allowedTypes.Contains(manifest.Type))
        {
            errors.Add($"unsupported type: {manifest.Type} (expected preprocess|postprocess|metric)");
        }

        if (!string.IsNullOrWhiteSpace(manifest.SchemaVersion) && !manifest.SchemaVersion.StartsWith("1.", StringComparison.Ordinal))
        {
            errors.Add($"unsupported schema_version: {manifest.SchemaVersion} (expected 1.x)");
        }

        if (!string.IsNullOrWhiteSpace(manifest.EntryAssembly))
        {
            var assemblyPath = Path.Combine(packageDir, manifest.EntryAssembly);
            if (!File.Exists(assemblyPath))
            {
                errors.Add($"entry assembly not found: {assemblyPath}");
            }
        }

        if (manifest.Files is { Count: > 0 })
        {
            foreach (var file in manifest.Files)
            {
                var fullPath = Path.Combine(packageDir, file);
                if (!File.Exists(fullPath))
                {
                    errors.Add($"declared file not found: {fullPath}");
                }
            }
        }

        return new PluginValidationResult(errors.Count == 0, manifest, errors);
    }

    private static void ValidateRequired(string? value, string name, ICollection<string> errors)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            errors.Add($"missing required field: {name}");
        }
    }
}

public sealed record PluginValidationResult(bool Valid, PluginManifest? Manifest, IReadOnlyList<string> Errors);

public sealed class PluginManifest
{
    [JsonPropertyName("schema_version")]
    public string? SchemaVersion { get; set; }

    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("version")]
    public string? Version { get; set; }

    [JsonPropertyName("type")]
    public string? Type { get; set; }

    [JsonPropertyName("entry_assembly")]
    public string? EntryAssembly { get; set; }

    [JsonPropertyName("entry_type")]
    public string? EntryType { get; set; }

    [JsonPropertyName("files")]
    public IReadOnlyList<string>? Files { get; set; }
}
