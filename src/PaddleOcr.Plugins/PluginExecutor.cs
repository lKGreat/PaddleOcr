using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using PaddleOcr.Core.Cli;
using PaddleOcr.Inference.Onnx;

namespace PaddleOcr.Plugins;

public sealed class PluginExecutor : ICommandExecutor
{
    public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (subCommand.Equals("validate-package", StringComparison.OrdinalIgnoreCase))
        {
            var packageDir = context.Options.TryGetValue("--package_dir", out var path) && !string.IsNullOrWhiteSpace(path)
                ? path
                : Directory.GetCurrentDirectory();
            var requireTrust = ParseBool(context, "--require_trust");

            var result = PluginPackageValidator.Validate(packageDir, requireTrust);
            if (!result.Valid)
            {
                return Task.FromResult(CommandResult.Fail("plugin validate-package failed:\n" + string.Join('\n', result.Errors)));
            }

            return Task.FromResult(CommandResult.Ok($"plugin validate-package passed: {packageDir} ({result.Manifest!.Name}@{result.Manifest.Version})"));
        }

        if (subCommand.Equals("verify-trust", StringComparison.OrdinalIgnoreCase))
        {
            var packageDir = context.Options.TryGetValue("--package_dir", out var path) && !string.IsNullOrWhiteSpace(path)
                ? path
                : Directory.GetCurrentDirectory();
            var result = PluginPackageValidator.Validate(packageDir, requireTrust: true);
            if (!result.Valid)
            {
                return Task.FromResult(CommandResult.Fail("plugin verify-trust failed:\n" + string.Join('\n', result.Errors)));
            }

            return Task.FromResult(CommandResult.Ok($"plugin verify-trust passed: {packageDir}"));
        }

        if (subCommand.Equals("load-runtime", StringComparison.OrdinalIgnoreCase))
        {
            if (!context.Options.TryGetValue("--package_dir", out var packageDir) || string.IsNullOrWhiteSpace(packageDir))
            {
                return Task.FromResult(CommandResult.Fail("plugin load-runtime requires --package_dir"));
            }
            var allowUntrusted = ParseBool(context, "--allow_untrusted");

            var loaded = PluginRuntimeLoader.LoadPackage(packageDir, out var message, requireTrust: !allowUntrusted);
            return Task.FromResult(loaded
                ? CommandResult.Ok($"plugin load-runtime passed: {message}")
                : CommandResult.Fail($"plugin load-runtime failed: {message}"));
        }

        if (subCommand.Equals("load-runtime-dir", StringComparison.OrdinalIgnoreCase))
        {
            if (!context.Options.TryGetValue("--plugins_root", out var pluginsRoot) || string.IsNullOrWhiteSpace(pluginsRoot))
            {
                return Task.FromResult(CommandResult.Fail("plugin load-runtime-dir requires --plugins_root"));
            }
            var allowUntrusted = ParseBool(context, "--allow_untrusted");

            var summary = PluginRuntimeLoader.LoadDirectory(pluginsRoot, requireTrust: !allowUntrusted);
            if (summary.Failed > 0)
            {
                return Task.FromResult(CommandResult.Fail($"plugin load-runtime-dir failed: loaded={summary.Loaded}, failed={summary.Failed}\n{string.Join('\n', summary.Messages)}"));
            }

            return Task.FromResult(CommandResult.Ok($"plugin load-runtime-dir passed: loaded={summary.Loaded}, failed={summary.Failed}"));
        }

        return Task.FromResult(CommandResult.Fail("plugin supports: validate-package | verify-trust | load-runtime | load-runtime-dir"));
    }

    private static bool ParseBool(PaddleOcr.Core.Cli.ExecutionContext context, string key)
    {
        return context.Options.TryGetValue(key, out var value)
               && bool.TryParse(value, out var flag)
               && flag;
    }
}

public static class PluginRuntimeLoader
{
    private static readonly PluginFaultIsolationPolicy FaultPolicy = PluginFaultIsolationPolicy.FailOpenWithFallback;

    public static bool LoadPackage(string packageDir, out string message, bool requireTrust = true)
    {
        var validation = PluginPackageValidator.Validate(packageDir, requireTrust);
        if (!validation.Valid || validation.Manifest is null)
        {
            message = string.Join("; ", validation.Errors);
            return false;
        }

        var manifest = validation.Manifest;
        var bindingName = string.IsNullOrWhiteSpace(manifest.RuntimeName) ? manifest.Name! : manifest.RuntimeName!;
        if (!string.IsNullOrWhiteSpace(manifest.AliasOf))
        {
            return RegisterAlias(manifest, bindingName, out message);
        }

        var assemblyPath = Path.Combine(packageDir, manifest.EntryAssembly!);
        try
        {
            var asm = Assembly.LoadFrom(assemblyPath);
            var type = asm.GetType(manifest.EntryType!, throwOnError: true);
            var instance = Activator.CreateInstance(type!);
            if (instance is null)
            {
                message = $"unable to instantiate entry type: {manifest.EntryType}";
                return false;
            }

            return RegisterInstance(manifest, bindingName, instance, out message);
        }
        catch (Exception ex)
        {
            message = ex.Message;
            return false;
        }
    }

    public static bool RegisterRuntimeInstance(PluginManifest manifest, object instance, out string message)
    {
        var bindingName = string.IsNullOrWhiteSpace(manifest.RuntimeName) ? manifest.Name ?? "plugin-runtime" : manifest.RuntimeName!;
        return RegisterInstance(manifest, bindingName, instance, out message);
    }

    public static PluginLoadSummary LoadDirectory(string pluginsRoot, bool requireTrust = true)
    {
        if (!Directory.Exists(pluginsRoot))
        {
            return new PluginLoadSummary(0, 1, [$"plugins root not found: {pluginsRoot}"]);
        }

        var dirs = Directory.EnumerateDirectories(pluginsRoot, "*", SearchOption.TopDirectoryOnly).ToList();
        var loaded = 0;
        var failed = 0;
        var messages = new List<string>();
        foreach (var dir in dirs)
        {
            var ok = LoadPackage(dir, out var msg, requireTrust);
            if (ok)
            {
                loaded++;
            }
            else
            {
                failed++;
                messages.Add($"{dir}: {msg}");
            }
        }

        return new PluginLoadSummary(loaded, failed, messages);
    }

    private static bool RegisterAlias(PluginManifest manifest, string bindingName, out string message)
    {
        var alias = manifest.AliasOf!;
        if (manifest.Type!.Equals("preprocess", StringComparison.OrdinalIgnoreCase))
        {
            var fn = InferencePreprocessRegistry.GetInputBuilder(alias);
            InferencePreprocessRegistry.RegisterInputBuilder(bindingName, fn);
            message = $"{bindingName} -> preprocess:{alias}";
            return true;
        }

        if (!manifest.Type.Equals("postprocess", StringComparison.OrdinalIgnoreCase))
        {
            message = $"alias registration does not support type: {manifest.Type}";
            return false;
        }

        var target = NormalizeRuntimeTarget(manifest.RuntimeTarget);
        if (target == "det")
        {
            var fn = InferenceComponentRegistry.GetDetPostprocessor(alias);
            InferenceComponentRegistry.RegisterDetPostprocessor(bindingName, fn);
            message = $"{bindingName} -> postprocess.det:{alias}";
            return true;
        }

        if (target == "rec")
        {
            var fn = InferenceComponentRegistry.GetRecPostprocessor(alias);
            InferenceComponentRegistry.RegisterRecPostprocessor(bindingName, fn);
            message = $"{bindingName} -> postprocess.rec:{alias}";
            return true;
        }

        if (target == "cls")
        {
            var fn = InferenceComponentRegistry.GetClsPostprocessor(alias);
            InferenceComponentRegistry.RegisterClsPostprocessor(bindingName, fn);
            message = $"{bindingName} -> postprocess.cls:{alias}";
            return true;
        }

        message = "postprocess alias requires runtime_target in det|rec|cls";
        return false;
    }

    private static bool RegisterInstance(PluginManifest manifest, string bindingName, object instance, out string message)
    {
        var hooks = instance as IPluginLifecycleHooks;
        if (manifest.Type!.Equals("preprocess", StringComparison.OrdinalIgnoreCase))
        {
            if (instance is not IInferencePreprocessPlugin plugin)
            {
                message = $"entry type does not implement {nameof(IInferencePreprocessPlugin)}";
                return false;
            }

            var fallback = InferencePreprocessRegistry.GetInputBuilder("rgb-chw-01");
            InferencePreprocessRegistry.RegisterInputBuilder(bindingName, (img, dims, h, w) =>
            {
                try
                {
                    return plugin.BuildInput(img, dims, h, w);
                }
                catch (Exception ex)
                {
                    hooks?.OnError("preprocess", ex);
                    if (FaultPolicy == PluginFaultIsolationPolicy.FailOpenWithFallback)
                    {
                        return fallback(img, dims, h, w);
                    }

                    throw;
                }
            });
            hooks?.OnLoaded(bindingName);
            message = $"{bindingName} -> preprocess:{plugin.Name}";
            return true;
        }

        if (!manifest.Type.Equals("postprocess", StringComparison.OrdinalIgnoreCase))
        {
            message = $"runtime load does not support plugin type: {manifest.Type}";
            return false;
        }

        var target = NormalizeRuntimeTarget(manifest.RuntimeTarget);
        if (target == "det")
        {
            if (instance is not IDetPostprocessPlugin det)
            {
                message = $"entry type does not implement {nameof(IDetPostprocessPlugin)}";
                return false;
            }

            var fallback = InferenceComponentRegistry.GetDetPostprocessor("db-multibox");
            InferenceComponentRegistry.RegisterDetPostprocessor(bindingName, (data, dims, width, height, thresh) =>
            {
                try
                {
                    return det.Postprocess(data, dims, width, height, thresh);
                }
                catch (Exception ex)
                {
                    hooks?.OnError("postprocess.det", ex);
                    if (FaultPolicy == PluginFaultIsolationPolicy.FailOpenWithFallback)
                    {
                        return fallback(data, dims, width, height, thresh);
                    }

                    throw;
                }
            });
            hooks?.OnLoaded(bindingName);
            message = $"{bindingName} -> postprocess.det:{det.Name}";
            return true;
        }

        if (target == "rec")
        {
            if (instance is not IRecPostprocessPlugin rec)
            {
                message = $"entry type does not implement {nameof(IRecPostprocessPlugin)}";
                return false;
            }

            var fallback = InferenceComponentRegistry.GetRecPostprocessor("ctc-greedy");
            InferenceComponentRegistry.RegisterRecPostprocessor(bindingName, (data, dims, charset) =>
            {
                try
                {
                    return rec.Postprocess(data, dims, charset);
                }
                catch (Exception ex)
                {
                    hooks?.OnError("postprocess.rec", ex);
                    if (FaultPolicy == PluginFaultIsolationPolicy.FailOpenWithFallback)
                    {
                        return fallback(data, dims, charset);
                    }

                    throw;
                }
            });
            hooks?.OnLoaded(bindingName);
            message = $"{bindingName} -> postprocess.rec:{rec.Name}";
            return true;
        }

        if (target == "cls")
        {
            if (instance is not IClsPostprocessPlugin cls)
            {
                message = $"entry type does not implement {nameof(IClsPostprocessPlugin)}";
                return false;
            }

            var fallback = InferenceComponentRegistry.GetClsPostprocessor("argmax-softmax");
            InferenceComponentRegistry.RegisterClsPostprocessor(bindingName, (logits, labels) =>
            {
                try
                {
                    return cls.Postprocess(logits, labels);
                }
                catch (Exception ex)
                {
                    hooks?.OnError("postprocess.cls", ex);
                    if (FaultPolicy == PluginFaultIsolationPolicy.FailOpenWithFallback)
                    {
                        return fallback(logits, labels);
                    }

                    throw;
                }
            });
            hooks?.OnLoaded(bindingName);
            message = $"{bindingName} -> postprocess.cls:{cls.Name}";
            return true;
        }

        message = "postprocess plugin requires runtime_target in det|rec|cls";
        return false;
    }

    private static string NormalizeRuntimeTarget(string? target)
    {
        if (string.IsNullOrWhiteSpace(target))
        {
            return string.Empty;
        }

        return target.Trim().ToLowerInvariant() switch
        {
            "det" or "db" => "det",
            "rec" or "ctc" => "rec",
            "cls" => "cls",
            _ => target.Trim().ToLowerInvariant()
        };
    }
}

public enum PluginFaultIsolationPolicy
{
    FailOpenWithFallback = 0,
    FailFast = 1
}

public interface IPluginLifecycleHooks
{
    void OnLoaded(string bindingName);
    void OnError(string stage, Exception exception);
}

public static class PluginPackageValidator
{
    public static PluginValidationResult Validate(string packageDir, bool requireTrust = false)
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

        var useAlias = !string.IsNullOrWhiteSpace(manifest.AliasOf);
        if (!useAlias)
        {
            ValidateRequired(manifest.EntryAssembly, "entry_assembly", errors);
            ValidateRequired(manifest.EntryType, "entry_type", errors);
        }

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

        if (string.Equals(manifest.Type, "postprocess", StringComparison.OrdinalIgnoreCase) &&
            string.IsNullOrWhiteSpace(manifest.RuntimeTarget))
        {
            errors.Add("postprocess plugin requires runtime_target (det|rec|cls)");
        }

        if (!string.IsNullOrWhiteSpace(manifest.SchemaVersion) && !manifest.SchemaVersion.StartsWith("1.", StringComparison.Ordinal))
        {
            errors.Add($"unsupported schema_version: {manifest.SchemaVersion} (expected 1.x)");
        }

        ValidateTrust(manifest, packageDir, useAlias, requireTrust, errors);

        if (!useAlias && !string.IsNullOrWhiteSpace(manifest.EntryAssembly))
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

    private static void ValidateTrust(
        PluginManifest manifest,
        string packageDir,
        bool useAlias,
        bool requireTrust,
        ICollection<string> errors)
    {
        if (manifest.Trust is null)
        {
            if (requireTrust)
            {
                errors.Add("missing trust metadata: trust");
            }

            return;
        }

        var trust = manifest.Trust;
        if (!string.IsNullOrWhiteSpace(trust.Algorithm) &&
            !trust.Algorithm.Equals("sha256", StringComparison.OrdinalIgnoreCase))
        {
            errors.Add($"unsupported trust algorithm: {trust.Algorithm} (expected sha256)");
        }

        if (trust.TrustLevel is not null)
        {
            var levels = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "verified", "internal", "untrusted"
            };
            if (!levels.Contains(trust.TrustLevel))
            {
                errors.Add($"unsupported trust_level: {trust.TrustLevel}");
            }
        }

        if (!useAlias && !string.IsNullOrWhiteSpace(manifest.EntryAssembly))
        {
            var entry = Path.Combine(packageDir, manifest.EntryAssembly);
            if (File.Exists(entry))
            {
                if (requireTrust && string.IsNullOrWhiteSpace(trust.EntryAssemblySha256))
                {
                    errors.Add("missing trust.entry_assembly_sha256");
                }
                else if (!string.IsNullOrWhiteSpace(trust.EntryAssemblySha256))
                {
                    var actual = ComputeSha256(entry);
                    if (!actual.Equals(NormalizeHex(trust.EntryAssemblySha256), StringComparison.OrdinalIgnoreCase))
                    {
                        errors.Add("trust hash mismatch: entry_assembly_sha256");
                    }
                }
            }
        }

        if (trust.FilesSha256 is { Count: > 0 })
        {
            foreach (var pair in trust.FilesSha256)
            {
                var full = Path.Combine(packageDir, pair.Key);
                if (!File.Exists(full))
                {
                    errors.Add($"trust file not found: {full}");
                    continue;
                }

                var actual = ComputeSha256(full);
                if (!actual.Equals(NormalizeHex(pair.Value), StringComparison.OrdinalIgnoreCase))
                {
                    errors.Add($"trust hash mismatch: {pair.Key}");
                }
            }
        }
    }

    private static void ValidateRequired(string? value, string name, ICollection<string> errors)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            errors.Add($"missing required field: {name}");
        }
    }

    private static string ComputeSha256(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        var hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string NormalizeHex(string value)
    {
        return value.Replace("-", string.Empty, StringComparison.Ordinal).Trim().ToLowerInvariant();
    }
}

public sealed record PluginValidationResult(bool Valid, PluginManifest? Manifest, IReadOnlyList<string> Errors);
public sealed record PluginLoadSummary(int Loaded, int Failed, IReadOnlyList<string> Messages);

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

    [JsonPropertyName("runtime_target")]
    public string? RuntimeTarget { get; set; }

    [JsonPropertyName("runtime_name")]
    public string? RuntimeName { get; set; }

    [JsonPropertyName("alias_of")]
    public string? AliasOf { get; set; }

    [JsonPropertyName("trust")]
    public PluginTrustMetadata? Trust { get; set; }
}

public sealed class PluginTrustMetadata
{
    [JsonPropertyName("algorithm")]
    public string? Algorithm { get; set; }

    [JsonPropertyName("entry_assembly_sha256")]
    public string? EntryAssemblySha256 { get; set; }

    [JsonPropertyName("files_sha256")]
    public Dictionary<string, string>? FilesSha256 { get; set; }

    [JsonPropertyName("signer")]
    public string? Signer { get; set; }

    [JsonPropertyName("trust_level")]
    public string? TrustLevel { get; set; }
}
