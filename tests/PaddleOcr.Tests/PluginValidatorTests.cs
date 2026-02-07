using FluentAssertions;
using PaddleOcr.Inference.Onnx;
using PaddleOcr.Plugins;
using System.Security.Cryptography;

namespace PaddleOcr.Tests;

public sealed class PluginValidatorTests
{
    [Fact]
    public void Validate_Should_Pass_For_Valid_Package()
    {
        var dir = CreateTempDir();
        File.WriteAllText(Path.Combine(dir, "demo.dll"), string.Empty);
        File.WriteAllText(
            Path.Combine(dir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "demo-plugin",
              "version": "1.2.0",
              "type": "postprocess",
              "runtime_target": "cls",
              "entry_assembly": "demo.dll",
              "entry_type": "Demo.Plugin",
              "files": [ "demo.dll" ],
              "trust": {
                "algorithm": "sha256",
                "entry_assembly_sha256": "__HASH__",
                "trust_level": "verified"
              }
            }
            """.Replace("__HASH__", Sha256Hex(Path.Combine(dir, "demo.dll"))));

        var result = PluginPackageValidator.Validate(dir);
        result.Valid.Should().BeTrue();
    }

    [Fact]
    public void Validate_Should_Fail_When_Manifest_Is_Missing()
    {
        var dir = CreateTempDir();
        var result = PluginPackageValidator.Validate(dir);
        result.Valid.Should().BeFalse();
        result.Errors.Should().Contain(x => x.Contains("missing manifest", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Validate_Should_Fail_When_Entry_Assembly_Not_Found()
    {
        var dir = CreateTempDir();
        File.WriteAllText(
            Path.Combine(dir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "demo-plugin",
              "version": "1.2.0",
              "type": "preprocess",
              "entry_assembly": "missing.dll",
              "entry_type": "Demo.Plugin"
            }
            """);

        var result = PluginPackageValidator.Validate(dir);
        result.Valid.Should().BeFalse();
        result.Errors.Should().Contain(x => x.Contains("entry assembly not found", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void LoadPackage_Should_Register_Preprocess_Alias()
    {
        var dir = CreateTempDir();
        File.WriteAllText(
            Path.Combine(dir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "alias-pre",
              "version": "1.0.0",
              "type": "preprocess",
              "runtime_name": "rgb-chw-01-alias",
              "alias_of": "rgb-chw-01",
              "trust": {
                "trust_level": "internal"
              }
            }
            """);

        var ok = PluginRuntimeLoader.LoadPackage(dir, out var message);

        ok.Should().BeTrue();
        message.Should().Contain("rgb-chw-01-alias");
    }

    [Fact]
    public void LoadDirectory_Should_Summarize_Failures()
    {
        var root = CreateTempDir();
        var okDir = Path.Combine(root, "ok");
        var badDir = Path.Combine(root, "bad");
        Directory.CreateDirectory(okDir);
        Directory.CreateDirectory(badDir);

        File.WriteAllText(
            Path.Combine(okDir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "alias-det",
              "version": "1.0.0",
              "type": "postprocess",
              "runtime_target": "det",
              "runtime_name": "det-alias",
              "alias_of": "db-multibox",
              "trust": {
                "trust_level": "verified"
              }
            }
            """);
        File.WriteAllText(
            Path.Combine(badDir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "bad",
              "version": "1.0.0",
              "type": "postprocess"
            }
            """);

        var summary = PluginRuntimeLoader.LoadDirectory(root);
        summary.Loaded.Should().Be(1);
        summary.Failed.Should().Be(1);
    }

    [Fact]
    public void Validate_Should_Fail_When_Trust_Required_But_Missing()
    {
        var dir = CreateTempDir();
        File.WriteAllText(Path.Combine(dir, "demo.dll"), "x");
        File.WriteAllText(
            Path.Combine(dir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "demo-plugin",
              "version": "1.2.0",
              "type": "preprocess",
              "entry_assembly": "demo.dll",
              "entry_type": "Demo.Plugin"
            }
            """);

        var result = PluginPackageValidator.Validate(dir, requireTrust: true);
        result.Valid.Should().BeFalse();
        result.Errors.Should().Contain(x => x.Contains("missing trust metadata", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Validate_Should_Fail_On_Trust_Hash_Mismatch()
    {
        var dir = CreateTempDir();
        File.WriteAllText(Path.Combine(dir, "demo.dll"), "abc");
        File.WriteAllText(
            Path.Combine(dir, "plugin.json"),
            """
            {
              "schema_version": "1.0",
              "name": "demo-plugin",
              "version": "1.2.0",
              "type": "preprocess",
              "entry_assembly": "demo.dll",
              "entry_type": "Demo.Plugin",
              "trust": {
                "algorithm": "sha256",
                "entry_assembly_sha256": "0000",
                "trust_level": "verified"
              }
            }
            """);

        var result = PluginPackageValidator.Validate(dir, requireTrust: true);
        result.Valid.Should().BeFalse();
        result.Errors.Should().Contain(x => x.Contains("hash mismatch", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void RegisterRuntimeInstance_Should_Apply_Fault_Isolation_And_Hooks()
    {
        var plugin = new ThrowingDetPlugin();
        var manifest = new PluginManifest
        {
            SchemaVersion = "1.0",
            Name = "throw-det",
            Version = "1.0.0",
            Type = "postprocess",
            RuntimeTarget = "det",
            RuntimeName = "throw-det-runtime"
        };

        var ok = PluginRuntimeLoader.RegisterRuntimeInstance(manifest, plugin, out var message);
        ok.Should().BeTrue();
        message.Should().Contain("throw-det-runtime");
        plugin.LoadedCount.Should().Be(1);

        var fn = InferenceComponentRegistry.GetDetPostprocessor("throw-det-runtime");
        var boxes = fn([1f, 1f, 1f, 1f], [1, 1, 2, 2], 32, 32, 0.5f);
        boxes.Should().NotBeNull();
        plugin.ErrorCount.Should().Be(1);
    }

    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_plugin_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static string Sha256Hex(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private sealed class ThrowingDetPlugin : IDetPostprocessPlugin, IPluginLifecycleHooks
    {
        public string Name => "throwing-det";
        public int LoadedCount { get; private set; }
        public int ErrorCount { get; private set; }

        public List<OcrBox> Postprocess(float[] data, int[] dims, int width, int height, float thresh)
        {
            throw new InvalidOperationException("plugin failure");
        }

        public void OnLoaded(string bindingName)
        {
            LoadedCount++;
        }

        public void OnError(string stage, Exception exception)
        {
            ErrorCount++;
        }
    }
}
