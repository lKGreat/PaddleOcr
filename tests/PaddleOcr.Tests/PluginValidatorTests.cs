using FluentAssertions;
using PaddleOcr.Plugins;

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
              "files": [ "demo.dll" ]
            }
            """);

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
              "alias_of": "rgb-chw-01"
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
              "alias_of": "db-multibox"
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

    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_plugin_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
