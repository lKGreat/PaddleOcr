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

    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_plugin_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
