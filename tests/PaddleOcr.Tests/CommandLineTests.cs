using FluentAssertions;
using PaddleOcr.Tools;

namespace PaddleOcr.Tests;

public sealed class CommandLineTests
{
    [Fact]
    public void Parse_Should_Read_Config_Overrides_And_Options()
    {
        var cmd = CommandLine.Parse([
            "train",
            "-c", "configs/det.yml",
            "-o", "Global.epoch_num=100", "Global.use_gpu=false",
            "--device", "cuda:0"
        ]);

        cmd.Root.Should().Be("train");
        cmd.ConfigPath.Should().Be("configs/det.yml");
        cmd.Overrides.Should().HaveCount(2);
        cmd.Options["--device"].Should().Be("cuda:0");
    }

    [Fact]
    public void Parse_Should_Read_Infer_Subcommand()
    {
        var cmd = CommandLine.Parse(["infer", "system", "--image_dir", "./imgs"]);
        cmd.Root.Should().Be("infer");
        cmd.Sub.Should().Be("system");
        cmd.Options["--image_dir"].Should().Be("./imgs");
    }
}

