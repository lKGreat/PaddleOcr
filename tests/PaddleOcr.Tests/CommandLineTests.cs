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

    [Fact]
    public void Parse_Should_Read_Doctor_Subcommand()
    {
        var cmd = CommandLine.Parse(["doctor", "check-models", "--det_model_dir", "a.onnx"]);
        cmd.Root.Should().Be("doctor");
        cmd.Sub.Should().Be("check-models");
        cmd.Options["--det_model_dir"].Should().Be("a.onnx");
    }

    [Fact]
    public void Parse_Should_Read_Doctor_Parity_Subcommand()
    {
        var cmd = CommandLine.Parse(["doctor", "parity-table-kie", "-c", "cfg.yml", "--mode", "table"]);
        cmd.Root.Should().Be("doctor");
        cmd.Sub.Should().Be("parity-table-kie");
        cmd.ConfigPath.Should().Be("cfg.yml");
        cmd.Options["--mode"].Should().Be("table");
    }

    [Fact]
    public void Parse_Should_Read_Doctor_TrainDetReady_Subcommand()
    {
        var cmd = CommandLine.Parse(["doctor", "train-det-ready", "-c", "det.yml"]);
        cmd.Root.Should().Be("doctor");
        cmd.Sub.Should().Be("train-det-ready");
        cmd.ConfigPath.Should().Be("det.yml");
    }

    [Fact]
    public void Parse_Should_Read_Doctor_DetParity_Subcommand()
    {
        var cmd = CommandLine.Parse(["doctor", "det-parity", "-c", "det.yml"]);
        cmd.Root.Should().Be("doctor");
        cmd.Sub.Should().Be("det-parity");
        cmd.ConfigPath.Should().Be("det.yml");
    }

    [Fact]
    public void Parse_Should_Read_Doctor_VerifyRecPaddle_Subcommand()
    {
        var cmd = CommandLine.Parse(["doctor", "verify-rec-paddle", "--model_dir", "./m", "--python_exe", "python"]);
        cmd.Root.Should().Be("doctor");
        cmd.Sub.Should().Be("verify-rec-paddle");
        cmd.Options["--model_dir"].Should().Be("./m");
        cmd.Options["--python_exe"].Should().Be("python");
    }

    [Fact]
    public void Parse_Should_Read_Doctor_TrainDevice_Subcommand()
    {
        var cmd = CommandLine.Parse(["doctor", "train-device", "-c", "cfg.yml", "--device", "cuda", "--use_amp", "true"]);
        cmd.Root.Should().Be("doctor");
        cmd.Sub.Should().Be("train-device");
        cmd.ConfigPath.Should().Be("cfg.yml");
        cmd.Options["--device"].Should().Be("cuda");
        cmd.Options["--use_amp"].Should().Be("true");
    }

    [Fact]
    public void Parse_Should_Read_Benchmark_Subcommand()
    {
        var cmd = CommandLine.Parse(["benchmark", "run", "--scenario", "infer:system", "--iterations", "20"]);
        cmd.Root.Should().Be("benchmark");
        cmd.Sub.Should().Be("run");
        cmd.Options["--scenario"].Should().Be("infer:system");
        cmd.Options["--iterations"].Should().Be("20");
    }

    [Fact]
    public void Parse_Should_Read_Plugin_Subcommand()
    {
        var cmd = CommandLine.Parse(["plugin", "validate-package", "--package_dir", "./plugins/demo"]);
        cmd.Root.Should().Be("plugin");
        cmd.Sub.Should().Be("validate-package");
        cmd.Options["--package_dir"].Should().Be("./plugins/demo");
    }

    [Fact]
    public void Parse_Should_Read_Plugin_LoadRuntime_Subcommand()
    {
        var cmd = CommandLine.Parse(["plugin", "load-runtime", "--package_dir", "./plugins/p1"]);
        cmd.Sub.Should().Be("load-runtime");
        cmd.Options["--package_dir"].Should().Be("./plugins/p1");
    }

    [Fact]
    public void Parse_Should_Read_Plugin_VerifyTrust_Subcommand()
    {
        var cmd = CommandLine.Parse(["plugin", "verify-trust", "--package_dir", "./plugins/p2"]);
        cmd.Sub.Should().Be("verify-trust");
        cmd.Options["--package_dir"].Should().Be("./plugins/p2");
    }
}
