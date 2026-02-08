using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Config;
using PaddleOcr.Core.Cli;
using PaddleOcr.Inference;
using PaddleOcr.Tools;

namespace PaddleOcr.Tests;

public sealed class PocrAppTests
{
    [Fact]
    public async Task RunAsync_Should_Route_To_Infer_Executor()
    {
        var training = new ProbeExecutor("training");
        var infer = new ProbeExecutor("inference");
        var export = new ProbeExecutor("export");
        var service = new ProbeExecutor("service");
        var e2e = new ProbeExecutor("e2e");
        var benchmark = new ProbeExecutor("benchmark");
        var plugin = new ProbeExecutor("plugin");

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            training,
            infer,
            export,
            service,
            e2e,
            benchmark,
            plugin);

        var code = await app.RunAsync(["infer", "system", "--image_dir", "./imgs"]);

        code.Should().Be(0);
        infer.Calls.Should().ContainSingle(c => c == "system");
    }

    private sealed class ProbeExecutor(string name) : ICommandExecutor
    {
        public List<string> Calls { get; } = [];

        public Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
        {
            Calls.Add(subCommand);
            return Task.FromResult(CommandResult.Ok($"{name}:{subCommand}"));
        }
    }

    [Fact]
    public async Task RunAsync_Should_Fail_For_System_Without_Rec_Model()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new InferenceExecutor(),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["infer", "system", "--image_dir", "./imgs", "--use_onnx", "true"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_Should_Route_Convert_CheckJsonModel_To_Export()
    {
        var export = new ProbeExecutor("export");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            export,
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["convert", "check-json-model", "--json_model_dir", "./m"]);

        code.Should().Be(0);
        export.Calls.Should().ContainSingle(c => c == "convert:check-json-model");
    }

    [Fact]
    public async Task RunAsync_Should_Route_ExportPdmodel_To_Export()
    {
        var export = new ProbeExecutor("export");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            export,
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var dir = Path.Combine(Path.GetTempPath(), "pocr_exportpd_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var cfg = Path.Combine(dir, "cfg.yml");
        await File.WriteAllTextAsync(cfg, "Architecture:\n  model_type: rec\nGlobal:\n  save_model_dir: ./out\n");

        var code = await app.RunAsync(["export-pdmodel", "-c", cfg]);

        code.Should().Be(0);
        export.Calls.Should().ContainSingle(c => c == "export-pdmodel");
    }

    [Fact]
    public async Task RunAsync_ConfigCheck_Should_Validate_Config()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_cfg_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var cfg = Path.Combine(dir, "ok.yml");
        await File.WriteAllTextAsync(cfg, "Global:\n  save_model_dir: ./out\nArchitecture:\n  model_type: cls\n");

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["config", "check", "-c", cfg]);
        code.Should().Be(0);
    }

    [Fact]
    public async Task RunAsync_ConfigDiff_Should_Report_Diff()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_cfg_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var a = Path.Combine(dir, "a.yml");
        var b = Path.Combine(dir, "b.yml");
        await File.WriteAllTextAsync(a, "Global:\n  epoch_num: 1\nArchitecture:\n  model_type: cls\n");
        await File.WriteAllTextAsync(b, "Global:\n  epoch_num: 2\nArchitecture:\n  model_type: cls\n");

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["config", "diff", "--base", a, "--target", b]);
        code.Should().Be(0);
    }

    [Fact]
    public async Task RunAsync_DoctorCheckModels_Should_Fail_When_File_Missing()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "check-models", "--det_model_dir", "not_exists.onnx"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_Should_Route_Benchmark_Run_To_Benchmark_Executor()
    {
        var benchmark = new ProbeExecutor("benchmark");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            benchmark,
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["benchmark", "run", "--scenario", "infer:system"]);

        code.Should().Be(0);
        benchmark.Calls.Should().ContainSingle(c => c == "run");
    }

    [Fact]
    public async Task RunAsync_Should_Route_Plugin_ValidatePackage_To_Plugin_Executor()
    {
        var plugin = new ProbeExecutor("plugin");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            plugin);

        var code = await app.RunAsync(["plugin", "validate-package", "--package_dir", "."]);

        code.Should().Be(0);
        plugin.Calls.Should().ContainSingle(c => c == "validate-package");
    }

    [Fact]
    public async Task RunAsync_Should_Route_Plugin_LoadRuntime_To_Plugin_Executor()
    {
        var plugin = new ProbeExecutor("plugin");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            plugin);

        var code = await app.RunAsync(["plugin", "load-runtime", "--package_dir", "."]);

        code.Should().Be(0);
        plugin.Calls.Should().ContainSingle(c => c == "load-runtime");
    }

    [Fact]
    public async Task RunAsync_Should_Route_Plugin_VerifyTrust_To_Plugin_Executor()
    {
        var plugin = new ProbeExecutor("plugin");
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            plugin);

        var code = await app.RunAsync(["plugin", "verify-trust", "--package_dir", "."]);

        code.Should().Be(0);
        plugin.Calls.Should().ContainSingle(c => c == "verify-trust");
    }

    [Fact]
    public async Task RunAsync_DoctorParityTableKie_Should_Fail_Without_Config()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "parity-table-kie"]);

        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_DoctorParityTableKie_Should_Pass_With_Required_Files()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_parity_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var table = Path.Combine(dir, "table.onnx");
        var det = Path.Combine(dir, "det.onnx");
        var rec = Path.Combine(dir, "rec.onnx");
        var kie = Path.Combine(dir, "kie.onnx");
        var ser = Path.Combine(dir, "ser.onnx");
        var re = Path.Combine(dir, "re.onnx");
        var dict = Path.Combine(dir, "dict.txt");
        foreach (var f in new[] { table, det, rec, kie, ser, re, dict })
        {
            await File.WriteAllTextAsync(f, "x");
        }

        var cfg = Path.Combine(dir, "cfg.yml");
        await File.WriteAllTextAsync(cfg,
            $"""
             Global:
               table_model_dir: {table}
               det_model_dir: {det}
               rec_model_dir: {rec}
               rec_char_dict_path: {dict}
               kie_model_dir: {kie}
               ser_model_dir: {ser}
               re_model_dir: {re}
             Architecture:
               model_type: kie
             """);

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "parity-table-kie", "-c", cfg, "--mode", "all"]);
        code.Should().Be(0);
    }

    [Fact]
    public async Task RunAsync_DoctorTrainDetReady_Should_Fail_Without_Config()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "train-det-ready"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_DoctorTrainDevice_Should_Fail_Without_Config()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "train-device"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_DoctorTrainDevice_Should_Pass_For_Cpu_Config()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_train_device_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var cfg = Path.Combine(dir, "cpu.yml");
            await File.WriteAllTextAsync(cfg,
                """
                Global:
                  device: cpu
                Architecture:
                  model_type: rec
                """);

            var app = new PocrApp(
                NullLogger.Instance,
                new ConfigLoader(),
                new ProbeExecutor("training"),
                new ProbeExecutor("inference"),
                new ProbeExecutor("export"),
                new ProbeExecutor("service"),
                new ProbeExecutor("e2e"),
                new ProbeExecutor("benchmark"),
                new ProbeExecutor("plugin"));

            var code = await app.RunAsync(["doctor", "train-device", "-c", cfg]);
            code.Should().Be(0);
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, true);
            }
        }
    }

    [Fact]
    public async Task RunAsync_DoctorTrainDetReady_Should_Pass_For_TinyDet_Config()
    {
        var root = FindRepoRoot();
        var samples = Path.Combine(root, "assets", "samples", "tiny_det");
        var trainLabel = Path.Combine(samples, "train.txt");
        var evalLabel = Path.Combine(samples, "test.txt");

        var dir = Path.Combine(Path.GetTempPath(), "pocr_det_ready_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var cfg = Path.Combine(dir, "det_ready.yml");
        await File.WriteAllTextAsync(cfg,
            $$"""
              Global:
                save_model_dir: {{dir.Replace("\\", "/")}}/out
              Architecture:
                model_type: det
              Train:
                dataset:
                  data_dir: {{samples.Replace("\\", "/")}}
                  label_file_list:
                    - {{trainLabel.Replace("\\", "/")}}
                  invalid_sample_policy: skip
                  min_valid_samples: 1
                  transforms:
                    - ResizeTextImg:
                        size: 128
              Eval:
                dataset:
                  data_dir: {{samples.Replace("\\", "/")}}
                  label_file_list:
                    - {{evalLabel.Replace("\\", "/")}}
                  transforms:
                    - ResizeTextImg:
                        size: 128
              """);

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "train-det-ready", "-c", cfg]);
        code.Should().Be(0);
    }

    [Fact]
    public async Task RunAsync_DoctorDetParity_Should_Fail_Without_Config()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "det-parity"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_DoctorDetParity_Should_Pass_For_Minimal_Det_Config()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_det_parity_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var cfg = Path.Combine(dir, "det_parity.yml");
        await File.WriteAllTextAsync(cfg,
            """
            Global:
              det_db_thresh: 0.3
              det_db_box_thresh: 0.6
              det_db_unclip_ratio: 1.5
              det_limit_side_len: 640
              det_box_type: quad
            Architecture:
              model_type: det
              algorithm: DB
            PostProcess:
              name: DBPostProcess
            """);

        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "det-parity", "-c", cfg]);
        code.Should().Be(0);
    }

    [Fact]
    public async Task RunAsync_DoctorVerifyRecPaddle_Should_Fail_Without_ModelDir()
    {
        var app = new PocrApp(
            NullLogger.Instance,
            new ConfigLoader(),
            new ProbeExecutor("training"),
            new ProbeExecutor("inference"),
            new ProbeExecutor("export"),
            new ProbeExecutor("service"),
            new ProbeExecutor("e2e"),
            new ProbeExecutor("benchmark"),
            new ProbeExecutor("plugin"));

        var code = await app.RunAsync(["doctor", "verify-rec-paddle"]);
        code.Should().Be(2);
    }

    [Fact]
    public async Task RunAsync_DoctorVerifyRecPaddle_Should_Fail_When_Model_Files_Missing()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_verify_rec_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var app = new PocrApp(
                NullLogger.Instance,
                new ConfigLoader(),
                new ProbeExecutor("training"),
                new ProbeExecutor("inference"),
                new ProbeExecutor("export"),
                new ProbeExecutor("service"),
                new ProbeExecutor("e2e"),
                new ProbeExecutor("benchmark"),
                new ProbeExecutor("plugin"));

            var code = await app.RunAsync(["doctor", "verify-rec-paddle", "--model_dir", dir]);
            code.Should().Be(2);
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, true);
            }
        }
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "PaddleOcr.slnx")))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        throw new InvalidOperationException("repo root not found");
    }
}
