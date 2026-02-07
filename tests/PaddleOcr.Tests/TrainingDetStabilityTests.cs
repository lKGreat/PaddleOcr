using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Config;
using PaddleOcr.Core.Cli;
using PaddleOcr.Training;
using System.Text.Json;

namespace PaddleOcr.Tests;

public sealed class TrainingDetStabilityTests
{
    [Fact]
    public async Task TrainDet_Should_Write_Audit_History_And_Extended_Result()
    {
        var root = FindRepoRoot();
        var samples = Path.Combine(root, "assets", "samples", "tiny_det");
        var trainLabel = Path.Combine(samples, "train.txt");
        var evalLabel = Path.Combine(samples, "test.txt");
        var output = Path.Combine(Path.GetTempPath(), "pocr_det_train_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(output);

        var cfgPath = Path.Combine(output, "det_train.yml");
        await File.WriteAllTextAsync(cfgPath,
            $$"""
              Global:
                epoch_num: 1
                save_model_dir: {{output.Replace("\\", "/")}}
                resume_training: false
                seed: 1024
                deterministic: true
                device: cpu
                nan_guard: true
                min_improve_delta: 0.0001
              Architecture:
                model_type: det
              Optimizer:
                lr:
                  learning_rate: 0.001
              Loss:
                det_shrink_ratio: 0.45
                det_thresh_min: 0.2
                det_thresh_max: 0.8
                det_shrink_loss_weight: 1.0
                det_threshold_loss_weight: 0.4
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
                loader:
                  batch_size_per_card: 2
              Eval:
                dataset:
                  data_dir: {{samples.Replace("\\", "/")}}
                  label_file_list:
                    - {{evalLabel.Replace("\\", "/")}}
                  transforms:
                    - ResizeTextImg:
                        size: 128
                loader:
                  batch_size_per_card: 2
              """);

        var loader = new ConfigLoader();
        var context = new PaddleOcr.Core.Cli.ExecutionContext(
            NullLogger.Instance,
            ["train", "-c", cfgPath],
            cfgPath,
            loader.Load(cfgPath),
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
            []);

        var executor = new TrainingExecutor();
        var result = await executor.ExecuteAsync("train", context);
        result.Success.Should().BeTrue();

        var trainResultPath = Path.Combine(output, "train_result.json");
        var runSummaryPath = Path.Combine(output, "train_run_summary.json");
        var historyPath = Path.Combine(output, "det_train_history.jsonl");
        var auditPath = Path.Combine(output, "det_data_audit.json");
        File.Exists(trainResultPath).Should().BeTrue();
        File.Exists(runSummaryPath).Should().BeTrue();
        File.Exists(historyPath).Should().BeTrue();
        File.Exists(auditPath).Should().BeTrue();

        using var trainDoc = JsonDocument.Parse(await File.ReadAllTextAsync(trainResultPath));
        trainDoc.RootElement.TryGetProperty("Seed", out _).Should().BeTrue();
        trainDoc.RootElement.TryGetProperty("Device", out _).Should().BeTrue();
        trainDoc.RootElement.TryGetProperty("NanDetected", out _).Should().BeTrue();

        using var summaryDoc = JsonDocument.Parse(await File.ReadAllTextAsync(runSummaryPath));
        summaryDoc.RootElement.TryGetProperty("Seed", out _).Should().BeTrue();
        summaryDoc.RootElement.TryGetProperty("Device", out _).Should().BeTrue();

        var historyLines = await File.ReadAllLinesAsync(historyPath);
        historyLines.Should().NotBeEmpty();
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
