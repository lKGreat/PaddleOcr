using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Config;
using PaddleOcr.Training;

namespace PaddleOcr.Tests;

public sealed class RecTrainingParityTests
{
    [Fact]
    public async Task TrainRec_ConfigDriven_Should_Save_PerStep_And_Best_Model()
    {
        var root = FindRepoRoot();
        var samples = Path.Combine(root, "assets", "samples", "tiny_rec");
        var trainLabel = Path.Combine(samples, "train.txt");
        var evalLabel = Path.Combine(samples, "test.txt");
        var dict = Path.Combine(samples, "dict.txt");
        var output = Path.Combine(Path.GetTempPath(), "pocr_rec_train_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(output);

        var cfgPath = Path.Combine(output, "rec_train.yml");
        await File.WriteAllTextAsync(cfgPath,
            $$"""
              Global:
                epoch_num: 1
                save_model_dir: {{output.Replace("\\", "/")}}
                resume_training: false
                seed: 1024
                deterministic: true
                device: cpu
                min_improve_delta: 0.000001
                print_batch_step: 1
                log_smooth_window: 2
                save_batch_model: true
                save_epoch_step: 1
                eval_batch_step: [0, 1]
                cal_metric_during_train: true
                calc_epoch_interval: 1
                character_dict_path: {{dict.Replace("\\", "/")}}
                max_text_length: 2
                use_space_char: true
              Optimizer:
                name: Adam
                lr:
                  name: Cosine
                  learning_rate: 0.001
                  warmup_epoch: 0
              Architecture:
                model_type: rec
                algorithm: SVTR_LCNet
                in_channels: 3
                Backbone:
                  name: MobileNetV1Enhance
                Neck:
                  name: SequenceEncoder
                  encoder_type: reshape
                Head:
                  name: CTCHead
                  hidden_size: 48
              Loss:
                name: CTCLoss
              Train:
                dataset:
                  data_dir: {{samples.Replace("\\", "/")}}
                  label_file_list:
                    - {{trainLabel.Replace("\\", "/")}}
                  transforms:
                    - DecodeImage:
                        img_mode: BGR
                        channel_first: false
                    - MultiLabelEncode:
                        gtc_encode: NRTRLabelEncode
                    - RecResizeImg:
                        image_shape: [3, 48, 192]
                    - KeepKeys:
                        keep_keys:
                        - image
                        - label_ctc
                        - label_gtc
                        - length
                        - valid_ratio
                loader:
                  batch_size_per_card: 1
              Eval:
                dataset:
                  data_dir: {{samples.Replace("\\", "/")}}
                  label_file_list:
                    - {{evalLabel.Replace("\\", "/")}}
                  transforms:
                    - DecodeImage:
                        img_mode: BGR
                        channel_first: false
                    - MultiLabelEncode:
                        gtc_encode: NRTRLabelEncode
                    - RecResizeImg:
                        image_shape: [3, 48, 192]
                    - KeepKeys:
                        keep_keys:
                        - image
                        - label_ctc
                        - label_gtc
                        - length
                        - valid_ratio
                loader:
                  batch_size_per_card: 1
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
        result.Success.Should().BeTrue(result.Message);

        File.Exists(Path.Combine(output, "best.pt")).Should().BeTrue();
        File.Exists(Path.Combine(output, "latest.pt")).Should().BeTrue();
        Directory.EnumerateFiles(output, "iter_step_*.pt").Should().NotBeEmpty();
        File.Exists(Path.Combine(output, "train_trace.jsonl")).Should().BeTrue();
        File.Exists(Path.Combine(output, "train_epoch_summary.jsonl")).Should().BeTrue();
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
