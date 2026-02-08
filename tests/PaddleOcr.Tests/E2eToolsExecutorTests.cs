using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using PaddleOcr.Core.Cli;
using PaddleOcr.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Tests;

public sealed class E2eToolsExecutorTests
{
    [Fact]
    public async Task ConvertLabel_Should_Write_PerImage_Label_File()
    {
        var temp = CreateTempDir();
        try
        {
            var labelPath = Path.Combine(temp, "labels.txt");
            var saveFolder = Path.Combine(temp, "out");
            File.WriteAllText(
                labelPath,
                "imgs/a.jpg\t[{\"transcription\":\"hello\",\"points\":[[0,0],[10,0],[10,10],[0,10]]}]");

            var executor = new E2eToolsExecutor();
            var context = new PaddleOcr.Core.Cli.ExecutionContext(
                NullLogger.Instance,
                ["e2e", "convert-label"],
                null,
                new Dictionary<string, object?>(),
                new Dictionary<string, string>
                {
                    ["--label_path"] = labelPath,
                    ["--save_folder"] = saveFolder
                },
                []);

            var result = await executor.ExecuteAsync("convert-label", context);

            result.Success.Should().BeTrue();
            var file = Path.Combine(saveFolder, "a.jpg.txt");
            File.Exists(file).Should().BeTrue();
            var content = await File.ReadAllTextAsync(file);
            content.Should().Contain("hello");
            content.Should().Contain("0\t0\t10\t0\t10\t10\t0\t10");
        }
        finally
        {
            SafeDelete(temp);
        }
    }

    [Fact]
    public async Task Eval_Should_Report_Perfect_Fmeasure_For_Exact_Match()
    {
        var temp = CreateTempDir();
        try
        {
            var gt = Path.Combine(temp, "gt");
            var pred = Path.Combine(temp, "pred");
            Directory.CreateDirectory(gt);
            Directory.CreateDirectory(pred);
            var line = "0\t0\t10\t0\t10\t10\t0\t10\t0\thello";
            await File.WriteAllTextAsync(Path.Combine(gt, "a.jpg.txt"), line);
            await File.WriteAllTextAsync(Path.Combine(pred, "a.jpg.txt"), "0\t0\t10\t0\t10\t10\t0\t10\thello");

            var executor = new E2eToolsExecutor();
            var context = new PaddleOcr.Core.Cli.ExecutionContext(
                NullLogger.Instance,
                ["e2e", "eval", gt, pred],
                null,
                new Dictionary<string, object?>(),
                new Dictionary<string, string>(),
                []);

            var result = await executor.ExecuteAsync("eval", context);

            result.Success.Should().BeTrue();
            result.Message.Should().Contain("f=1.0000");
        }
        finally
        {
            SafeDelete(temp);
        }
    }

    [Fact]
    public async Task PrepareRecDet_Should_Generate_Train_And_Val_Label_Files()
    {
        var temp = CreateTempDir();
        try
        {
            var imagesRoot = Path.Combine(temp, "images");
            Directory.CreateDirectory(imagesRoot);
            var imgPath = Path.Combine(imagesRoot, "a.jpg");
            using (var img = new Image<Rgb24>(64, 32))
            {
                img.SaveAsJpeg(imgPath);
            }

            var labelPath = Path.Combine(temp, "det_labels.txt");
            await File.WriteAllTextAsync(
                labelPath,
                "a.jpg\t[{\"transcription\":\"abc\",\"points\":[[1,1],[30,1],[30,20],[1,20]]}]");

            var outDir = Path.Combine(temp, "rec_data");
            var executor = new E2eToolsExecutor();
            var context = new PaddleOcr.Core.Cli.ExecutionContext(
                NullLogger.Instance,
                ["e2e", "prepare-rec-det"],
                null,
                new Dictionary<string, object?>(),
                new Dictionary<string, string>
                {
                    ["--label_path"] = labelPath,
                    ["--image_root"] = imagesRoot,
                    ["--output_dir"] = outDir,
                    ["--train_ratio"] = "0.8",
                    ["--seed"] = "7"
                },
                []);

            var result = await executor.ExecuteAsync("prepare-rec-det", context);

            result.Success.Should().BeTrue();
            File.Exists(Path.Combine(outDir, "train_list.txt")).Should().BeTrue();
            File.Exists(Path.Combine(outDir, "val_list.txt")).Should().BeTrue();
            File.Exists(Path.Combine(outDir, "prepare_rec_summary.json")).Should().BeTrue();
            Directory.Exists(Path.Combine(outDir, "images")).Should().BeTrue();
            var train = await File.ReadAllLinesAsync(Path.Combine(outDir, "train_list.txt"));
            var val = await File.ReadAllLinesAsync(Path.Combine(outDir, "val_list.txt"));
            (train.Length + val.Length).Should().BeGreaterThan(0);
            train.Concat(val).Should().OnlyContain(x => x.Contains('\t'));
        }
        finally
        {
            SafeDelete(temp);
        }
    }

    private static string CreateTempDir()
    {
        var path = Path.Combine(Path.GetTempPath(), "pocr-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static void SafeDelete(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, true);
            }
        }
        catch
        {
            // ignored in tests
        }
    }
}
