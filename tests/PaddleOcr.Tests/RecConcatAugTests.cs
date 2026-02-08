using System;
using System.IO;
using System.Linq;
using FluentAssertions;
using PaddleOcr.Data;
using PaddleOcr.Data.LabelEncoders;
using PaddleOcr.Training;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace PaddleOcr.Tests;

public sealed class RecConcatAugTests
{
    [Fact]
    public void RecConcatAug_Should_Concatenate_Text_And_Respect_MaxWhRatio()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "pocr_concat_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmp);

        // create two small images 20x10
        var img1 = Path.Combine(tmp, "img1.png");
        var img2 = Path.Combine(tmp, "img2.png");
        using (var im = new Image<Rgb24>(20, 10))
        {
            im.Save(img1);
            im.Save(img2);
        }

        var labelFile = Path.Combine(tmp, "labels.txt");
        File.WriteAllText(labelFile, $"{img1}\tab{Environment.NewLine}{img2}\tcd");

        var ctc = new CTCLabelEncode(10, null, true);
        var nrtr = new NRTRLabelEncode(10, null, true);
        var resizer = new RecResizeImg();
        var concatOptions = new RecConcatAugmentOptions(true, 1.0f, 1, MaxWhRatio: 10f, ImageHeight: 32);

        var dataset = new ConfigRecDataset(
            new[] { labelFile },
            dataDir: tmp,
            targetH: 32,
            targetW: 64,
            maxTextLength: 10,
            ctcEncoder: ctc,
            gtcEncoder: nrtr,
            resizer: resizer,
            enableAugmentation: false,
            useMultiScale: false,
            delimiter: "\t",
            seed: 123,
            concatOptions: concatOptions);

        var batch = dataset.GetBatches(batchSize: 1, shuffle: false, rng: new Random(7)).First();

        // extract first sample labels (flattened length = maxTextLength)
        var firstLabelSlice = batch.LabelCtc.Take(10).ToArray();
        var nonZero = firstLabelSlice.Count(v => v > 0);

        nonZero.Should().BeGreaterThan(2, "concat should append another word");
        batch.ValidRatios.All(v => v <= 1.0f + 1e-6).Should().BeTrue();
        batch.Width.Should().Be(64);
        batch.Height.Should().Be(32);
    }

    [Fact]
    public void RecConcatAug_When_Ratio_Exceeded_Should_Stop_Concat()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "pocr_concat_break_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmp);

        var img1 = Path.Combine(tmp, "img1.png");
        var img2 = Path.Combine(tmp, "img2.png");
        using (var im = new Image<Rgb24>(20, 10))
        {
            im.Save(img1);
            im.Save(img2);
        }

        var labelFile = Path.Combine(tmp, "labels.txt");
        File.WriteAllText(labelFile, $"{img1}\tab{Environment.NewLine}{img2}\tcd");

        var ctc = new CTCLabelEncode(10, null, true);
        var nrtr = new NRTRLabelEncode(10, null, true);
        var concatOptions = new RecConcatAugmentOptions(true, 1.0f, 2, MaxWhRatio: 1f, ImageHeight: 32);

        var dataset = new ConfigRecDataset(
            new[] { labelFile },
            dataDir: tmp,
            targetH: 32,
            targetW: 64,
            maxTextLength: 10,
            ctcEncoder: ctc,
            gtcEncoder: nrtr,
            resizer: new RecResizeImg(),
            enableAugmentation: false,
            useMultiScale: false,
            delimiter: "\t",
            seed: 123,
            concatOptions: concatOptions);

        var batch = dataset.GetBatches(batchSize: 1, shuffle: false, rng: new Random(7)).First();
        var firstLabelSlice = batch.LabelCtc.Take(10).ToArray();
        var nonZero = firstLabelSlice.Count(v => v > 0);

        nonZero.Should().Be(2, "ratio overflow should stop concatenation");
    }

    [Fact]
    public void GetBatches_DropLast_Should_Discard_Last_Incomplete_Batch()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "pocr_drop_last_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmp);

        var img = Path.Combine(tmp, "img.png");
        using (var im = new Image<Rgb24>(20, 10))
        {
            im.Save(img);
        }

        var labelFile = Path.Combine(tmp, "labels.txt");
        File.WriteAllText(labelFile, $"{img}\ta{Environment.NewLine}{img}\tb{Environment.NewLine}{img}\tc");

        var dataset = new ConfigRecDataset(
            new[] { labelFile },
            dataDir: tmp,
            targetH: 32,
            targetW: 64,
            maxTextLength: 10,
            ctcEncoder: new CTCLabelEncode(10, null, true),
            gtcEncoder: new NRTRLabelEncode(10, null, true),
            resizer: new RecResizeImg(),
            enableAugmentation: false,
            useMultiScale: false,
            delimiter: "\t",
            seed: 123,
            concatOptions: RecConcatAugmentOptions.Disabled);

        var kept = dataset.GetBatches(batchSize: 2, shuffle: false, rng: new Random(1), dropLast: true).ToList();
        var notDropped = dataset.GetBatches(batchSize: 2, shuffle: false, rng: new Random(1), dropLast: false).ToList();

        kept.Should().HaveCount(1);
        kept[0].Batch.Should().Be(2);
        notDropped.Sum(x => x.Batch).Should().Be(3);
    }
}
