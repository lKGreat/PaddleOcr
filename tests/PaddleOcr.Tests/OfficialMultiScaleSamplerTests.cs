using FluentAssertions;
using PaddleOcr.Training;

namespace PaddleOcr.Tests;

public sealed class OfficialMultiScaleSamplerTests
{
    [Fact]
    public void Sampler_Should_Build_Full_Batches_With_Wrap_When_Needed()
    {
        var sampler = new OfficialMultiScaleSampler(
            sampleCount: 3,
            scales: new[] { (Width: 320, Height: 32) },
            firstBatchSize: 2,
            fixBatchSize: true,
            dividedFactor: new[] { 8, 16 },
            isTraining: true,
            dsWidth: false);

        var batches = sampler.BuildEpochBatches(new Random(1));

        batches.Should().HaveCount(2);
        batches[0].SampleIndices.Should().HaveCount(2);
        batches[1].SampleIndices.Should().HaveCount(2);
        batches.All(x => x.Width == 320 && x.Height == 32).Should().BeTrue();
    }

    [Fact]
    public void Sampler_With_DsWidth_Should_Use_Ratio_Current_For_Width()
    {
        var sampler = new OfficialMultiScaleSampler(
            sampleCount: 3,
            scales: new[] { (Width: 320, Height: 32) },
            firstBatchSize: 2,
            fixBatchSize: true,
            dividedFactor: new[] { 8, 16 },
            isTraining: true,
            dsWidth: true,
            whRatios: new[] { 10f, 2f, 3f },
            whRatioSort: new[] { 1, 2, 0 },
            maxW: 480f);

        var batches = sampler.BuildEpochBatches(new Random(1));

        batches.Should().NotBeEmpty();
        batches.Any(x => x.Width == 64).Should().BeTrue();
        batches.Any(x => x.SampleIndices.SequenceEqual(new[] { 1, 2 })).Should().BeTrue();
    }
}
