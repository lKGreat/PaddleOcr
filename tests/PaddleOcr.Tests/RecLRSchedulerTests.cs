using FluentAssertions;
using PaddleOcr.Training.Rec.Schedulers;

namespace PaddleOcr.Tests;

public sealed class RecLRSchedulerTests
{
    [Fact]
    public void LinearWarmupCosine_Should_UseStepSchedule_WhenMaxStepsProvided()
    {
        var scheduler = new LinearWarmupCosine(
            initialLr: 0.001f,
            minLr: 0.00001f,
            warmupEpochs: 1,
            maxEpochs: 10,
            warmupSteps: 10,
            maxSteps: 100);

        scheduler.Step(step: 1, epoch: 1);
        scheduler.CurrentLR.Should().BeApproximately(0.000109, 1e-6);

        scheduler.Step(step: 10, epoch: 1);
        scheduler.CurrentLR.Should().BeApproximately(0.001, 1e-8);

        scheduler.Step(step: 100, epoch: 10);
        scheduler.CurrentLR.Should().BeApproximately(0.00001, 1e-8);
    }

    [Fact]
    public void CosineAnnealingDecay_Should_UseStepProgress_WhenMaxStepsProvided()
    {
        var scheduler = new CosineAnnealingDecay(
            initialLr: 0.001f,
            minLr: 0.00001f,
            maxEpochs: 10,
            maxSteps: 100);

        scheduler.Step(step: 50, epoch: 1);
        scheduler.CurrentLR.Should().BeApproximately(0.000505, 1e-6);

        scheduler.Step(step: 100, epoch: 10);
        scheduler.CurrentLR.Should().BeApproximately(0.00001, 1e-8);
    }
}
