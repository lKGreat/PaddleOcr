using FluentAssertions;
using PaddleOcr.Training;

namespace PaddleOcr.Tests;

public sealed class DetMetricEvaluatorTests
{
    [Fact]
    public void EvaluateSingle_Should_Report_PerfectMatch()
    {
        var pred = NewMask(8, 8, (2, 2, 5, 5));
        var gt = NewMask(8, 8, (2, 2, 5, 5));

        var summary = DetMetricEvaluator.EvaluateSingle(pred, gt, 8, 8, 0.5f);

        summary.TruePositive.Should().Be(1);
        summary.FalsePositive.Should().Be(0);
        summary.FalseNegative.Should().Be(0);
        summary.Precision.Should().Be(1f);
        summary.Recall.Should().Be(1f);
        summary.Fscore.Should().Be(1f);
        summary.MeanIou.Should().BeApproximately(1f, 1e-6f);
    }

    [Fact]
    public void EvaluateSingle_Should_Report_FalsePositive_And_Miss()
    {
        var pred = NewMask(8, 8, (1, 1, 3, 3), (5, 5, 6, 6));
        var gt = NewMask(8, 8, (1, 1, 3, 3));

        var summary = DetMetricEvaluator.EvaluateSingle(pred, gt, 8, 8, 0.5f);

        summary.TruePositive.Should().Be(1);
        summary.FalsePositive.Should().Be(1);
        summary.FalseNegative.Should().Be(0);
        summary.Precision.Should().BeApproximately(0.5f, 1e-6f);
        summary.Recall.Should().BeApproximately(1f, 1e-6f);
        summary.Fscore.Should().BeApproximately(2f * 0.5f / 1.5f, 1e-6f);
    }

    [Fact]
    public void EvaluateSingle_Should_Report_Miss_When_IoU_Below_Threshold()
    {
        var pred = NewMask(8, 8, (0, 0, 2, 2));
        var gt = NewMask(8, 8, (4, 4, 6, 6));

        var summary = DetMetricEvaluator.EvaluateSingle(pred, gt, 8, 8, 0.5f);

        summary.TruePositive.Should().Be(0);
        summary.FalsePositive.Should().Be(1);
        summary.FalseNegative.Should().Be(1);
        summary.Fscore.Should().Be(0f);
        summary.MeanIou.Should().Be(0f);
    }

    private static bool[] NewMask(int width, int height, params (int X1, int Y1, int X2, int Y2)[] boxes)
    {
        var mask = new bool[width * height];
        foreach (var (x1, y1, x2, y2) in boxes)
        {
            for (var y = y1; y <= y2; y++)
            {
                for (var x = x1; x <= x2; x++)
                {
                    mask[y * width + x] = true;
                }
            }
        }

        return mask;
    }
}
