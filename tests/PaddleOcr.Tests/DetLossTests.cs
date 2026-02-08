using FluentAssertions;
using PaddleOcr.Training.Det.Losses;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;

namespace PaddleOcr.Tests;

public sealed class DetLossTests
{
    [Fact]
    public void DiceLoss_CorrectOutput()
    {
        // Arrange
        using var diceLoss = new DiceLoss(eps: 1e-6f);
        using var pred = torch.tensor(new float[,] { { 0.9f, 0.1f }, { 0.2f, 0.8f } });
        using var gt = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var mask = torch.ones_like(pred);

        // Act
        using var loss = diceLoss.Forward(pred, gt, mask);

        // Assert
        loss.shape.Should().BeEmpty(); // Scalar
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f).And.BeLessThan(1f);
    }

    [Fact]
    public void DiceLoss_PerfectPrediction_ZeroLoss()
    {
        // Arrange
        using var diceLoss = new DiceLoss(eps: 1e-6f);
        using var pred = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var gt = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var mask = torch.ones_like(pred);

        // Act
        using var loss = diceLoss.Forward(pred, gt, mask);

        // Assert
        var lossValue = loss.ToSingle();
        lossValue.Should().BeApproximately(0f, 1e-5f);
    }

    [Fact]
    public void DiceLoss_WithWeights()
    {
        // Arrange
        using var diceLoss = new DiceLoss(eps: 1e-6f);
        using var pred = torch.tensor(new float[,] { { 0.9f, 0.1f }, { 0.2f, 0.8f } });
        using var gt = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var mask = torch.ones_like(pred);
        using var weights = torch.tensor(new float[,] { { 2f, 1f }, { 1f, 2f } });

        // Act
        using var loss = diceLoss.Forward(pred, gt, mask, weights);

        // Assert
        loss.shape.Should().BeEmpty(); // Scalar
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f).And.BeLessThan(1f);
    }

    [Fact]
    public void MaskL1Loss_CorrectOutput()
    {
        // Arrange
        using var maskL1Loss = new MaskL1Loss(eps: 1e-6f);
        using var pred = torch.tensor(new float[,] { { 0.7f, 0.3f }, { 0.4f, 0.6f } });
        using var gt = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var mask = torch.ones_like(pred);

        // Act
        using var loss = maskL1Loss.Forward(pred, gt, mask);

        // Assert
        loss.shape.Should().BeEmpty(); // Scalar
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f).And.BeLessThan(1f);
    }

    [Fact]
    public void MaskL1Loss_PerfectPrediction_ZeroLoss()
    {
        // Arrange
        using var maskL1Loss = new MaskL1Loss(eps: 1e-6f);
        using var pred = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var gt = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var mask = torch.ones_like(pred);

        // Act
        using var loss = maskL1Loss.Forward(pred, gt, mask);

        // Assert
        var lossValue = loss.ToSingle();
        lossValue.Should().BeApproximately(0f, 1e-5f);
    }

    [Fact]
    public void MaskL1Loss_WithPartialMask()
    {
        // Arrange
        using var maskL1Loss = new MaskL1Loss(eps: 1e-6f);
        using var pred = torch.tensor(new float[,] { { 0.7f, 0.3f }, { 0.4f, 0.6f } });
        using var gt = torch.tensor(new float[,] { { 1f, 0f }, { 0f, 1f } });
        using var mask = torch.tensor(new float[,] { { 1f, 1f }, { 0f, 0f } }); // Only first row is valid

        // Act
        using var loss = maskL1Loss.Forward(pred, gt, mask);

        // Assert
        loss.shape.Should().BeEmpty();
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void BalanceLoss_OHEM_Logic()
    {
        // Arrange
        using var balanceLoss = new BalanceLoss(balanceLoss: true, negativeRatio: 2f);
        using var pred = torch.randn(2, 4, 4);  // [B, H, W]
        using var gt = torch.zeros(2, 4, 4);
        gt[0, 1, 1] = 1f;  // One positive sample
        gt[1, 2, 2] = 1f;  // One positive sample
        using var mask = torch.ones_like(pred);

        // Act
        using var loss = balanceLoss.Forward(pred, gt, mask);

        // Assert
        loss.shape.Should().BeEmpty(); // Scalar
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void BalanceLoss_WithoutBalance()
    {
        // Arrange
        using var balanceLoss = new BalanceLoss(balanceLoss: false);
        using var pred = torch.randn(2, 4, 4);
        using var gt = torch.zeros(2, 4, 4);
        gt[0, 1, 1] = 1f;
        using var mask = torch.ones_like(pred);

        // Act
        using var loss = balanceLoss.Forward(pred, gt, mask);

        // Assert
        loss.shape.Should().BeEmpty();
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void DBLoss_WeightedCombination()
    {
        // Arrange
        const float alpha = 5f;
        const float beta = 10f;
        using var dbLoss = new DBLoss(alpha: alpha, beta: beta, balanceLoss: true, ohemRatio: 3f);

        // Create mock predictions: [B, 3, H, W]
        using var maps = torch.randn(2, 3, 8, 8);
        var predictions = new Dictionary<string, Tensor> { ["maps"] = maps };

        // Create mock ground truth
        using var gtShrinkMap = torch.zeros(2, 8, 8);
        gtShrinkMap[0, 2, 2] = 1f;
        gtShrinkMap[1, 3, 3] = 1f;
        using var gtShrinkMask = torch.ones(2, 8, 8);
        using var gtThresholdMap = torch.rand(2, 8, 8);
        using var gtThresholdMask = torch.ones(2, 8, 8);

        var batch = new Dictionary<string, Tensor>
        {
            ["shrink_map"] = gtShrinkMap,
            ["shrink_mask"] = gtShrinkMask,
            ["threshold_map"] = gtThresholdMap,
            ["threshold_mask"] = gtThresholdMask
        };

        // Act
        var losses = dbLoss.Forward(predictions, batch);

        // Assert
        losses.Should().ContainKey("loss");
        losses.Should().ContainKey("loss_shrink_maps");
        losses.Should().ContainKey("loss_threshold_maps");
        losses.Should().ContainKey("loss_binary_maps");

        using var totalLoss = losses["loss"];
        totalLoss.shape.Should().BeEmpty(); // Scalar
        var lossValue = totalLoss.ToSingle();
        lossValue.Should().BeGreaterThan(0f);

        // Verify all component losses are valid
        using var shrinkLoss = losses["loss_shrink_maps"];
        using var thresholdLoss = losses["loss_threshold_maps"];
        using var binaryLoss = losses["loss_binary_maps"];

        shrinkLoss.ToSingle().Should().BeGreaterThan(0f);
        thresholdLoss.ToSingle().Should().BeGreaterThan(0f);
        binaryLoss.ToSingle().Should().BeGreaterThan(0f);
    }

    [Fact]
    public void DBLoss_GradientFlow()
    {
        // Arrange
        using var dbLoss = new DBLoss(alpha: 5f, beta: 10f, balanceLoss: true, ohemRatio: 3f);

        // Create predictions with requires_grad=true
        using var maps = torch.randn(new long[] { 1, 3, 4, 4 }, requires_grad: true);
        var predictions = new Dictionary<string, Tensor> { ["maps"] = maps };

        using var gtShrinkMap = torch.zeros(1, 4, 4);
        gtShrinkMap[0, 1, 1] = 1f;
        using var gtShrinkMask = torch.ones(1, 4, 4);
        using var gtThresholdMap = torch.rand(1, 4, 4);
        using var gtThresholdMask = torch.ones(1, 4, 4);

        var batch = new Dictionary<string, Tensor>
        {
            ["shrink_map"] = gtShrinkMap,
            ["shrink_mask"] = gtShrinkMask,
            ["threshold_map"] = gtThresholdMap,
            ["threshold_mask"] = gtThresholdMask
        };

        // Act
        var losses = dbLoss.Forward(predictions, batch);
        using var loss = losses["loss"];
        loss.backward();

        // Assert
        maps.grad.Should().NotBeNull();
        var gradNorm = maps.grad!.norm().ToSingle();
        gradNorm.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void DBLoss_MissingMapsKey_ThrowsException()
    {
        // Arrange
        using var dbLoss = new DBLoss();
        var predictions = new Dictionary<string, Tensor>(); // Missing "maps" key

        using var shrinkMap = torch.zeros(1, 4, 4);
        using var shrinkMask = torch.ones(1, 4, 4);
        using var thresholdMap = torch.zeros(1, 4, 4);
        using var thresholdMask = torch.ones(1, 4, 4);

        var batch = new Dictionary<string, Tensor>
        {
            ["shrink_map"] = shrinkMap,
            ["shrink_mask"] = shrinkMask,
            ["threshold_map"] = thresholdMap,
            ["threshold_mask"] = thresholdMask
        };

        // Act & Assert
        var act = () => dbLoss.Forward(predictions, batch);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*must contain 'maps' key*");
    }

    [Fact]
    public void DBLoss_InvalidMapsShape_ThrowsException()
    {
        // Arrange
        using var dbLoss = new DBLoss();
        using var invalidMaps = torch.randn(1, 2, 4, 4); // Wrong channel count (should be 3)
        var predictions = new Dictionary<string, Tensor> { ["maps"] = invalidMaps };

        using var shrinkMap = torch.zeros(1, 4, 4);
        using var shrinkMask = torch.ones(1, 4, 4);
        using var thresholdMap = torch.zeros(1, 4, 4);
        using var thresholdMask = torch.ones(1, 4, 4);

        var batch = new Dictionary<string, Tensor>
        {
            ["shrink_map"] = shrinkMap,
            ["shrink_mask"] = shrinkMask,
            ["threshold_map"] = thresholdMap,
            ["threshold_mask"] = thresholdMask
        };

        // Act & Assert
        var act = () => dbLoss.Forward(predictions, batch);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*must have shape [B, 3, H, W]*");
    }

    [Fact]
    public void DetLossBuilder_BuildDBLoss()
    {
        // Arrange & Act
        var loss = PaddleOcr.Training.Det.DetLossBuilder.BuildLoss("DBLoss");

        // Assert
        loss.Should().NotBeNull();
        loss.Should().BeOfType<DBLoss>();
    }

    [Fact]
    public void DetLossBuilder_WithConfig()
    {
        // Arrange
        var config = new Dictionary<string, object>
        {
            ["alpha"] = 7f,
            ["beta"] = 12f,
            ["balance_loss"] = false,
            ["ohem_ratio"] = 5f
        };

        // Act
        var loss = PaddleOcr.Training.Det.DetLossBuilder.BuildLoss("DB", config);

        // Assert
        loss.Should().NotBeNull();
        loss.Should().BeOfType<DBLoss>();
    }
}
