using FluentAssertions;
using PaddleOcr.Training.Cls;
using PaddleOcr.Training.Cls.Backbones;
using PaddleOcr.Training.Cls.Heads;
using PaddleOcr.Training.Cls.Losses;
using TorchSharp;
using static TorchSharp.torch;
using Xunit;

namespace PaddleOcr.Tests;

public sealed class ClsModelBuilderTests
{
    [Fact]
    public void BuildBackbone_MobileNetV3Small_CorrectOutput()
    {
        // Arrange & Act
        var (backbone, outChannels) = ClsModelBuilder.BuildBackbone("MobileNetV3", inChannels: 3, scale: 1.0f);

        // Assert
        backbone.Should().NotBeNull();
        backbone.Should().BeOfType<ClsMobileNetV3>();
        outChannels.Should().BeGreaterThan(0);
    }

    [Fact]
    public void BuildBackbone_MobileNetV3Large_CorrectOutput()
    {
        // Arrange & Act
        var (backbone, outChannels) = ClsModelBuilder.BuildBackbone("MobileNetV3_large", inChannels: 3, scale: 0.5f);

        // Assert
        backbone.Should().NotBeNull();
        outChannels.Should().BeGreaterThan(0);
    }

    [Fact]
    public void BuildHead_ClsHead_CorrectShape()
    {
        // Arrange
        const int inChannels = 128;
        const int numClasses = 4;

        // Act
        var head = ClsModelBuilder.BuildHead("ClsHead", inChannels, numClasses);

        // Assert
        head.Should().NotBeNull();
        head.Should().BeOfType<ClsHead>();
        ((ClsHead)head).NumClasses.Should().Be(numClasses);
    }

    [Fact]
    public void ClsModel_ForwardPass_CorrectShape()
    {
        // Arrange
        const int batchSize = 2;
        const int numClasses = 2;
        const int height = 48;
        const int width = 192;

        var model = ClsModelBuilder.Build(
            backboneName: "MobileNetV3",
            headName: "ClsHead",
            numClasses: numClasses,
            inChannels: 3,
            scale: 0.35f);

        using var input = torch.randn(batchSize, 3, height, width);

        // Act
        model.eval(); // Set to eval mode (applies softmax)
        using var output = model.forward(input);

        // Assert
        output.shape.Should().Equal(new long[] { batchSize, numClasses });

        // In eval mode, output should be probabilities (sum to 1)
        for (int b = 0; b < batchSize; b++)
        {
            using var batchProbs = output[b];
            var sum = batchProbs.sum().ToSingle();
            sum.Should().BeApproximately(1.0f, 1e-5f);
        }
    }

    [Fact]
    public void ClsModel_TrainingMode_OutputsLogits()
    {
        // Arrange
        const int batchSize = 2;
        const int numClasses = 2;

        var model = ClsModelBuilder.Build(
            backboneName: "MobileNetV3",
            headName: "ClsHead",
            numClasses: numClasses,
            scale: 0.35f);

        using var input = torch.randn(batchSize, 3, 48, 192);

        // Act
        model.train(); // Set to training mode (no softmax)
        using var output = model.forward(input);

        // Assert
        output.shape.Should().Equal(new long[] { batchSize, numClasses });

        // In training mode, outputs are logits (not probabilities, won't sum to 1)
        for (int b = 0; b < batchSize; b++)
        {
            using var batchLogits = output[b];
            var sum = batchLogits.sum().ToSingle();
            // Logits don't sum to 1, so this should fail if we check for 1.0
            sum.Should().NotBe(1.0f);
        }
    }

    [Fact]
    public void ClsModel_ForwardDict_ReturnsCorrectFormat()
    {
        // Arrange
        var model = ClsModelBuilder.Build("MobileNetV3", "ClsHead", numClasses: 2, scale: 0.35f);
        using var input = torch.randn(1, 3, 48, 192);

        // Act
        model.eval();
        var result = model.ForwardDict(input);

        // Assert
        result.Should().ContainKey("predict");
        using var predict = result["predict"];
        predict.shape.Should().Equal(new long[] { 1, 2 });
    }

    [Fact]
    public void ClsModel_GradientFlow()
    {
        // Arrange
        var model = ClsModelBuilder.Build("MobileNetV3", "ClsHead", numClasses: 2, scale: 0.35f);
        using var input = torch.randn(new long[] { 1, 3, 48, 192 }, requires_grad: true);
        using var target = torch.tensor(new long[] { 0 });

        // Act
        model.train();
        using var output = model.forward(input);
        using var loss = torch.nn.functional.cross_entropy(output, target);
        loss.backward();

        // Assert
        input.grad.Should().NotBeNull();
        var gradNorm = input.grad!.norm().ToSingle();
        gradNorm.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void ClsLoss_CorrectOutput()
    {
        // Arrange
        using var clsLoss = new ClsLoss();
        using var predictions = torch.randn(4, 3); // 4 samples, 3 classes
        using var labels = torch.tensor(new long[] { 0, 1, 2, 1 });

        // Act
        var losses = clsLoss.Forward(predictions, labels);

        // Assert
        losses.Should().ContainKey("loss");
        using var loss = losses["loss"];
        loss.shape.Should().BeEmpty(); // Scalar
        var lossValue = loss.ToSingle();
        lossValue.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void ClsLoss_PerfectPrediction_LowLoss()
    {
        // Arrange
        using var clsLoss = new ClsLoss();

        // Create "perfect" predictions (high logit for correct class)
        using var predictions = torch.zeros(2, 3);
        predictions[0, 0] = 10f; // Class 0 predicted with high confidence
        predictions[1, 1] = 10f; // Class 1 predicted with high confidence

        using var labels = torch.tensor(new long[] { 0, 1 });

        // Act
        var losses = clsLoss.Forward(predictions, labels);

        // Assert
        using var loss = losses["loss"];
        var lossValue = loss.ToSingle();
        lossValue.Should().BeLessThan(0.01f); // Very low loss for perfect predictions
    }

    [Fact]
    public void ClsLossBuilder_BuildClsLoss()
    {
        // Arrange & Act
        var loss = ClsLossBuilder.BuildLoss("ClsLoss");

        // Assert
        loss.Should().NotBeNull();
        loss.Should().BeOfType<ClsLoss>();
    }

    [Fact]
    public void ClsModelBuilder_BuildFromConfig()
    {
        // Arrange
        var config = new Dictionary<string, object>
        {
            ["Backbone"] = new Dictionary<string, object>
            {
                ["name"] = "MobileNetV3",
                ["scale"] = 0.5f
            },
            ["Head"] = new Dictionary<string, object>
            {
                ["name"] = "ClsHead",
                ["num_classes"] = 4
            }
        };

        // Act
        var model = ClsModelBuilder.BuildFromConfig(config, inChannels: 3);

        // Assert
        model.Should().NotBeNull();
        model.BackboneName.Should().Be("MobileNetV3");
        model.HeadName.Should().Be("ClsHead");
    }

    [Fact]
    public void ClsModelBuilder_BuildFromConfig_ClassDimAlternative()
    {
        // Arrange (using "class_dim" instead of "num_classes")
        var config = new Dictionary<string, object>
        {
            ["Backbone"] = new Dictionary<string, object>
            {
                ["name"] = "MobileNetV3_large"
            },
            ["Head"] = new Dictionary<string, object>
            {
                ["name"] = "ClsHead",
                ["class_dim"] = 2
            }
        };

        // Act
        var model = ClsModelBuilder.BuildFromConfig(config);

        // Assert
        model.Should().NotBeNull();
    }

    [Fact]
    public void ClsMobileNetV3_Small_OutputShape()
    {
        // Arrange
        var backbone = new ClsMobileNetV3(inChannels: 3, modelName: "small", scale: 1.0f);
        using var input = torch.randn(1, 3, 224, 224);

        // Act
        using var output = backbone.forward(input);

        // Assert
        output.shape.Length.Should().Be(4); // [B, C, H, W]
        output.shape[0].Should().Be(1); // Batch size
        output.shape[1].Should().Be(backbone.OutChannels); // Channels
        output.shape[2].Should().BeLessThan(224); // Height reduced
        output.shape[3].Should().BeLessThan(224); // Width reduced
    }

    [Fact]
    public void ClsMobileNetV3_Large_OutputShape()
    {
        // Arrange
        var backbone = new ClsMobileNetV3(inChannels: 3, modelName: "large", scale: 0.5f);
        using var input = torch.randn(2, 3, 128, 128);

        // Act
        using var output = backbone.forward(input);

        // Assert
        output.shape.Length.Should().Be(4); // [B, C, H, W]
        output.shape[0].Should().Be(2); // Batch size
        output.shape[1].Should().Be(backbone.OutChannels);
    }

    [Fact]
    public void ClsHead_OutputShape()
    {
        // Arrange
        const int inChannels = 256;
        const int numClasses = 4;
        var head = new ClsHead(inChannels, numClasses);
        using var input = torch.randn(2, inChannels, 7, 7);

        // Act
        head.eval();
        using var output = head.forward(input);

        // Assert
        output.shape.Should().Equal(new long[] { 2, numClasses });
    }
}
