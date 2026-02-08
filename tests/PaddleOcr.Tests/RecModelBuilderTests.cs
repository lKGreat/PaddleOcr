using FluentAssertions;
using PaddleOcr.Training.Rec;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Tests;

public sealed class RecModelBuilderTests
{
    [Fact]
    public void BuildBackbone_Should_Resolve_PPLCNetV3_Alias()
    {
        var (module, outChannels) = RecModelBuilder.BuildBackbone("PPLCNetV3");
        try
        {
            outChannels.Should().BeGreaterThan(0);
            module.GetType().Name.Should().Contain("PPLCNetV3");
        }
        finally
        {
            module.Dispose();
        }
    }

    [Fact]
    public void Build_WithTransform_Should_Run_ForwardDict()
    {
        using var model = RecModelBuilder.Build(
            backboneName: "MobileNetV1Enhance",
            neckName: "SequenceEncoder",
            headName: "CTCHead",
            numClasses: 8,
            inChannels: 3,
            hiddenSize: 48,
            maxLen: 6,
            neckEncoderType: "reshape",
            transformName: "STN_ON");

        model.TransformName.Should().Be("STN_ON");

        using var x = rand([1, 3, 32, 100], dtype: ScalarType.Float32);
        var predictions = model.ForwardDict(x);
        try
        {
            predictions.Should().ContainKey("predict");
        }
        finally
        {
            foreach (var tensor in predictions.Values)
            {
                tensor.Dispose();
            }
        }
    }
}
