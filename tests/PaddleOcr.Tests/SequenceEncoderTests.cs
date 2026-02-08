using FluentAssertions;
using PaddleOcr.Training.Rec.Necks;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Tests;

public sealed class SequenceEncoderTests
{
    [Fact]
    public void Forward_RnnEncoder_Should_Output_Sequence()
    {
        using var encoder = new SequenceEncoder(inChannels: 16, encoderType: "rnn", hiddenSize: 8);
        using var x = rand([2, 16, 1, 20], dtype: ScalarType.Float32);
        using var y = encoder.call(x);

        y.shape.Length.Should().Be(3);
        y.shape[0].Should().Be(2);
        y.shape[1].Should().Be(20);
        y.shape[2].Should().Be(16); // bidirectional hidden_size*2
    }

    [Fact]
    public void Forward_SvtrEncoder_Should_Run_Svtr_Then_Im2Seq()
    {
        using var encoder = new SequenceEncoder(inChannels: 32, encoderType: "svtr", hiddenSize: 16);
        using var x = rand([2, 32, 1, 18], dtype: ScalarType.Float32);
        using var y = encoder.call(x);

        y.shape.Length.Should().Be(3);
        y.shape[0].Should().Be(2);
        y.shape[1].Should().Be(18);
        y.shape[2].Should().Be(32);
    }
}
