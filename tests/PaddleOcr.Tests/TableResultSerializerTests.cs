using FluentAssertions;
using PaddleOcr.Inference.Onnx;

namespace PaddleOcr.Tests;

public sealed class TableResultSerializerTests
{
    [Fact]
    public void BuildLine_Should_Be_Deterministic_For_Same_Input()
    {
        var tensors = new List<TensorOutput>
        {
            new([1, 2, 3, 4], [1, 1, 2, 2]),
            new([9, 8, 7], [1, 3])
        };
        var ocr = new List<OcrItem>
        {
            new("A1", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.99f),
            new("B2", [[12, 0], [20, 0], [20, 10], [12, 10]], 0.88f)
        };

        var a = TableResultSerializer.BuildLine("demo.png", tensors, ocr);
        var b = TableResultSerializer.BuildLine("demo.png", tensors, ocr);

        a.Should().Be(b);
        a.Should().StartWith("demo.png\t");
        a.Should().Contain("\"table_tensors\"");
        a.Should().Contain("\"ocr\"");
    }
}
