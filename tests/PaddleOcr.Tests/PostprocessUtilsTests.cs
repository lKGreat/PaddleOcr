using FluentAssertions;
using PaddleOcr.Inference.Onnx;

namespace PaddleOcr.Tests;

public sealed class PostprocessUtilsTests
{
    [Fact]
    public void DecodeCls_Should_Return_Label_And_Score()
    {
        var res = PostprocessUtils.DecodeCls([0.1f, 2.3f], ["0", "180"]);
        res.Label.Should().Be("180");
        res.Score.Should().BeGreaterThan(0.5f);
    }

    [Fact]
    public void DecodeRecCtc_Should_Remove_Blank_And_Repeat()
    {
        var charset = new List<string> { "", "a", "b", "c" };

        // time=4, classes=4
        // t0:a, t1:a(repeat), t2:blank, t3:b
        var logits = new float[]
        {
            0f, 5f, 0f, 0f,
            0f, 4f, 0f, 0f,
            5f, 0f, 0f, 0f,
            0f, 0f, 5f, 0f
        };

        var res = PostprocessUtils.DecodeRecCtc(logits, [1, 4, 4], charset);
        res.Text.Should().Be("ab");
        res.Score.Should().BeGreaterThan(0.8f);
    }

    [Fact]
    public void DetectBoxes_Should_Map_To_Image_Coordinates()
    {
        // 4x4 map with center area activated
        var map = new float[]
        {
            0,0,0,0,
            0,1,1,0,
            0,1,1,0,
            0,0,0,0
        };

        var boxes = PostprocessUtils.DetectBoxes(map, [1, 1, 4, 4], 400, 200, 0.5f);
        boxes.Should().HaveCount(1);
        boxes[0].Points.Should().HaveCount(4);
    }

    [Fact]
    public void DetectBoxes_Should_Be_Deterministic_For_Same_Input()
    {
        var map = new float[]
        {
            0,0,0,0,
            0,1,1,0,
            0,1,1,0,
            0,0,0,0
        };

        var a = PostprocessUtils.DetectBoxes(map, [1, 1, 4, 4], 400, 200, 0.5f);
        var b = PostprocessUtils.DetectBoxes(map, [1, 1, 4, 4], 400, 200, 0.5f);
        a.Should().BeEquivalentTo(b);
    }

    [Fact]
    public void SortBoxes_Should_Order_By_TopThenLeft()
    {
        var boxes = new List<OcrBox>
        {
            new([[100, 100],[150,100],[150,130],[100,130]]),
            new([[20, 95],[70,95],[70,120],[20,120]]),
            new([[10, 200],[60,200],[60,220],[10,220]])
        };

        var sorted = PostprocessUtils.SortBoxes(boxes);
        sorted[0].Points[0][0].Should().Be(20);
        sorted[1].Points[0][0].Should().Be(100);
        sorted[2].Points[0][1].Should().Be(200);
    }
}
