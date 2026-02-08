using FluentAssertions;
using PaddleOcr.Training;

namespace PaddleOcr.Tests;

public sealed class RecLabelLineParserTests
{
    [Fact]
    public void TryParse_Should_Support_Tab_Separator()
    {
        var ok = RecLabelLineParser.TryParse("train/word_1.png\tGenaxis Theatre", out var img, out var text);

        ok.Should().BeTrue();
        img.Should().Be("train/word_1.png");
        text.Should().Be("Genaxis Theatre");
    }

    [Fact]
    public void TryParse_Should_Support_Space_Separator()
    {
        var ok = RecLabelLineParser.TryParse("train/word_2.png [06]", out var img, out var text);

        ok.Should().BeTrue();
        img.Should().Be("train/word_2.png");
        text.Should().Be("[06]");
    }

    [Fact]
    public void TryParse_Should_Preserve_Text_With_Inner_Spaces()
    {
        var ok = RecLabelLineParser.TryParse("train/word_3.png    New York City", out var img, out var text);

        ok.Should().BeTrue();
        img.Should().Be("train/word_3.png");
        text.Should().Be("New York City");
    }

    [Fact]
    public void TryParse_Should_Honor_Custom_Delimiter()
    {
        var ok = RecLabelLineParser.TryParse("train/word_4.png,hello world", ",", out var img, out var text);

        ok.Should().BeTrue();
        img.Should().Be("train/word_4.png");
        text.Should().Be("hello world");
    }
}
