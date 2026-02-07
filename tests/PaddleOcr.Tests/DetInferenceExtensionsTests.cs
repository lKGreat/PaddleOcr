using System.Text.Json;
using FluentAssertions;
using PaddleOcr.Inference.Onnx;

namespace PaddleOcr.Tests;

public sealed class DetInferenceExtensionsTests
{
    [Theory]
    [InlineData("DB", 74, 36, 226, 114)]
    [InlineData("DB++", 74, 36, 226, 114)]
    [InlineData("EAST", 99, 49, 201, 101)]
    [InlineData("SAST", 99, 49, 201, 101)]
    [InlineData("PSE", 99, 49, 201, 101)]
    [InlineData("FCE", 74, 36, 226, 114)]
    [InlineData("CT", 99, 49, 201, 101)]
    public void DecodeBoxes_Should_Match_Golden_For_Each_Algorithm(string algorithm, int x1, int y1, int x2, int y2)
    {
        var map = new float[]
        {
            0,0,0,0,
            0,1,1,0,
            0,1,1,0,
            0,0,0,0
        };
        var outputs = new List<TensorOutput>
        {
            new(map, [1, 1, 4, 4]),
            new(map, [1, 1, 4, 4])
        };

        var options = NewOptions(algorithm);
        var boxes = DetInferenceExtensions.DecodeBoxes(options, outputs, 400, 200);
        boxes.Should().HaveCount(1, $"algorithm={algorithm}");
        var rect = ToRect(boxes[0]);
        rect.Should().Be((x1, y1, x2, y2), $"algorithm={algorithm}");
    }

    [Fact]
    public void ResolveDetInputSize_Should_Use_Model_Static_Dims()
    {
        var options = NewOptions("DB");
        var size = DetInferenceExtensions.ResolveDetInputSize(options, 2000, 1000, [1, 3, 960, 960]);
        size.Should().Be((960, 960));
    }

    [Fact]
    public void ResolveDetInputSize_Should_Respect_Max_LimitType()
    {
        var options = NewOptions("DB") with
        {
            DetLimitType = "max",
            DetLimitSideLen = 640
        };
        var size = DetInferenceExtensions.ResolveDetInputSize(options, 2000, 1000, [1, 3, -1, -1]);
        size.Should().Be((640, 320));
    }

    [Fact]
    public void ResolveDetInputSize_Should_Respect_Min_LimitType()
    {
        var options = NewOptions("DB") with
        {
            DetLimitType = "min",
            DetLimitSideLen = 736
        };
        var size = DetInferenceExtensions.ResolveDetInputSize(options, 400, 100, [1, 3, -1, -1]);
        size.Should().Be((2944, 736));
    }

    [Fact]
    public void WriteDetMetrics_Should_Include_Quality_When_Gt_Provided()
    {
        var dir = Path.Combine(Path.GetTempPath(), "pocr_det_metrics_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var label = Path.Combine(dir, "gt.txt");
        var metricsPath = Path.Combine(dir, "det_metrics.json");
        File.WriteAllText(label, "img0.png\t[{\"transcription\":\"\",\"points\":[[10,10],[30,10],[30,30],[10,30]]}]\n");

        var predictions = new Dictionary<string, List<OcrBox>>(StringComparer.OrdinalIgnoreCase)
        {
            ["img0.png"] =
            [
                new OcrBox(
                [
                    [10, 10],
                    [30, 10],
                    [30, 30],
                    [10, 30]
                ])
            ]
        };
        var runtime = new Dictionary<string, DetRuntimeProfile>(StringComparer.OrdinalIgnoreCase)
        {
            ["img0.png"] = new DetRuntimeProfile(1.2, 2.3, 3.4, 6.9, 200, 100, 192, 96)
        };

        var options = NewOptions("DB") with
        {
            DetGtLabelPath = label,
            DetEvalIouThresh = 0.5f,
            DetMetricsPath = metricsPath
        };

        DetInferenceExtensions.WriteDetMetrics(options, dir, predictions, runtime);

        File.Exists(metricsPath).Should().BeTrue();
        using var doc = JsonDocument.Parse(File.ReadAllText(metricsPath));
        doc.RootElement.GetProperty("schema_version").GetString().Should().Be("1.1");
        var quality = doc.RootElement.GetProperty("quality");
        quality.ValueKind.Should().Be(JsonValueKind.Object);
        quality.GetProperty("hmean").GetSingle().Should().BeGreaterThan(0.9f);
        var perImage = doc.RootElement.GetProperty("per_image");
        perImage.GetArrayLength().Should().Be(1);
        perImage[0].GetProperty("image").GetString().Should().Be("img0.png");
        perImage[0].GetProperty("hmean").GetSingle().Should().BeGreaterThan(0.9f);
        perImage[0].GetProperty("total_ms").GetDouble().Should().BeApproximately(6.9d, 0.0001d);
        perImage[0].GetProperty("input_width").GetInt32().Should().Be(192);
        perImage[0].GetProperty("input_height").GetInt32().Should().Be(96);
        var runtimeProfile = doc.RootElement.GetProperty("algorithm_runtime_profile");
        runtimeProfile.GetProperty("image_count").GetInt32().Should().Be(1);
        runtimeProfile.GetProperty("avg_total_ms").GetDouble().Should().BeApproximately(6.9d, 0.0001d);
        var runtimePerImage = doc.RootElement.GetProperty("runtime_per_image");
        runtimePerImage.GetArrayLength().Should().Be(1);
        runtimePerImage[0].GetProperty("image").GetString().Should().Be("img0.png");
        runtimePerImage[0].GetProperty("total_ms").GetDouble().Should().BeApproximately(6.9d, 0.0001d);
    }

    private static (int X1, int Y1, int X2, int Y2) ToRect(OcrBox box)
    {
        var xs = box.Points.Select(p => p[0]).ToArray();
        var ys = box.Points.Select(p => p[1]).ToArray();
        return (xs.Min(), ys.Min(), xs.Max(), ys.Max());
    }

    private static DetOnnxOptions NewOptions(string algorithm)
    {
        return new DetOnnxOptions(
            ImageDir: ".",
            DetModelPath: "det.onnx",
            OutputDir: ".",
            DetAlgorithm: algorithm,
            DetThresh: 0.3f,
            DetBoxThresh: 0.6f,
            DetUnclipRatio: 1.5f,
            UseDilation: false,
            BoxType: "quad",
            DetLimitSideLen: 640,
            DetLimitType: "max",
            SaveResPath: null,
            DetEastScoreThresh: 0.8f,
            DetEastCoverThresh: 0.1f,
            DetEastNmsThresh: 0.2f,
            DetSastScoreThresh: 0.5f,
            DetSastNmsThresh: 0.2f,
            DetPseThresh: 0.5f,
            DetPseBoxThresh: 0.1f,
            DetPseMinArea: 16f,
            DetPseScale: 1f,
            FceScales: [8, 16, 32],
            FceAlpha: 1f,
            FceBeta: 1f,
            FceFourierDegree: 5,
            DetGtLabelPath: null,
            DetEvalIouThresh: 0.5f,
            DetMetricsPath: null);
    }
}
