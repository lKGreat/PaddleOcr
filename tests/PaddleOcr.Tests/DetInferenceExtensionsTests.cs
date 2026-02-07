using System.Text.Json;
using FluentAssertions;
using PaddleOcr.Inference.Onnx;

namespace PaddleOcr.Tests;

public sealed class DetInferenceExtensionsTests
{
    [Fact]
    public void DecodeBoxes_Should_Work_For_All_Det_Algorithms()
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

        foreach (var algorithm in new[] { "DB", "DB++", "EAST", "SAST", "PSE", "FCE", "CT" })
        {
            var options = NewOptions(algorithm);
            var boxes = DetInferenceExtensions.DecodeBoxes(options, outputs, 400, 200);
            boxes.Should().NotBeEmpty($"algorithm={algorithm}");
        }
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
            ["img0.png"] = new DetRuntimeProfile(1.2, 2.3, 3.4, 6.9)
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
        var runtimeProfile = doc.RootElement.GetProperty("algorithm_runtime_profile");
        runtimeProfile.GetProperty("image_count").GetInt32().Should().Be(1);
        runtimeProfile.GetProperty("avg_total_ms").GetDouble().Should().BeApproximately(6.9d, 0.0001d);
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
