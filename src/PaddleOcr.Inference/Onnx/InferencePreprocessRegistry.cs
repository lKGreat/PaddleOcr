using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PaddleOcr.Inference.Onnx;

public static class InferencePreprocessRegistry
{
    private static readonly Dictionary<string, Func<Image<Rgb24>, IReadOnlyList<int>, int, int, DenseTensor<float>>> InputBuilders =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["rgb-chw-01"] = BuildRgbChw01,
            ["det-db-chw-01"] = BuildDetDbChw01,
            ["det-dbpp-chw-01"] = BuildDetDbppChw01
        };

    public static Func<Image<Rgb24>, IReadOnlyList<int>, int, int, DenseTensor<float>> GetInputBuilder(string name = "rgb-chw-01")
    {
        return InputBuilders.TryGetValue(name, out var fn) ? fn : InputBuilders["rgb-chw-01"];
    }

    public static void RegisterInputBuilder(string name, Func<Image<Rgb24>, IReadOnlyList<int>, int, int, DenseTensor<float>> fn)
    {
        InputBuilders[name] = fn;
    }

    private static DenseTensor<float> BuildRgbChw01(Image<Rgb24> src, IReadOnlyList<int> dims, int defaultH, int defaultW)
    {
        return BuildRgbChwNormalized(
            src,
            dims,
            defaultH,
            defaultW,
            [0.485f, 0.456f, 0.406f],
            [0.229f, 0.224f, 0.225f],
            1f / 255f);
    }

    private static DenseTensor<float> BuildDetDbChw01(Image<Rgb24> src, IReadOnlyList<int> dims, int defaultH, int defaultW)
    {
        return BuildRgbChwNormalized(
            src,
            dims,
            defaultH,
            defaultW,
            [0.485f, 0.456f, 0.406f],
            [0.229f, 0.224f, 0.225f],
            1f / 255f);
    }

    private static DenseTensor<float> BuildDetDbppChw01(Image<Rgb24> src, IReadOnlyList<int> dims, int defaultH, int defaultW)
    {
        return BuildRgbChwNormalized(
            src,
            dims,
            defaultH,
            defaultW,
            [0.4810938f, 0.45752457f, 0.40787053f],
            [1f, 1f, 1f],
            1f / 255f);
    }

    private static DenseTensor<float> BuildRgbChwNormalized(
        Image<Rgb24> src,
        IReadOnlyList<int> dims,
        int defaultH,
        int defaultW,
        IReadOnlyList<float> mean,
        IReadOnlyList<float> std,
        float scale)
    {
        var n = dims.Count > 0 && dims[0] > 0 ? dims[0] : 1;
        var c = dims.Count > 1 && dims[1] > 0 ? dims[1] : 3;
        var h = dims.Count > 2 && dims[2] > 0 ? dims[2] : defaultH;
        var w = dims.Count > 3 && dims[3] > 0 ? dims[3] : defaultW;

        using var img = src.Clone();
        img.Mutate(x => x.Resize(w, h));

        var tensor = new DenseTensor<float>([n, c, h, w]);
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var p = img[x, y];
                tensor[0, 0, y, x] = (p.R * scale - mean[0]) / std[0];
                tensor[0, 1, y, x] = (p.G * scale - mean[1]) / std[1];
                tensor[0, 2, y, x] = (p.B * scale - mean[2]) / std[2];
            }
        }

        return tensor;
    }
}
