namespace PaddleOcr.Inference.Onnx;

internal static class InferenceComponentRegistry
{
    private static readonly Dictionary<string, Func<float[], int[], int, int, float, List<OcrBox>>> DetPostprocessors =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["db-multibox"] = (data, dims, w, h, thresh) => PostprocessUtils.DetectBoxes(data, dims, w, h, thresh)
        };

    private static readonly Dictionary<string, Func<float[], int[], IReadOnlyList<string>, RecResult>> RecPostprocessors =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["ctc-greedy"] = (data, dims, charset) => PostprocessUtils.DecodeRecCtc(data, dims, charset)
        };

    public static Func<float[], int[], int, int, float, List<OcrBox>> GetDetPostprocessor(string name = "db-multibox")
    {
        return DetPostprocessors.TryGetValue(name, out var fn) ? fn : DetPostprocessors["db-multibox"];
    }

    public static Func<float[], int[], IReadOnlyList<string>, RecResult> GetRecPostprocessor(string name = "ctc-greedy")
    {
        return RecPostprocessors.TryGetValue(name, out var fn) ? fn : RecPostprocessors["ctc-greedy"];
    }

    public static void RegisterDetPostprocessor(string name, Func<float[], int[], int, int, float, List<OcrBox>> fn)
    {
        DetPostprocessors[name] = fn;
    }

    public static void RegisterRecPostprocessor(string name, Func<float[], int[], IReadOnlyList<string>, RecResult> fn)
    {
        RecPostprocessors[name] = fn;
    }
}

