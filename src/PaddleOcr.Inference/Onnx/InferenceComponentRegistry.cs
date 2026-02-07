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

    private static readonly Dictionary<string, Func<float[], IReadOnlyList<string>, ClsResult>> ClsPostprocessors =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["argmax-softmax"] = (logits, labels) => PostprocessUtils.DecodeCls(logits, labels)
        };

    public static Func<float[], int[], int, int, float, List<OcrBox>> GetDetPostprocessor(string name = "db-multibox")
    {
        return DetPostprocessors.TryGetValue(name, out var fn) ? fn : DetPostprocessors["db-multibox"];
    }

    public static Func<float[], int[], IReadOnlyList<string>, RecResult> GetRecPostprocessor(string name = "ctc-greedy")
    {
        return RecPostprocessors.TryGetValue(name, out var fn) ? fn : RecPostprocessors["ctc-greedy"];
    }

    public static Func<float[], IReadOnlyList<string>, ClsResult> GetClsPostprocessor(string name = "argmax-softmax")
    {
        return ClsPostprocessors.TryGetValue(name, out var fn) ? fn : ClsPostprocessors["argmax-softmax"];
    }

    public static void RegisterDetPostprocessor(string name, Func<float[], int[], int, int, float, List<OcrBox>> fn)
    {
        DetPostprocessors[name] = fn;
    }

    public static void RegisterRecPostprocessor(string name, Func<float[], int[], IReadOnlyList<string>, RecResult> fn)
    {
        RecPostprocessors[name] = fn;
    }

    public static void RegisterClsPostprocessor(string name, Func<float[], IReadOnlyList<string>, ClsResult> fn)
    {
        ClsPostprocessors[name] = fn;
    }
}
