using PaddleOcr.Inference.Rec;
using PaddleOcr.Inference.Rec.Postprocessors;
using PaddleOcr.Models;

namespace PaddleOcr.Inference.Onnx;

public static class InferenceComponentRegistry
{
    private static readonly Dictionary<string, Func<float[], int[], int, int, float, List<OcrBox>>> DetPostprocessors =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["db-multibox"] = (data, dims, w, h, thresh) => PostprocessUtils.DetectBoxes(data, dims, w, h, thresh)
        };

    private static readonly Dictionary<string, Func<float[], int[], IReadOnlyList<string>, RecResult>> RecPostprocessors =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["ctc-greedy"] = (data, dims, charset) => PostprocessUtils.DecodeRecCtc(data, dims, charset),
            ["ctc"] = (data, dims, charset) => new CtcLabelDecoder().Decode(data, dims, charset),
            ["attn"] = (data, dims, charset) => new AttnLabelDecoder().Decode(data, dims, charset),
            ["srn"] = (data, dims, charset) => new SrnLabelDecoder().Decode(data, dims, charset),
            ["nrtr"] = (data, dims, charset) => new NrtrLabelDecoder().Decode(data, dims, charset),
            ["sar"] = (data, dims, charset) => new SarLabelDecoder().Decode(data, dims, charset),
            ["vitstr"] = (data, dims, charset) => new ViTStrLabelDecoder().Decode(data, dims, charset),
            ["abinet"] = (data, dims, charset) => new ABINetLabelDecoder().Decode(data, dims, charset),
            ["spin"] = (data, dims, charset) => new SpinLabelDecoder().Decode(data, dims, charset),
            ["can"] = (data, dims, charset) => new CanLabelDecoder().Decode(data, dims, charset),
            ["latexocr"] = (data, dims, charset) => new LaTeXOcrDecoder().Decode(data, dims, charset),
            ["parseq"] = (data, dims, charset) => new ParseQLabelDecoder().Decode(data, dims, charset),
            ["cppd"] = (data, dims, charset) => new CppdLabelDecoder().Decode(data, dims, charset),
            ["pren"] = (data, dims, charset) => new PrenLabelDecoder().Decode(data, dims, charset),
            ["unimernet"] = (data, dims, charset) => new UniMerNetDecoder().Decode(data, dims, charset),
            ["visionlan"] = (data, dims, charset) => new VisionLanDecoder().Decode(data, dims, charset),
            ["rfl"] = (data, dims, charset) => new RflLabelDecoder().Decode(data, dims, charset),
            ["satrn"] = (data, dims, charset) => new SatrnLabelDecoder().Decode(data, dims, charset)
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

    /// <summary>
    /// 根据 RecAlgorithm 获取对应的后处理器。
    /// </summary>
    public static Func<float[], int[], IReadOnlyList<string>, RecResult> GetRecPostprocessor(RecAlgorithm algorithm)
    {
        var name = algorithm.GetDefaultPostprocessor();
        return GetRecPostprocessor(name);
    }

    /// <summary>
    /// 获取强类型的 IRecPostprocessor 实例。
    /// </summary>
    public static IRecPostprocessor GetRecPostprocessorInstance(string name = "ctc")
    {
        return RecPostprocessorFactory.Create(name);
    }

    /// <summary>
    /// 获取强类型的 IRecPostprocessor 实例。
    /// </summary>
    public static IRecPostprocessor GetRecPostprocessorInstance(RecAlgorithm algorithm)
    {
        return RecPostprocessorFactory.Create(algorithm);
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
