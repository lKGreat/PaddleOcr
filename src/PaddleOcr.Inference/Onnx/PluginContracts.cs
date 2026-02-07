using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Inference.Onnx;

public interface IInferencePreprocessPlugin
{
    string Name { get; }
    DenseTensor<float> BuildInput(Image<Rgb24> src, IReadOnlyList<int> dims, int defaultH, int defaultW);
}

public interface IDetPostprocessPlugin
{
    string Name { get; }
    List<OcrBox> Postprocess(float[] data, int[] dims, int width, int height, float thresh);
}

public interface IRecPostprocessPlugin
{
    string Name { get; }
    RecResult Postprocess(float[] data, int[] dims, IReadOnlyList<string> charset);
}

public interface IClsPostprocessPlugin
{
    string Name { get; }
    ClsResult Postprocess(float[] logits, IReadOnlyList<string> labels);
}
