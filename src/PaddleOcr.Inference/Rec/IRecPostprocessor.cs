using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec;

/// <summary>
/// Rec 后处理解码器接口。
/// 将 ONNX 模型输出的 logits 解码为文本结果。
/// </summary>
public interface IRecPostprocessor
{
    /// <summary>
    /// 解码 logits 为文本识别结果。
    /// </summary>
    /// <param name="logits">模型输出的 logits 数组</param>
    /// <param name="dims">logits 的维度信息</param>
    /// <param name="charset">字符集</param>
    /// <returns>识别结果</returns>
    RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset);
}
