using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// UniMERNet 解码器：基于 tokenizer 解码 + 标准化。
/// 在 ONNX 推理中使用简化的基于字符集的解码。
/// </summary>
public sealed class UniMerNetDecoder : RecDecoderBase
{
    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count == 0)
        {
            return new RecResult(string.Empty, 0f);
        }

        var (time, classes) = ParseDims(logits, dims, charset.Count);
        if (time * classes != logits.Length || time == 0)
        {
            return new RecResult(string.Empty, 0f);
        }

        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            if (idx == 0)
            {
                break;
            }

            if (idx < charset.Count)
            {
                textChars.Add(charset[idx]);
                scores.Add(prob);
            }
        }

        var text = string.Concat(textChars);
        text = NormalizeOutput(text);
        var score = scores.Count == 0 ? 0f : scores.Average();
        return new RecResult(text, score);
    }

    private static string NormalizeOutput(string text)
    {
        // 基本标准化
        text = text.Trim();
        // 移除多余空白
        while (text.Contains("  "))
        {
            text = text.Replace("  ", " ");
        }

        return text;
    }
}
