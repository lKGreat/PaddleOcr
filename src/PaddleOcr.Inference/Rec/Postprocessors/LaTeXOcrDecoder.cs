using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// LaTeXOCR 解码器：基于 token ID 解码。
/// 在 ONNX 推理场景中，模型直接输出 token IDs，通过字符集映射回文本。
/// 在没有 HuggingFace tokenizer 的情况下，使用简化的基于字符集的解码。
/// </summary>
public sealed class LaTeXOcrDecoder : RecDecoderBase
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

        // argmax 获取每个 time step 的 token
        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            // 通常 0 是 padding/eos
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
        // 后处理：清理 LaTeX 特殊格式
        text = PostProcessLatex(text);
        var score = scores.Count == 0 ? 0f : scores.Average();
        return new RecResult(text, score);
    }

    private static string PostProcessLatex(string text)
    {
        // 移除多余空格
        text = text.Trim();
        // 移除前后的 $ 符号（如果有）
        if (text.StartsWith("$") && text.EndsWith("$"))
        {
            text = text[1..^1].Trim();
        }

        return text;
    }
}
