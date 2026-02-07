using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// CAN 解码器：取 pred_prob + argmin 找结束位置 + 空格拼接符号。
/// 用于 LaTeX 数学公式识别。
/// </summary>
public sealed class CanLabelDecoder : RecDecoderBase
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

        // argmax 获取每个 time step 的 token 和概率
        var tokens = new int[time];
        var probs = new float[time];
        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            tokens[t] = idx;
            probs[t] = prob;
        }

        // 找到序列结束位置（首个 0 token 或序列末尾）
        var endPos = time;
        for (var t = 0; t < time; t++)
        {
            if (tokens[t] == 0)
            {
                endPos = t;
                break;
            }
        }

        // 拼接符号，用空格分隔
        var symbols = new List<string>();
        var scores = new List<float>();
        for (var t = 0; t < endPos; t++)
        {
            if (tokens[t] > 0 && tokens[t] < charset.Count)
            {
                symbols.Add(charset[tokens[t]]);
                scores.Add(probs[t]);
            }
        }

        var text = string.Join(" ", symbols);
        var score = scores.Count == 0 ? 0f : scores.Average();
        return new RecResult(text, score);
    }
}
