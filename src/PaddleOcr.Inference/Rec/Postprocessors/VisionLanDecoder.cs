using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// VisionLAN 解码器：长度预测 + 拼接 + 索引偏移。
/// 与标准 CTC 不同，VisionLAN 的字符索引需要减 1。
/// </summary>
public sealed class VisionLanDecoder : RecDecoderBase
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
            var probs = new float[classes];
            Array.Copy(logits, t * classes, probs, 0, classes);
            var softmaxed = Softmax(probs);

            var best = 0;
            for (var i = 1; i < classes; i++)
            {
                if (softmaxed[i] > softmaxed[best]) best = i;
            }

            // VisionLAN 中 token 0 表示 EOS
            if (best == 0)
            {
                break;
            }

            // 索引偏移: token - 1 映射到 charset
            var charIdx = best - 1;
            if (charIdx >= 0 && charIdx < charset.Count)
            {
                textChars.Add(charset[charIdx]);
                scores.Add(softmaxed[best]);
            }
        }

        return BuildResult(textChars, scores);
    }
}
