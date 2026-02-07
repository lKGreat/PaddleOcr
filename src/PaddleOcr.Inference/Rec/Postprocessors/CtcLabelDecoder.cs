using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// CTC 解码器：argmax + 去重复 + 去blank(index 0)。
/// 适用于 CRNN, SVTR, SVTR_LCNet, SVTR_HGNet, STARNet 等 CTC 系列算法。
/// </summary>
public sealed class CtcLabelDecoder : RecDecoderBase
{
    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 1)
        {
            return new RecResult(string.Empty, 0f);
        }

        var (time, classes) = ParseDims(logits, dims, charset.Count);
        if (time * classes != logits.Length)
        {
            return new RecResult(string.Empty, 0f);
        }

        var tokens = new int[time];
        var scores = new float[time];
        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            tokens[t] = idx;
            scores[t] = prob;
        }

        var textChars = new List<string>();
        var keptScores = new List<float>();
        var prev = -1;
        for (var i = 0; i < time; i++)
        {
            var token = tokens[i];
            // blank token = 0, 跳过重复
            if (token == 0 || token == prev)
            {
                prev = token;
                continue;
            }

            if (token < charset.Count)
            {
                textChars.Add(charset[token]);
                keptScores.Add(scores[i]);
            }

            prev = token;
        }

        return BuildResult(textChars, keptScores);
    }
}
