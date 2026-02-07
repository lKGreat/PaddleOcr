using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// ParseQ 解码器：reshape + 在 EOS 截断。
/// charset 格式: [EOS] + characters + [BOS] + [PAD]
/// EOS=0, BOS=charset.Count-2, PAD=charset.Count-1
/// </summary>
public sealed class ParseQLabelDecoder : RecDecoderBase
{
    private const int EosIdx = 0;

    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 3)
        {
            return new RecResult(string.Empty, 0f);
        }

        var (time, classes) = ParseDims(logits, dims, charset.Count);
        if (time * classes != logits.Length || time == 0)
        {
            return new RecResult(string.Empty, 0f);
        }

        var bosIdx = charset.Count - 2;
        var padIdx = charset.Count - 1;

        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            // 在 EOS 处截断
            if (idx == EosIdx)
            {
                break;
            }

            // 跳过 BOS 和 PAD
            if (idx == bosIdx || idx == padIdx)
            {
                continue;
            }

            if (idx < charset.Count)
            {
                textChars.Add(charset[idx]);
                scores.Add(prob);
            }
        }

        return BuildResult(textChars, scores);
    }
}
