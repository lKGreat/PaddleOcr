using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// PREN 解码器：在 EOS(1) 截断 + 忽略 PAD(0)/UNK(2)。
/// charset 格式: [&lt;PAD&gt;, &lt;EOS&gt;, &lt;UNK&gt;] + characters
/// PAD=0, EOS=1, UNK=2
/// </summary>
public sealed class PrenLabelDecoder : RecDecoderBase
{
    private const int PadIdx = 0;
    private const int EosIdx = 1;
    private const int UnkIdx = 2;

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

            // 跳过 PAD 和 UNK
            if (idx == PadIdx || idx == UnkIdx)
            {
                continue;
            }

            if (idx < charset.Count)
            {
                textChars.Add(charset[idx]);
                scores.Add(prob);
            }
        }

        if (textChars.Count == 0)
        {
            return new RecResult(string.Empty, 0f);
        }

        return BuildResult(textChars, scores);
    }
}
