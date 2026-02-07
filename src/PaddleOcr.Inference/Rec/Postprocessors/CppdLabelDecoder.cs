using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// CPPD 解码器：继承 NRTR 逻辑 + 处理 tuple/dict 输入。
/// charset 格式: [&lt;/s&gt;] + characters
/// /s=0
/// </summary>
public sealed class CppdLabelDecoder : RecDecoderBase
{
    private const int EosIdx = 0;

    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 1)
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
            if (idx == EosIdx)
            {
                break;
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
