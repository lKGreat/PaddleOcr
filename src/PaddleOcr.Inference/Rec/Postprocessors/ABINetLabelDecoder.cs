using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// ABINet 解码器：取 align 输出 + NRTR 解码逻辑。
/// charset 格式: [&lt;/s&gt;] + characters
/// /s=0
/// </summary>
public sealed class ABINetLabelDecoder : RecDecoderBase
{
    private const int EosIdx = 0;

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
