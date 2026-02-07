using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// SPIN 解码器：继承 Attn 逻辑，特殊字符序不同。
/// charset 格式: [sos] + [eos] + characters
/// sos=0, eos=1
/// </summary>
public sealed class SpinLabelDecoder : RecDecoderBase
{
    private const int SosIdx = 0;
    private const int EosIdx = 1;

    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 2)
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

            if (idx == SosIdx)
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
