using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// SATRN 解码器：同 SAR 逻辑。
/// charset 格式: [&lt;UKN&gt;, &lt;BOS/EOS&gt;, &lt;PAD&gt;] + characters
/// UKN=0, BOS/EOS=1, PAD=2
/// </summary>
public sealed class SatrnLabelDecoder : RecDecoderBase
{
    private readonly bool _removeSymbol;

    public SatrnLabelDecoder(bool removeSymbol = false)
    {
        _removeSymbol = removeSymbol;
    }

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

        // SATRN 特殊 token: UKN=0, BOS/EOS=1, PAD=2 (在 charset 头部)
        var boseosIdx = 1;
        var padIdx = 2;

        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            if (idx == boseosIdx)
            {
                break;
            }

            if (idx == padIdx || idx == 0)
            {
                continue;
            }

            if (idx < charset.Count)
            {
                var ch = charset[idx];
                if (_removeSymbol && !IsAlphanumericOrChinese(ch))
                {
                    continue;
                }

                textChars.Add(ch);
                scores.Add(prob);
            }
        }

        return BuildResult(textChars, scores);
    }

    private static bool IsAlphanumericOrChinese(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return false;
        }

        foreach (var c in s)
        {
            if (char.IsLetterOrDigit(c) || (c >= 0x4E00 && c <= 0x9FFF))
            {
                return true;
            }
        }

        return false;
    }
}
