using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// SAR 解码器：argmax + 在 BOS/EOS 截断 + 可选符号移除。
/// charset 格式: characters + [&lt;UKN&gt;, &lt;BOS/EOS&gt;, &lt;PAD&gt;]
/// </summary>
public sealed class SarLabelDecoder : RecDecoderBase
{
    private readonly bool _removeSymbol;

    public SarLabelDecoder(bool removeSymbol = false)
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
        if (time * classes != logits.Length)
        {
            return new RecResult(string.Empty, 0f);
        }

        // BOS/EOS 是 charset 倒数第二个，PAD 是最后一个
        var boseosIdx = charset.Count - 2;
        var padIdx = charset.Count - 1;

        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            if (idx == boseosIdx)
            {
                break;
            }

            if (idx == padIdx)
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
}
