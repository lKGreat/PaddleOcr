using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// NRTR 解码器：跳过首 token + 在 &lt;/s&gt; 截断。
/// charset 格式: [blank, &lt;unk&gt;, &lt;s&gt;, &lt;/s&gt;] + characters
/// blank=0, unk=1, s=2, /s=3
/// </summary>
public sealed class NrtrLabelDecoder : RecDecoderBase
{
    // 特殊 token 索引
    private const int BlankIdx = 0;
    private const int UnkIdx = 1;
    private const int SosIdx = 2;
    private const int EosIdx = 3;

    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 4)
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

        // 跳过第一个 token（通常是 <s>）
        for (var t = 1; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            // 在 </s> 处截断
            if (idx == EosIdx)
            {
                break;
            }

            // 跳过 blank, unk, sos
            if (idx == BlankIdx || idx == UnkIdx || idx == SosIdx)
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
