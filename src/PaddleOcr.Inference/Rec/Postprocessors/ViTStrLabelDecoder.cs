using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// ViTSTR 解码器：继承 NRTR 逻辑 + 跳过首列。
/// charset 格式: [&lt;s&gt;, &lt;/s&gt;] + characters
/// s=0, /s=1
/// </summary>
public sealed class ViTStrLabelDecoder : RecDecoderBase
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

        // 跳过首列 (preds[:, 1:])
        for (var t = 1; t < time; t++)
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
