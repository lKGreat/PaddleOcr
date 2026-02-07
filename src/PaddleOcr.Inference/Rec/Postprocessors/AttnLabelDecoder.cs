using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// Attention 解码器：argmax + 在 EOS 处截断。
/// 适用于 RARE 等 attention 系列算法。
/// charset 格式: [sos] + characters + [eos]
/// </summary>
public sealed class AttnLabelDecoder : RecDecoderBase
{
    // 在 AttnLabelDecode 中，特殊字符为 [sos] 在头、[eos] 在尾
    // charset 的构造: add_special_char => ["sos"] + dict_character + ["eos"]
    // 因此 sos = 0, eos = charset.Count - 1
    // 在 __call__ 中，ignored_tokens = get_ignored_tokens() = [sos_idx=0, eos_idx=last]

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

        var eosIdx = charset.Count - 1;
        var sosIdx = 0;

        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * classes, classes);
            if (idx == eosIdx)
            {
                break;
            }

            if (idx == sosIdx)
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
