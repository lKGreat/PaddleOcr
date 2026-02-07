using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// SRN 解码器：reshape + argmax + 去 SOS/EOS。
/// SRN 输出 preds["predict"] 形状为 [batch, max_text_length * char_num]，
/// 需要 reshape 为 [batch, max_text_length, char_num]。
/// charset 格式: characters + [sos] + [eos]
/// </summary>
public sealed class SrnLabelDecoder : RecDecoderBase
{
    private readonly int _maxTextLength;

    public SrnLabelDecoder(int maxTextLength = 25)
    {
        _maxTextLength = maxTextLength;
    }

    public override RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset)
    {
        if (logits.Length == 0 || charset.Count <= 2)
        {
            return new RecResult(string.Empty, 0f);
        }

        // charset 中 sos 和 eos 在末尾
        var sosIdx = charset.Count - 2;
        var eosIdx = charset.Count - 1;

        // 计算 char_num
        var charNum = logits.Length / _maxTextLength;
        if (charNum <= 0 || charNum * _maxTextLength != logits.Length)
        {
            // 尝试从 dims 推断
            if (dims.Length >= 2)
            {
                charNum = dims[^1];
            }
            else
            {
                return new RecResult(string.Empty, 0f);
            }
        }

        var time = logits.Length / charNum;
        var textChars = new List<string>();
        var scores = new List<float>();

        for (var t = 0; t < time; t++)
        {
            var (idx, prob) = ArgmaxWithProb(logits, t * charNum, charNum);
            if (idx == eosIdx || idx == sosIdx)
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
