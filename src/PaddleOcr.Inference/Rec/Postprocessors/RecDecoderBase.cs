using PaddleOcr.Models;

namespace PaddleOcr.Inference.Rec.Postprocessors;

/// <summary>
/// Rec 后处理解码器基类，提供公共工具方法。
/// </summary>
public abstract class RecDecoderBase : IRecPostprocessor
{
    public abstract RecResult Decode(float[] logits, int[] dims, IReadOnlyList<string> charset);

    /// <summary>
    /// 对一维 slice 执行 softmax。
    /// </summary>
    protected static float[] Softmax(float[] x)
    {
        if (x.Length == 0)
        {
            return x;
        }

        var max = x[0];
        for (var i = 1; i < x.Length; i++)
        {
            if (x[i] > max) max = x[i];
        }

        var exps = new float[x.Length];
        var sum = 0f;
        for (var i = 0; i < x.Length; i++)
        {
            exps[i] = MathF.Exp(x[i] - max);
            sum += exps[i];
        }

        if (sum <= 0f)
        {
            var uniform = 1f / x.Length;
            for (var i = 0; i < exps.Length; i++) exps[i] = uniform;
            return exps;
        }

        for (var i = 0; i < exps.Length; i++)
        {
            exps[i] /= sum;
        }

        return exps;
    }

    /// <summary>
    /// 判断字符串是否仅由字母、数字或 CJK 字符组成。
    /// </summary>
    protected static bool IsAlphanumericOrChinese(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return false;
        }

        foreach (var c in s)
        {
            if (!char.IsLetterOrDigit(c) && !(c >= '\u4e00' && c <= '\u9fff'))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// 在 logits 的指定 time step 上做 argmax，返回 (index, maxProb)。
    /// </summary>
    protected static (int Index, float Prob) ArgmaxWithProb(float[] logits, int offset, int classes)
    {
        var slice = new float[classes];
        Array.Copy(logits, offset, slice, 0, classes);
        var probs = Softmax(slice);
        var best = 0;
        for (var i = 1; i < probs.Length; i++)
        {
            if (probs[i] > probs[best]) best = i;
        }

        return (best, probs[best]);
    }

    /// <summary>
    /// 在 logits 的指定 time step 上做 argmax（无 softmax），返回 (index, rawValue)。
    /// </summary>
    protected static (int Index, float Value) ArgmaxRaw(float[] logits, int offset, int classes)
    {
        var best = 0;
        var bestVal = logits[offset];
        for (var i = 1; i < classes; i++)
        {
            if (logits[offset + i] > bestVal)
            {
                best = i;
                bestVal = logits[offset + i];
            }
        }

        return (best, bestVal);
    }

    /// <summary>
    /// 解析维度信息，获取 (timeSteps, numClasses)。
    /// </summary>
    protected static (int TimeSteps, int Classes) ParseDims(float[] logits, int[] dims, int charsetCount)
    {
        int classes;
        if (dims.Length >= 2)
        {
            classes = dims[^1];
        }
        else
        {
            classes = charsetCount;
        }

        if (classes <= 0 || classes > logits.Length)
        {
            classes = charsetCount;
        }

        var time = logits.Length / Math.Max(1, classes);
        return (time, classes);
    }

    /// <summary>
    /// 构建 RecResult，文本和平均置信度。
    /// </summary>
    protected static RecResult BuildResult(List<string> textChars, List<float> scores)
    {
        var text = string.Concat(textChars);
        var score = scores.Count == 0 ? 0f : scores.Average();
        return new RecResult(text, score);
    }
}
