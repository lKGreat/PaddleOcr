using System.Text;
using PaddleOcr.Models;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// RecMetrics：Rec 模型评估指标，包括 BLEU score、Exp rate、错误分析、混淆矩阵等。
/// </summary>
public static class RecMetrics
{
    /// <summary>
    /// 计算 BLEU score（用于 LaTeXOCR, UniMERNet, PP-FormulaNet）。
    /// </summary>
    public static float CalculateBleu(string reference, string prediction, int maxN = 4)
    {
        if (string.IsNullOrEmpty(reference) && string.IsNullOrEmpty(prediction))
        {
            return 1.0f;
        }

        if (string.IsNullOrEmpty(reference) || string.IsNullOrEmpty(prediction))
        {
            return 0.0f;
        }

        var refTokens = Tokenize(reference);
        var predTokens = Tokenize(prediction);

        if (refTokens.Count == 0 || predTokens.Count == 0)
        {
            return 0.0f;
        }

        // 计算各阶 n-gram 的精确度
        var precisions = new List<float>();
        for (var n = 1; n <= maxN; n++)
        {
            var refNgrams = GetNgrams(refTokens, n);
            var predNgrams = GetNgrams(predTokens, n);

            if (predNgrams.Count == 0)
            {
                precisions.Add(0.0f);
                continue;
            }

            var matches = 0;
            var refCounts = CountNgrams(refNgrams);
            var predCounts = CountNgrams(predNgrams);

            foreach (var (ngram, count) in predCounts)
            {
                if (refCounts.TryGetValue(ngram, out var refCount))
                {
                    matches += Math.Min(count, refCount);
                }
            }

            precisions.Add(matches / (float)predNgrams.Count);
        }

        // 计算几何平均
        var bleu = 1.0f;
        foreach (var p in precisions)
        {
            if (p > 0)
            {
                bleu *= MathF.Pow(p, 1.0f / maxN);
            }
            else
            {
                return 0.0f;
            }
        }

        // 长度惩罚
        var bp = predTokens.Count > refTokens.Count ? 1.0f : MathF.Exp(1.0f - refTokens.Count / (float)predTokens.Count);
        return bp * bleu;
    }

    /// <summary>
    /// 计算 Exp rate / Express match rate（公式模型）。
    /// </summary>
    public static float CalculateExpRate(string reference, string prediction)
    {
        // Exp rate 是精确匹配率
        return string.Equals(reference, prediction, StringComparison.Ordinal) ? 1.0f : 0.0f;
    }

    /// <summary>
    /// 计算错误分析（替换/插入/删除错误）。
    /// </summary>
    public static ErrorAnalysis AnalyzeErrors(string reference, string prediction)
    {
        var (substitutions, insertions, deletions) = LevenshteinDetails(reference, prediction);
        var totalChars = Math.Max(reference.Length, prediction.Length);
        var errorRate = totalChars == 0 ? 0.0f : (substitutions + insertions + deletions) / (float)totalChars;

        return new ErrorAnalysis(
            Substitutions: substitutions,
            Insertions: insertions,
            Deletions: deletions,
            TotalErrors: substitutions + insertions + deletions,
            ErrorRate: errorRate,
            ReferenceLength: reference.Length,
            PredictionLength: prediction.Length);
    }

    /// <summary>
    /// 计算混淆矩阵。
    /// </summary>
    public static ConfusionMatrix CalculateConfusionMatrix(
        IReadOnlyList<string> references,
        IReadOnlyList<string> predictions,
        IReadOnlyList<char>? vocab = null)
    {
        vocab ??= "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".ToList();

        var matrix = new Dictionary<(char, char), int>();
        var refChars = new Dictionary<char, int>();
        var predChars = new Dictionary<char, int>();

        for (var i = 0; i < references.Count && i < predictions.Count; i++)
        {
            var refText = references[i];
            var predText = predictions[i];

            // 对齐字符序列
            var aligned = AlignSequences(refText, predText);

            foreach (var (r, p) in aligned)
            {
                if (vocab.Contains(r))
                {
                    refChars.TryGetValue(r, out var rc);
                    refChars[r] = rc + 1;
                }

                if (vocab.Contains(p))
                {
                    predChars.TryGetValue(p, out var pc);
                    predChars[p] = pc + 1;
                }

                if (vocab.Contains(r) && vocab.Contains(p))
                {
                    matrix.TryGetValue((r, p), out var count);
                    matrix[(r, p)] = count + 1;
                }
            }
        }

        return new ConfusionMatrix(
            Matrix: matrix,
            ReferenceCharCounts: refChars,
            PredictionCharCounts: predChars,
            Vocabulary: vocab.ToList());
    }

    /// <summary>
    /// 计算每字符准确率分解。
    /// </summary>
    public static Dictionary<char, float> CalculatePerCharacterAccuracy(
        IReadOnlyList<string> references,
        IReadOnlyList<string> predictions)
    {
        var charStats = new Dictionary<char, (int Correct, int Total)>();

        for (var i = 0; i < references.Count && i < predictions.Count; i++)
        {
            var refText = references[i];
            var predText = predictions[i];

            var aligned = AlignSequences(refText, predText);
            foreach (var (r, p) in aligned)
            {
                if (!charStats.ContainsKey(r))
                {
                    charStats[r] = (0, 0);
                }

                var (correct, total) = charStats[r];
                charStats[r] = (correct + (r == p ? 1 : 0), total + 1);
            }
        }

        return charStats.ToDictionary(
            kv => kv.Key,
            kv => kv.Value.Total == 0 ? 0.0f : kv.Value.Correct / (float)kv.Value.Total);
    }

    private static List<string> Tokenize(string text)
    {
        // Unicode 感知分词：
        // - 连续的拉丁字母/数字作为一个 token
        // - CJK 等非拉丁字符每个作为一个 token
        // - 忽略空白
        var tokens = new List<string>();
        var i = 0;
        while (i < text.Length)
        {
            var ch = text[i];
            if (char.IsWhiteSpace(ch))
            {
                i++;
                continue;
            }

            if (char.IsLetterOrDigit(ch) && ch < 0x3000)
            {
                // 拉丁/数字：连续收集
                var start = i;
                while (i < text.Length && char.IsLetterOrDigit(text[i]) && text[i] < 0x3000)
                {
                    i++;
                }

                tokens.Add(text[start..i]);
            }
            else
            {
                // CJK 或其他非拉丁字符：逐个 token
                tokens.Add(ch.ToString());
                i++;
            }
        }

        return tokens;
    }

    private static List<string> GetNgrams(List<string> tokens, int n)
    {
        var ngrams = new List<string>();
        for (var i = 0; i <= tokens.Count - n; i++)
        {
            ngrams.Add(string.Join("", tokens.Skip(i).Take(n)));
        }

        return ngrams;
    }

    private static Dictionary<string, int> CountNgrams(List<string> ngrams)
    {
        var counts = new Dictionary<string, int>();
        foreach (var ngram in ngrams)
        {
            counts.TryGetValue(ngram, out var count);
            counts[ngram] = count + 1;
        }

        return counts;
    }

    private static (int Substitutions, int Insertions, int Deletions) LevenshteinDetails(string left, string right)
    {
        if (left.Length == 0)
        {
            return (0, right.Length, 0);
        }

        if (right.Length == 0)
        {
            return (0, 0, left.Length);
        }

        var n = left.Length;
        var m = right.Length;
        var dp = new (int Cost, int Subs, int Ins, int Del)[n + 1, m + 1];

        // 初始化
        for (var j = 0; j <= m; j++)
        {
            dp[0, j] = (j, 0, j, 0);
        }

        for (var i = 0; i <= n; i++)
        {
            dp[i, 0] = (i, 0, 0, i);
        }

        // 填充 DP 表
        for (var i = 1; i <= n; i++)
        {
            for (var j = 1; j <= m; j++)
            {
                if (left[i - 1] == right[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1];
                }
                else
                {
                    var del = dp[i - 1, j];
                    var ins = dp[i, j - 1];
                    var sub = dp[i - 1, j - 1];

                    if (del.Cost <= ins.Cost && del.Cost <= sub.Cost)
                    {
                        dp[i, j] = (del.Cost + 1, del.Subs, del.Ins, del.Del + 1);
                    }
                    else if (ins.Cost <= sub.Cost)
                    {
                        dp[i, j] = (ins.Cost + 1, ins.Subs, ins.Ins + 1, ins.Del);
                    }
                    else
                    {
                        dp[i, j] = (sub.Cost + 1, sub.Subs + 1, sub.Ins, sub.Del);
                    }
                }
            }
        }

        return (dp[n, m].Subs, dp[n, m].Ins, dp[n, m].Del);
    }

    private static List<(char Ref, char Pred)> AlignSequences(string reference, string prediction)
    {
        // Needleman-Wunsch 全局序列对齐算法
        var n = reference.Length;
        var m = prediction.Length;
        var alignment = new List<(char, char)>();

        if (n == 0 && m == 0)
        {
            return alignment;
        }

        if (n == 0)
        {
            foreach (var p in prediction)
            {
                alignment.Add(('\0', p));
            }

            return alignment;
        }

        if (m == 0)
        {
            foreach (var r in reference)
            {
                alignment.Add((r, '\0'));
            }

            return alignment;
        }

        // DP 表: score[i, j] = 对齐 reference[0..i-1] 和 prediction[0..j-1] 的最优分数
        const int matchScore = 1;
        const int mismatchPenalty = -1;
        const int gapPenalty = -1;

        var score = new int[n + 1, m + 1];
        for (var i = 0; i <= n; i++) score[i, 0] = i * gapPenalty;
        for (var j = 0; j <= m; j++) score[0, j] = j * gapPenalty;

        for (var i = 1; i <= n; i++)
        {
            for (var j = 1; j <= m; j++)
            {
                var diagScore = score[i - 1, j - 1] + (reference[i - 1] == prediction[j - 1] ? matchScore : mismatchPenalty);
                var upScore = score[i - 1, j] + gapPenalty;
                var leftScore = score[i, j - 1] + gapPenalty;
                score[i, j] = Math.Max(diagScore, Math.Max(upScore, leftScore));
            }
        }

        // 回溯获取对齐
        var aligned = new List<(char, char)>();
        var ci = n;
        var cj = m;
        while (ci > 0 || cj > 0)
        {
            if (ci > 0 && cj > 0)
            {
                var diagScore = score[ci - 1, cj - 1] + (reference[ci - 1] == prediction[cj - 1] ? matchScore : mismatchPenalty);
                if (score[ci, cj] == diagScore)
                {
                    aligned.Add((reference[ci - 1], prediction[cj - 1]));
                    ci--;
                    cj--;
                    continue;
                }
            }

            if (ci > 0 && score[ci, cj] == score[ci - 1, cj] + gapPenalty)
            {
                aligned.Add((reference[ci - 1], '\0'));
                ci--;
            }
            else
            {
                aligned.Add(('\0', prediction[cj - 1]));
                cj--;
            }
        }

        aligned.Reverse();
        return aligned;
    }
}

/// <summary>
/// 错误分析结果。
/// </summary>
public sealed record ErrorAnalysis(
    int Substitutions,
    int Insertions,
    int Deletions,
    int TotalErrors,
    float ErrorRate,
    int ReferenceLength,
    int PredictionLength);

/// <summary>
/// 混淆矩阵。
/// </summary>
public sealed record ConfusionMatrix(
    Dictionary<(char Reference, char Prediction), int> Matrix,
    Dictionary<char, int> ReferenceCharCounts,
    Dictionary<char, int> PredictionCharCounts,
    List<char> Vocabulary);

/// <summary>
/// 扩展的 Rec 评估指标。
/// </summary>
public sealed record ExtendedRecEvalMetrics(
    float Accuracy,
    float CharacterAccuracy,
    float AvgEditDistance,
    float? BleuScore = null,
    float? ExpRate = null,
    ErrorAnalysis? ErrorAnalysis = null,
    Dictionary<char, float>? PerCharacterAccuracy = null);
