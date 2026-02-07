namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// Rec 标签编码器接口。
/// 不同算法使用不同的特殊 token 和 padding 策略。
/// </summary>
public interface IRecLabelEncoder
{
    /// <summary>
    /// 编码文本为模型可用的标签数组。
    /// </summary>
    /// <param name="text">原始文本</param>
    /// <returns>编码结果，包含 label 数组和实际文本长度；如果文本无效则返回 null。</returns>
    RecLabelEncodeResult? Encode(string text);

    /// <summary>
    /// 字符总数（含特殊 token）。
    /// </summary>
    int NumClasses { get; }

    /// <summary>
    /// 字符列表（含特殊 token）。
    /// </summary>
    IReadOnlyList<string> Characters { get; }
}

/// <summary>
/// 标签编码结果。
/// </summary>
public sealed record RecLabelEncodeResult(long[] Label, int Length);

/// <summary>
/// Multi-label 编码结果（CTC + GTC 双标签）。
/// </summary>
public sealed record MultiLabelEncodeResult(long[] LabelCtc, long[] LabelGtc, int Length, string GtcKey);
