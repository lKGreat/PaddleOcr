using System.Text;

namespace PaddleOcr.Data.LabelEncoders;

/// <summary>
/// Rec 标签编码器基类。
/// 处理字典加载和基本的 text-to-index 转换。
/// 参考: ppocr/data/imaug/label_ops.py - BaseRecLabelEncode
/// </summary>
public abstract class BaseRecLabelEncoder : IRecLabelEncoder
{
    private readonly Dictionary<string, int> _dict;
    private readonly List<string> _characters;
    private readonly int _maxTextLen;
    private readonly bool _lower;

    protected BaseRecLabelEncoder(int maxTextLength, string? characterDictPath, bool useSpaceChar, bool lower = false)
    {
        _maxTextLen = maxTextLength;
        _lower = lower;

        var dictChars = new List<string>();
        if (!string.IsNullOrWhiteSpace(characterDictPath) && File.Exists(characterDictPath))
        {
            foreach (var line in File.ReadLines(characterDictPath, Encoding.UTF8))
            {
                var token = line.TrimEnd('\r', '\n');
                if (token.Length > 0)
                {
                    dictChars.Add(token);
                }
            }

            if (useSpaceChar && !dictChars.Contains(" "))
            {
                dictChars.Add(" ");
            }
        }
        else
        {
            dictChars.AddRange("0123456789abcdefghijklmnopqrstuvwxyz".Select(c => c.ToString()));
        }

        // 子类添加特殊字符
        dictChars = AddSpecialChar(dictChars);

        _dict = new Dictionary<string, int>();
        _characters = new List<string>();
        for (var i = 0; i < dictChars.Count; i++)
        {
            _dict[dictChars[i]] = i;
            _characters.Add(dictChars[i]);
        }
    }

    public int NumClasses => _characters.Count;
    public IReadOnlyList<string> Characters => _characters;
    protected int MaxTextLen => _maxTextLen;
    protected IReadOnlyDictionary<string, int> Dict => _dict;

    /// <summary>
    /// 子类覆盖此方法添加算法特定的特殊字符。
    /// </summary>
    protected abstract List<string> AddSpecialChar(List<string> dictCharacter);

    /// <summary>
    /// 将文本转换为字符索引列表。
    /// </summary>
    protected List<int>? EncodeText(string text)
    {
        if (text.Length == 0 || text.Length > _maxTextLen)
        {
            return null;
        }

        var processedText = _lower ? text.ToLowerInvariant() : text;
        var result = new List<int>();
        foreach (var ch in processedText)
        {
            var key = ch.ToString();
            if (_dict.TryGetValue(key, out var idx))
            {
                result.Add(idx);
            }
            // 忽略不在字典中的字符
        }

        return result.Count == 0 ? null : result;
    }

    public abstract RecLabelEncodeResult? Encode(string text);
}
