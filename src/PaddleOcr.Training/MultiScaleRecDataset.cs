using PaddleOcr.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PaddleOcr.Training;

/// <summary>
/// MultiScaleRecDataset：多尺度数据集，PP-OCRv4 使用。
/// 在每个 batch 中使用相同的 (H, W)，但不同 batch 可以有不同宽度。
/// 参考: ppocr/data/multi_scale_dataloader.py - MultiScaleDataSet
/// </summary>
public sealed class MultiScaleRecDataset
{
    private readonly List<(string ImagePath, string Text)> _samples;
    private readonly int _height;
    private readonly int[] _widths;
    private readonly int _maxTextLength;
    private readonly IReadOnlyDictionary<char, int> _charToId;
    private readonly bool _enableAugmentation;
    private readonly IRecTrainingResize _resizer;

    /// <param name="labelFile">标签文件路径</param>
    /// <param name="dataDir">图像目录</param>
    /// <param name="height">固定高度</param>
    /// <param name="widths">候选宽度列表（例如 [320, 256, 192, 128, 96, 64]）</param>
    /// <param name="maxTextLength">最大文本长度</param>
    /// <param name="charToId">字符映射</param>
    /// <param name="enableAugmentation">是否启用数据增强</param>
    /// <param name="resizer">Resize 策略</param>
    public MultiScaleRecDataset(
        string labelFile,
        string dataDir,
        int height,
        int[] widths,
        int maxTextLength,
        IReadOnlyDictionary<char, int> charToId,
        bool enableAugmentation = false,
        IRecTrainingResize? resizer = null)
    {
        _height = height;
        _widths = widths.Length > 0 ? widths : [320];
        _maxTextLength = maxTextLength;
        _charToId = charToId;
        _enableAugmentation = enableAugmentation;
        _resizer = resizer ?? new RecResizeImg();
        _samples = LoadSamples(labelFile, dataDir);
    }

    public int Count => _samples.Count;

    /// <summary>
    /// 生成多尺度 batch。每个 batch 内图像使用相同宽度。
    /// 不同 batch 轮换使用 _widths 中的不同宽度。
    /// </summary>
    public IEnumerable<(float[] Images, long[] Labels, float[] ValidRatios, int Batch, int Width)> GetBatches(
        int batchSize, bool shuffle, Random rng)
    {
        var indices = Enumerable.Range(0, _samples.Count).ToList();
        if (shuffle)
        {
            for (var i = indices.Count - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        var widthIdx = 0;
        for (var offset = 0; offset < indices.Count; offset += batchSize)
        {
            var targetW = _widths[widthIdx % _widths.Length];
            widthIdx++;

            var take = Math.Min(batchSize, indices.Count - offset);
            var images = new float[take * 3 * _height * targetW];
            var labels = new long[take * _maxTextLength];
            var validRatios = new float[take];

            for (var bi = 0; bi < take; bi++)
            {
                var sample = _samples[indices[offset + bi]];
                var result = LoadAndResize(sample.ImagePath, targetW);
                Array.Copy(result.Data, 0, images, bi * result.Data.Length, result.Data.Length);
                validRatios[bi] = result.ValidRatio;

                var seq = SimpleRecDataset.Encode(sample.Text, _maxTextLength, _charToId);
                Array.Copy(seq, 0, labels, bi * _maxTextLength, _maxTextLength);
            }

            yield return (images, labels, validRatios, take, targetW);
        }
    }

    private RecResizeResult LoadAndResize(string imagePath, int targetW)
    {
        using var img = Image.Load<Rgb24>(imagePath);

        if (_enableAugmentation)
        {
            RecAugmentation.ApplyAugmentation(img);
        }

        return _resizer.Resize(img, 3, _height, targetW);
    }

    private static List<(string ImagePath, string Text)> LoadSamples(string labelFile, string dataDir)
    {
        if (!File.Exists(labelFile))
        {
            throw new FileNotFoundException($"Label file not found: {labelFile}");
        }

        var result = new List<(string, string)>();
        foreach (var line in File.ReadLines(labelFile))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var split = line.Split('\t', 2);
            if (split.Length != 2)
            {
                continue;
            }

            var img = split[0].Trim();
            var fullPath = Path.IsPathRooted(img) ? img : Path.GetFullPath(Path.Combine(dataDir, img));
            if (!File.Exists(fullPath))
            {
                continue;
            }

            var text = split[1].Trim();
            if (text.Length == 0)
            {
                continue;
            }

            result.Add((fullPath, text));
        }

        if (result.Count == 0)
        {
            throw new InvalidOperationException("No valid rec samples found.");
        }

        return result;
    }
}
