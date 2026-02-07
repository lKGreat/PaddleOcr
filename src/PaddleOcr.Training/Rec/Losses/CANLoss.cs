using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Losses;

/// <summary>
/// CANLoss：CAN 损失，包含两部分：
/// 1. word_average_loss：符号的平均准确率（CrossEntropy）
/// 2. counting_loss：每个符号的计数损失（SmoothL1Loss × 3 个 counting head）
/// 参考: ppocr/losses/rec_can_loss.py
/// </summary>
public sealed class CANLoss : IRecLoss
{
    private readonly int _outChannel;
    private readonly int _ratio;

    public CANLoss(int outChannel = 111, int ratio = 16)
    {
        _outChannel = outChannel;
        _ratio = ratio;
    }

    public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch)
    {
        // word_probs: [B, T, C], counting_preds/1/2: [B, C]
        var wordProbs = predictions.ContainsKey("word_probs") ? predictions["word_probs"] : predictions["predict"];
        var targets = batch["label"].to(ScalarType.Int64);

        // 1. word loss: CrossEntropy
        var wordLoss = functional.cross_entropy(
            wordProbs.reshape(-1, wordProbs.shape[^1]),
            targets.reshape(-1));

        // 2. counting loss (如果模型输出了 counting heads)
        Tensor countingLoss;
        if (predictions.ContainsKey("counting_preds") &&
            predictions.ContainsKey("counting_preds1") &&
            predictions.ContainsKey("counting_preds2"))
        {
            var countingLabels = GenCountingLabel(targets, _outChannel);
            using var cl1 = functional.smooth_l1_loss(predictions["counting_preds"], countingLabels);
            using var cl2 = functional.smooth_l1_loss(predictions["counting_preds1"], countingLabels);
            using var cl3 = functional.smooth_l1_loss(predictions["counting_preds2"], countingLabels);
            countingLoss = cl1 + cl2 + cl3;
            countingLabels.Dispose();
        }
        else
        {
            countingLoss = zeros(1, device: wordLoss.device);
        }

        using var totalLoss = wordLoss + countingLoss;
        countingLoss.Dispose();

        return new Dictionary<string, Tensor>
        {
            ["loss"] = totalLoss.clone(),
            ["word_loss"] = wordLoss,
        };
    }

    /// <summary>
    /// 生成 counting label：统计每个 batch 中每个符号出现的次数。
    /// </summary>
    private static Tensor GenCountingLabel(Tensor labels, int channel)
    {
        var b = (int)labels.shape[0];
        var t = (int)labels.shape[1];
        var countingData = new float[b * channel];

        var labelsData = labels.cpu().data<long>().ToArray();
        // 忽略的 token: 0(blank), 1, 和最后几个特殊 token
        var ignore = new HashSet<long> { 0, 1 };
        if (channel > 4)
        {
            ignore.Add(channel - 4);
            ignore.Add(channel - 3);
            ignore.Add(channel - 2);
            ignore.Add(channel - 1);
        }

        for (var i = 0; i < b; i++)
        {
            for (var j = 0; j < t; j++)
            {
                var k = labelsData[i * t + j];
                if (k >= 0 && k < channel && !ignore.Contains(k))
                {
                    countingData[i * channel + k] += 1;
                }
            }
        }

        return tensor(countingData, ScalarType.Float32).reshape(b, channel).to(labels.device);
    }
}
