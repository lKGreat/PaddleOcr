using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// MultiHead：组合 CTC + SAR/NRTR，共享编码器，分别输出。
/// 用于 PP-OCRv3/v4 的多头训练。
/// </summary>
public sealed class MultiHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly CTCHead _ctcHead;
    private readonly AttentionHead? _attnHead;
    private readonly int _outChannels;

    public MultiHead(int inChannels, int outChannelsCtc, int outChannelsAttn = 0, int hiddenSize = 48)
        : base(nameof(MultiHead))
    {
        _outChannels = outChannelsCtc;
        _ctcHead = new CTCHead(inChannels, outChannelsCtc);
        if (outChannelsAttn > 0)
        {
            _attnHead = new AttentionHead(inChannels, outChannelsAttn, hiddenSize);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // 默认返回 CTC 输出
        return _ctcHead.forward(input);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var result = new Dictionary<string, Tensor>();

        // CTC 分支
        var ctcOut = _ctcHead.Forward(input, targets);
        result["ctc"] = ctcOut["predict"];

        // Attention 分支
        if (_attnHead is not null)
        {
            var attnOut = _attnHead.Forward(input, targets);
            result["gtc"] = attnOut["predict"];
        }

        result["predict"] = result["ctc"];
        return result;
    }
}
