using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// Attention Head：GRU 注意力解码器。
/// 用于 RARE 等 attention 系列算法。
/// </summary>
public sealed class AttentionHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly int _hiddenSize;
    private readonly int _outChannels;
    private readonly Module<Tensor, Tensor> _embedding;
    private readonly TorchSharp.Modules.GRUCell _cell;
    private readonly Module<Tensor, Tensor> _attnProj;
    private readonly Module<Tensor, Tensor> _outputProj;
    private readonly int _maxLen;

    public AttentionHead(int inChannels, int outChannels, int hiddenSize = 256, int maxLen = 25) : base(nameof(AttentionHead))
    {
        _hiddenSize = hiddenSize;
        _outChannels = outChannels;
        _maxLen = maxLen;
        _embedding = Embedding(outChannels, hiddenSize);
        _cell = nn.GRUCell(hiddenSize + inChannels, hiddenSize);
        _attnProj = Linear(hiddenSize, inChannels);
        _outputProj = Linear(hiddenSize, outChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C]
        var b = input.shape[0];
        using var hidden = zeros(b, _hiddenSize, device: input.device, dtype: input.dtype);
        using var sosToken = zeros(new long[] { b }, ScalarType.Int64, device: input.device);

        var outputs = new List<Tensor>();
        var h = hidden;
        var token = sosToken;

        for (var t = 0; t < _maxLen; t++)
        {
            using var emb = _embedding.call(token); // [B, hiddenSize]
            using var attnWeight = functional.softmax(_attnProj.call(h), dim: -1); // [B, inChannels]
            using var context = torch.bmm(
                attnWeight.unsqueeze(1), input).squeeze(1); // [B, C]
            using var cellInput = cat(new[] { emb, context }, dim: -1);
            h = _cell.call(cellInput, h);
            using var output = _outputProj.call(h); // [B, outChannels]
            outputs.Add(output.unsqueeze(1));

            if (!training)
            {
                token = output.argmax(-1);
            }
        }

        var result = cat(outputs.ToArray(), dim: 1); // [B, maxLen, outChannels]
        foreach (var o in outputs) o.Dispose();
        return result;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
