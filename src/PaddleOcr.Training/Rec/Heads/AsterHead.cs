using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// AsterHead：Attention-based recognition head for ASTER algorithm.
/// 使用 GRU + Attention 解码器。
/// 参考: ppocr/modeling/heads/rec_aster_head.py
/// </summary>
public sealed class AsterHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly int _numClasses;
    private readonly int _sDim;
    private readonly int _attDim;
    private readonly int _maxLen;
    private readonly Module<Tensor, Tensor> _embedder;
    private readonly Module<Tensor, Tensor> _embedFc;
    private readonly Module<Tensor, Tensor> _sEmbed;
    private readonly Module<Tensor, Tensor> _xEmbed;
    private readonly Module<Tensor, Tensor> _wEmbed;
    private readonly TorchSharp.Modules.Embedding _tgtEmbedding;
    private readonly TorchSharp.Modules.GRUCell _gru;
    private readonly Module<Tensor, Tensor> _fc;

    public AsterHead(int inChannels, int outChannels, int sDim = 512, int maxLen = 25) : base(nameof(AsterHead))
    {
        _numClasses = outChannels;
        _sDim = sDim;
        _attDim = sDim;
        _maxLen = maxLen;

        // Embedder
        _embedder = Linear(25 * inChannels, 300);
        _embedFc = Linear(300, sDim);

        // Attention unit
        _sEmbed = Linear(sDim, _attDim);
        _xEmbed = Linear(inChannels, _attDim);
        _wEmbed = Linear(_attDim, 1);

        // Decoder unit
        _tgtEmbedding = Embedding(outChannels + 1, _attDim);
        _gru = GRUCell(inChannels + _attDim, sDim);
        _fc = Linear(sDim, outChannels);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Simple forward: returns logits for inference
        var b = input.shape[0];
        var embed = ComputeEmbedding(input);
        var state = _embedFc.call(embed);

        var outputs = new List<Tensor>();
        for (var i = 0; i < _maxLen; i++)
        {
            Tensor yPrev;
            if (i == 0)
            {
                yPrev = torch.full(b, _numClasses, dtype: ScalarType.Int64, device: input.device);
            }
            else
            {
                var lastOutput = outputs[^1];
                yPrev = lastOutput.argmax(-1);
            }

            var (output, newState) = DecoderStep(input, state, yPrev);
            outputs.Add(output);
            state = newState;
        }

        return torch.stack(outputs.ToArray(), dim: 1);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var result = new Dictionary<string, Tensor>();
        var embed = ComputeEmbedding(input);
        result["embedding_vectors"] = embed;

        var state = _embedFc.call(embed);
        var b = input.shape[0];

        if (targets is not null && targets.TryGetValue("label", out var label))
        {
            var length = targets.TryGetValue("length", out var len) ? (int)len.max().item<long>() : _maxLen;
            var outputs = new List<Tensor>();

            for (var i = 0; i < length; i++)
            {
                Tensor yPrev;
                if (i == 0)
                {
                    yPrev = torch.full(b, _numClasses, dtype: ScalarType.Int64, device: input.device);
                }
                else
                {
                    yPrev = label.slice(1, i - 1, i, 1).squeeze(1);
                }

                var (output, newState) = DecoderStep(input, state, yPrev);
                outputs.Add(output.unsqueeze(1));
                state = newState;
            }

            result["rec_pred"] = torch.cat(outputs.ToArray(), dim: 1);
        }
        else
        {
            result["rec_pred"] = forward(input);
        }

        return result;
    }

    private Tensor ComputeEmbedding(Tensor x)
    {
        var flat = x.reshape(x.shape[0], -1);
        // Truncate/pad to match embedder input size
        var embedInSize = ((TorchSharp.Modules.Linear)_embedder).weight!.shape[1];
        if (flat.shape[1] > embedInSize)
        {
            flat = flat.slice(1, 0, embedInSize, 1);
        }
        else if (flat.shape[1] < embedInSize)
        {
            flat = functional.pad(flat, new long[] { 0, embedInSize - flat.shape[1] });
        }
        return _embedder.call(flat);
    }

    private (Tensor output, Tensor state) DecoderStep(Tensor x, Tensor state, Tensor yPrev)
    {
        var b = x.shape[0];
        var t = x.shape[1];

        // Attention
        var xFlat = x.reshape(-1, x.shape[2]);
        var xProj = _xEmbed.call(xFlat).reshape(b, t, -1);
        var sProj = _sEmbed.call(state).unsqueeze(1).expand(-1, t, -1);
        var sumTanh = torch.tanh(sProj + xProj).reshape(-1, _attDim);
        var alpha = functional.softmax(_wEmbed.call(sumTanh).reshape(b, t), dim: 1);
        var context = torch.matmul(alpha.unsqueeze(1), x).squeeze(1);

        // GRU step
        var yEmbed = _tgtEmbedding.call(yPrev);
        var concatContext = torch.cat([yEmbed, context], dim: 1);
        var newState = _gru.call(concatContext, state);
        var output = _fc.call(newState);

        return (output, newState);
    }
}
