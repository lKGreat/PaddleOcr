using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Heads;

/// <summary>
/// CTC Head：Linear -> vocab_size，可选 2 层 FC。
/// Includes anti-collapse bias initialization: blank (index 0) gets a negative
/// initial bias to prevent CTC collapse where model predicts all-blank.
/// This is necessary because PyTorch's CTC loss has different gradient behavior
/// from PaddlePaddle's WarpCTC, making the blank class a local attractor.
/// </summary>
public sealed class CTCHead : Module<Tensor, Tensor>, IRecHead
{
    private readonly Module<Tensor, Tensor> _fc;
    private readonly int _outChannels;

    public CTCHead(int inChannels, int outChannels, int midChannels = 0) : base(nameof(CTCHead))
    {
        _outChannels = outChannels;
        if (midChannels > 0)
        {
            _fc = Sequential(
                Linear(inChannels, midChannels),
                ReLU(),
                Linear(midChannels, outChannels)
            );
        }
        else
        {
            _fc = Linear(inChannels, outChannels);
        }

        RegisterComponents();

        // Anti-CTC-collapse initialization: bias blank (index 0) negatively
        // so the model initially prefers characters over blank.
        // This prevents the CTC blank attractor problem in PyTorch.
        InitAntiCollapseBias();
    }

    /// <summary>
    /// Set the bias of the output layer so that blank (index 0) has a negative bias.
    /// This makes the initial softmax probability of blank lower than characters,
    /// giving the CTC alignment paths a chance to find character predictions.
    /// </summary>
    private void InitAntiCollapseBias()
    {
        // Find the last Linear layer in the fc module
        Module<Tensor, Tensor>? lastLinear = null;
        if (_fc is TorchSharp.Modules.Linear linear)
        {
            lastLinear = linear;
        }
        else if (_fc is TorchSharp.Modules.Sequential seq)
        {
            foreach (var (name, module) in seq.named_children())
            {
                if (module is TorchSharp.Modules.Linear lin)
                {
                    lastLinear = lin;
                }
            }
        }

        if (lastLinear is null) return;

        // Find the bias parameter
        foreach (var (name, param) in lastLinear.named_parameters())
        {
            if (name.Contains("bias") && param.shape[0] == _outChannels)
            {
                using var noGrad = torch.no_grad();
                // Set blank bias to -3.0 (P(blank) ≈ exp(-3) / sum ≈ 0.05/438 ≈ negligible)
                // Set all character biases to small positive value
                param.fill_(0.1f);
                param[0] = torch.tensor(-3.0f);
                break;
            }
        }
    }

    public override Tensor forward(Tensor input)
    {
        // input: [B, W, C] -> [B, W, outChannels]
        var logits = _fc.call(input);
        if (!training)
        {
            logits = functional.softmax(logits, dim: -1);
        }

        return logits;
    }

    public Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        var logits = forward(input);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
