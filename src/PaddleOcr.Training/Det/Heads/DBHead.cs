using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Heads;

/// <summary>
/// DBHead：Differentiable Binarization 检测头。
/// 含 binarize 分支（shrink map）和 thresh 分支（threshold map），
/// 训练时通过 step function 生成 binary map。
/// 参考: ppocr/modeling/heads/det_db_head.py
/// </summary>
public sealed class DBHead : Module<Tensor, Dictionary<string, Tensor>>, IDetHead
{
    private readonly HeadBranch _binarize;
    private readonly HeadBranch _thresh;
    private readonly int _k;

    public DBHead(int inChannels, int k = 50) : base(nameof(DBHead))
    {
        _k = k;
        _binarize = new HeadBranch(inChannels, "binarize");
        _thresh = new HeadBranch(inChannels, "thresh");
        RegisterComponents();
    }

    public override Dictionary<string, Tensor> forward(Tensor x)
    {
        return Forward(x, training: true);
    }

    public Dictionary<string, Tensor> Forward(Tensor input, bool training)
    {
        var shrinkMaps = _binarize.call(input); // [B, 1, H, W]

        if (!training)
        {
            return new Dictionary<string, Tensor> { ["maps"] = shrinkMaps };
        }

        var thresholdMaps = _thresh.call(input); // [B, 1, H, W]
        // Differentiable binarization: DB = sigmoid(k * (shrink - thresh))
        var binaryMaps = torch.reciprocal(1 + torch.exp(-_k * (shrinkMaps - thresholdMaps)));
        var maps = torch.cat([shrinkMaps, thresholdMaps, binaryMaps], dim: 1); // [B, 3, H, W]

        return new Dictionary<string, Tensor> { ["maps"] = maps };
    }
}

/// <summary>
/// Head branch: Conv2d → BN → ReLU → ConvTranspose2d → BN → ReLU → ConvTranspose2d → Sigmoid
/// 从 FPN 特征上采样到原始分辨率的 1/4，再上采样到原始分辨率。
/// </summary>
internal sealed class HeadBranch : Module<Tensor, Tensor>
{
    private readonly Conv2d _conv1;
    private readonly BatchNorm2d _bn1;
    private readonly ConvTranspose2d _conv2;
    private readonly BatchNorm2d _bn2;
    private readonly ConvTranspose2d _conv3;

    public HeadBranch(int inChannels, string name) : base(name)
    {
        var midChannels = inChannels / 4;
        _conv1 = Conv2d(inChannels, midChannels, 3, padding: 1, bias: false);
        _bn1 = BatchNorm2d(midChannels);
        _conv2 = ConvTranspose2d(midChannels, midChannels, 2, stride: 2);
        _bn2 = BatchNorm2d(midChannels);
        _conv3 = ConvTranspose2d(midChannels, 1, 2, stride: 2);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = functional.relu(_bn1.call(_conv1.call(x)));
        x = functional.relu(_bn2.call(_conv2.call(x)));
        x = torch.sigmoid(_conv3.call(x));
        return x;
    }
}
