using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// PPHGNet_small backbone：HG_Block + ESE模块 + 自适应池化。
/// 用于 PP-OCRv4 的 SVTR_HGNet 配置。
/// </summary>
public sealed class PPHGNetSmall : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _stem;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _stages;
    private readonly Module<Tensor, Tensor> _pool;
    public int OutChannels { get; }

    public PPHGNetSmall(int inChannels = 3) : base(nameof(PPHGNetSmall))
    {
        // 简化的 HGNet stem
        _stem = Sequential(
            Conv2d(inChannels, 48, (3, 3), stride: (2, 1), padding: (1, 1), bias: false),
            BatchNorm2d(48),
            ReLU(),
            Conv2d(48, 48, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(48),
            ReLU(),
            Conv2d(48, 96, (3, 3), stride: (1, 2), padding: (1, 1), bias: false),
            BatchNorm2d(96),
            ReLU()
        );

        // 简化的 HG stages
        _stages = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        _stages.Add(HGBlock(96, 192, (2, 1)));
        _stages.Add(HGBlock(192, 384, (1, 2)));
        _stages.Add(HGBlock(384, 512, (2, 1)));
        _stages.Add(HGBlock(512, 768, 1));

        OutChannels = 768;
        _pool = AdaptiveAvgPool2d(new long[] { 1, 40 });
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _stem.call(input);
        foreach (var stage in _stages)
        {
            x = stage.call(x);
        }

        return _pool.call(x);
    }

    private static Module<Tensor, Tensor> HGBlock(int inCh, int outCh, (int, int) stride)
    {
        return Sequential(
            Conv2d(inCh, outCh, (3, 3), stride: (stride.Item1, stride.Item2), padding: (1, 1), bias: false),
            BatchNorm2d(outCh),
            ReLU(),
            Conv2d(outCh, outCh, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
    }

    private static Module<Tensor, Tensor> HGBlock(int inCh, int outCh, int stride)
    {
        return HGBlock(inCh, outCh, (stride, stride));
    }
}
