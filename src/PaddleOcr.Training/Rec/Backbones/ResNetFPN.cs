using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ResNetFPN backbone：ResNet with Feature Pyramid Network for recognition.
/// 参考: ppocr/modeling/backbones/rec_resnet_fpn.py
/// </summary>
public sealed class ResNetFPN : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _conv;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _blockList;
    private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _baseBlock;
    private readonly int[] _depth;
    public int OutChannels { get; } = 512;

    public ResNetFPN(int inChannels = 1, int layers = 50) : base(nameof(ResNetFPN))
    {
        var supported = new Dictionary<int, (int[], bool)>
        {
            { 18, ([2, 2, 2, 2], false) },
            { 34, ([3, 4, 6, 3], false) },
            { 50, ([3, 4, 6, 3], true) },
            { 101, ([3, 4, 23, 3], true) },
            { 152, ([3, 8, 36, 3], true) },
        };

        var (depth, useBottleneck) = supported[layers];
        _depth = depth;
        var numFilters = new[] { 64, 128, 256, 512 };

        _conv = Sequential(
            Conv2d(inChannels, 64, 7, stride: 2, padding: 3, bias: false),
            BatchNorm2d(64),
            ReLU()
        );

        _blockList = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
        var inCh = 64;

        if (useBottleneck)
        {
            for (var block = 0; block < depth.Length; block++)
            {
                for (var i = 0; i < depth[block]; i++)
                {
                    var stride = i == 0 ? (block < 2 ? (2, 2) : (1, 1)) : (1, 1);
                    _blockList.Add(FPNBottleneckBlock(inCh, numFilters[block], stride));
                    inCh = numFilters[block] * 4;
                }
            }
        }
        else
        {
            for (var block = 0; block < depth.Length; block++)
            {
                for (var i = 0; i < depth[block]; i++)
                {
                    var stride = (i == 0 && block != 0) ? (2, 1) : (1, 1);
                    _blockList.Add(FPNBasicBlock(inCh, numFilters[block], stride, block == 0 && i == 0));
                    inCh = numFilters[block];
                }
            }
        }

        // FPN fusion layers
        var outChList = new[] { inCh / 4, inCh / 2, inCh };
        _baseBlock = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();

        for (var i = 0; i < 2; i++)
        {
            var idx = i == 0 ? -2 : -3;
            var inFpn = outChList[idx + 3] + outChList[idx + 2];
            _baseBlock.Add(Conv2d(inFpn, outChList[idx + 2], 1, bias: true));
            _baseBlock.Add(Conv2d(outChList[idx + 2], outChList[idx + 2], 3, padding: 1, bias: true));
            _baseBlock.Add(Sequential(BatchNorm2d(outChList[idx + 2]), ReLU()));
        }

        _baseBlock.Add(Conv2d(outChList[0], 512, 1, bias: true));

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _conv.call(input);

        var fpnList = new List<int>();
        var sum = 0;
        for (var i = 0; i < _depth.Length; i++)
        {
            sum += _depth[i];
            fpnList.Add(sum);
        }

        var feats = new List<Tensor>();
        for (var i = 0; i < _blockList.Count; i++)
        {
            x = _blockList[i].call(x);
            if (fpnList.Contains(i + 1))
            {
                feats.Add(x);
            }
        }

        var @base = feats[^1];

        // FPN top-down fusion
        var blockIdx = 0;
        for (var j = 0; j < 2; j++)
        {
            var prevFeat = feats[feats.Count - j - 2];
            @base = torch.cat([@base, prevFeat], dim: 1);
            @base = _baseBlock[blockIdx * 3].call(@base);
            @base = _baseBlock[blockIdx * 3 + 1].call(@base);
            @base = _baseBlock[blockIdx * 3 + 2].call(@base);
            blockIdx++;
        }

        @base = _baseBlock[^1].call(@base);
        return @base;
    }

    private static Module<Tensor, Tensor> FPNBasicBlock(int inCh, int outCh, (int, int) stride, bool isFirst)
    {
        return new FPNBasicBlockModule(inCh, outCh, stride, isFirst);
    }

    private static Module<Tensor, Tensor> FPNBottleneckBlock(int inCh, int outCh, (int, int) stride)
    {
        return new FPNBottleneckBlockModule(inCh, outCh, stride);
    }
}

internal sealed class FPNBasicBlockModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv0;
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly Module<Tensor, Tensor>? _shortcut;
    public int OutputChannels { get; }

    public FPNBasicBlockModule(int inCh, int outCh, (int, int) stride, bool isFirst) : base(nameof(FPNBasicBlockModule))
    {
        OutputChannels = outCh;
        _conv0 = Sequential(
            Conv2d(inCh, outCh, (3L, 3L), stride: ((long)stride.Item1, (long)stride.Item2), padding: (1L, 1L), bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
        _conv1 = Sequential(
            Conv2d(outCh, outCh, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(outCh)
        );

        if (inCh != outCh || stride != (1, 1) || isFirst)
        {
            _shortcut = Sequential(
                Conv2d(inCh, outCh, (1L, 1L), stride: ((long)stride.Item1, (long)stride.Item2), bias: false),
                BatchNorm2d(outCh)
            );
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var y = _conv0.call(input);
        y = _conv1.call(y);
        var shortcut = _shortcut is not null ? _shortcut.call(input) : input;
        return functional.relu(y + shortcut);
    }
}

internal sealed class FPNBottleneckBlockModule : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv0;
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly Module<Tensor, Tensor> _conv2;
    private readonly Module<Tensor, Tensor>? _shortcut;
    public int OutputChannels { get; }

    public FPNBottleneckBlockModule(int inCh, int outCh, (int, int) stride) : base(nameof(FPNBottleneckBlockModule))
    {
        OutputChannels = outCh * 4;
        _conv0 = Sequential(
            Conv2d(inCh, outCh, 1, bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
        _conv1 = Sequential(
            Conv2d(outCh, outCh, (3L, 3L), stride: ((long)stride.Item1, (long)stride.Item2), padding: (1L, 1L), bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
        _conv2 = Sequential(
            Conv2d(outCh, outCh * 4, 1, bias: false),
            BatchNorm2d(outCh * 4)
        );

        if (inCh != outCh * 4 || stride != (1, 1))
        {
            _shortcut = Sequential(
                Conv2d(inCh, outCh * 4, (1L, 1L), stride: ((long)stride.Item1, (long)stride.Item2), bias: false),
                BatchNorm2d(outCh * 4)
            );
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var y = _conv0.call(input);
        y = _conv1.call(y);
        y = _conv2.call(y);
        var shortcut = _shortcut is not null ? _shortcut.call(input) : input;
        return functional.relu(y + shortcut);
    }
}
