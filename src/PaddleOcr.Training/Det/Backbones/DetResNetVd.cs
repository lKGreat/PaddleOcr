using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Backbones;

/// <summary>
/// ResNet_vd for Detection：输出多尺度特征图列表 [c2, c3, c4, c5]。
/// 支持 layers=18/34/50（带 deformable conv 选项可后续扩展）。
/// 参考: ppocr/modeling/backbones/det_resnet_vd.py
/// </summary>
public sealed class DetResNetVd : Module<Tensor, Tensor[]>, IDetBackbone
{
    private readonly Module<Tensor, Tensor> _stem;
    private readonly ModuleList<Sequential> _layers;

    public int[] OutChannels { get; }

    public DetResNetVd(int inChannels = 3, int layers = 18) : base(nameof(DetResNetVd))
    {
        int[] blockNums;
        bool useBottleneck;
        int[] channelList;

        switch (layers)
        {
            case 18:
                blockNums = [2, 2, 2, 2];
                useBottleneck = false;
                channelList = [64, 64, 128, 256, 512];
                break;
            case 34:
                blockNums = [3, 4, 6, 3];
                useBottleneck = false;
                channelList = [64, 64, 128, 256, 512];
                break;
            case 50:
                blockNums = [3, 4, 6, 3];
                useBottleneck = true;
                channelList = [64, 256, 512, 1024, 2048];
                break;
            default:
                throw new ArgumentException($"Unsupported ResNet layers: {layers}");
        }

        // Stem: 3 conv layers (vd style)
        var stemMid = 32;
        _stem = Sequential(
            Conv2d(inChannels, stemMid, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(stemMid),
            ReLU(),
            Conv2d(stemMid, stemMid, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(stemMid),
            ReLU(),
            Conv2d(stemMid, channelList[0], 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(channelList[0]),
            ReLU(),
            MaxPool2d(3, stride: 2, padding: 1));

        _layers = new ModuleList<Sequential>();
        var outChs = new int[4];

        for (int stageIdx = 0; stageIdx < 4; stageIdx++)
        {
            var blocks = new List<Module<Tensor, Tensor>>();
            var inCh = channelList[stageIdx];
            var outCh = channelList[stageIdx + 1];

            for (int blockIdx = 0; blockIdx < blockNums[stageIdx]; blockIdx++)
            {
                var stride = (blockIdx == 0 && stageIdx > 0) ? 2 : 1;
                var blockInCh = blockIdx == 0 ? inCh : outCh;

                if (useBottleneck)
                {
                    blocks.Add(new BottleneckBlock(blockInCh, outCh / 4, outCh, stride));
                }
                else
                {
                    blocks.Add(new BasicBlock(blockInCh, outCh, stride));
                }
            }

            _layers.Add(Sequential(blocks.ToArray()));
            outChs[stageIdx] = outCh;
        }

        OutChannels = outChs;
        RegisterComponents();
    }

    public override Tensor[] forward(Tensor x)
    {
        x = ((Module<Tensor, Tensor>)_stem).call(x);
        var outputs = new Tensor[4];
        for (int i = 0; i < 4; i++)
        {
            x = _layers[i].call(x);
            outputs[i] = x;
        }
        return outputs;
    }
}

internal sealed class BasicBlock : Module<Tensor, Tensor>
{
    private readonly Conv2d _conv1;
    private readonly BatchNorm2d _bn1;
    private readonly Conv2d _conv2;
    private readonly BatchNorm2d _bn2;
    private readonly Module<Tensor, Tensor>? _downsample;

    public BasicBlock(int inChannels, int outChannels, int stride)
        : base(nameof(BasicBlock))
    {
        _conv1 = Conv2d(inChannels, outChannels, 3, stride: stride, padding: 1, bias: false);
        _bn1 = BatchNorm2d(outChannels);
        _conv2 = Conv2d(outChannels, outChannels, 3, stride: 1, padding: 1, bias: false);
        _bn2 = BatchNorm2d(outChannels);

        if (stride != 1 || inChannels != outChannels)
        {
            _downsample = Sequential(
                Conv2d(inChannels, outChannels, 1, stride: stride, bias: false),
                BatchNorm2d(outChannels));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var identity = input;
        var x = functional.relu(_bn1.call(_conv1.call(input)));
        x = _bn2.call(_conv2.call(x));

        if (_downsample is not null)
        {
            identity = ((Module<Tensor, Tensor>)_downsample).call(identity);
        }

        x = x + identity;
        return functional.relu(x);
    }
}

internal sealed class BottleneckBlock : Module<Tensor, Tensor>
{
    private readonly Conv2d _conv1;
    private readonly BatchNorm2d _bn1;
    private readonly Conv2d _conv2;
    private readonly BatchNorm2d _bn2;
    private readonly Conv2d _conv3;
    private readonly BatchNorm2d _bn3;
    private readonly Module<Tensor, Tensor>? _downsample;

    public BottleneckBlock(int inChannels, int midChannels, int outChannels, int stride)
        : base(nameof(BottleneckBlock))
    {
        _conv1 = Conv2d(inChannels, midChannels, 1, bias: false);
        _bn1 = BatchNorm2d(midChannels);
        _conv2 = Conv2d(midChannels, midChannels, 3, stride: stride, padding: 1, bias: false);
        _bn2 = BatchNorm2d(midChannels);
        _conv3 = Conv2d(midChannels, outChannels, 1, bias: false);
        _bn3 = BatchNorm2d(outChannels);

        if (stride != 1 || inChannels != outChannels)
        {
            _downsample = Sequential(
                Conv2d(inChannels, outChannels, 1, stride: stride, bias: false),
                BatchNorm2d(outChannels));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var identity = input;
        var x = functional.relu(_bn1.call(_conv1.call(input)));
        x = functional.relu(_bn2.call(_conv2.call(x)));
        x = _bn3.call(_conv3.call(x));

        if (_downsample is not null)
        {
            identity = ((Module<Tensor, Tensor>)_downsample).call(identity);
        }

        x = x + identity;
        return functional.relu(x);
    }
}
