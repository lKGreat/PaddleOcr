using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ResNetRFL backbone：Reciprocal Feature Learning 双分支共享骨干。
/// 参考: ppocr/modeling/backbones/rec_resnet_rfl.py
/// </summary>
public sealed class ResNetRFL : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly bool _useCnt;
    private readonly bool _useSeq;
    // Seq branch
    private readonly Module<Tensor, Tensor>? _maxpool3;
    private readonly Module<Tensor, Tensor>? _layer3;
    private readonly Module<Tensor, Tensor>? _conv3;
    private readonly Module<Tensor, Tensor>? _bn3;
    private readonly Module<Tensor, Tensor>? _layer4;
    private readonly Module<Tensor, Tensor>? _conv4_1;
    private readonly Module<Tensor, Tensor>? _bn4_1;
    private readonly Module<Tensor, Tensor>? _conv4_2;
    private readonly Module<Tensor, Tensor>? _bn4_2;
    // Cnt branch
    private readonly Module<Tensor, Tensor>? _vMaxpool3;
    private readonly Module<Tensor, Tensor>? _vLayer3;
    private readonly Module<Tensor, Tensor>? _vConv3;
    private readonly Module<Tensor, Tensor>? _vBn3;
    private readonly Module<Tensor, Tensor>? _vLayer4;
    private readonly Module<Tensor, Tensor>? _vConv4_1;
    private readonly Module<Tensor, Tensor>? _vBn4_1;
    private readonly Module<Tensor, Tensor>? _vConv4_2;
    private readonly Module<Tensor, Tensor>? _vBn4_2;
    public int OutChannels { get; }

    public ResNetRFL(int inChannels = 3, int outChannels = 512, bool useCnt = true, bool useSeq = true) : base(nameof(ResNetRFL))
    {
        _useCnt = useCnt;
        _useSeq = useSeq;
        OutChannels = outChannels;

        _backbone = BuildBaseResNet(inChannels, outChannels);

        var outChBlock = new[] { outChannels / 4, outChannels / 2, outChannels, outChannels };

        if (useSeq)
        {
            var inplanes = outChannels / 2;
            _maxpool3 = MaxPool2d((2, 2), stride: (2, 1), padding: (0, 1));
            (_layer3, inplanes) = MakeLayer(inplanes, outChBlock[2], 5);
            _conv3 = Conv2d(outChBlock[2], outChBlock[2], 3, stride: 1, padding: 1, bias: false);
            _bn3 = BatchNorm2d(outChBlock[2]);
            (_layer4, inplanes) = MakeLayer(inplanes, outChBlock[3], 3);
            _conv4_1 = Conv2d(outChBlock[3], outChBlock[3], (2L, 2L), stride: (2L, 1L), padding: (0L, 1L), bias: false);
            _bn4_1 = BatchNorm2d(outChBlock[3]);
            _conv4_2 = Conv2d(outChBlock[3], outChBlock[3], 2, stride: 1, padding: 0L, bias: false);
            _bn4_2 = BatchNorm2d(outChBlock[3]);
        }

        if (useCnt)
        {
            var inplanes = outChannels / 2;
            _vMaxpool3 = MaxPool2d((2, 2), stride: (2, 1), padding: (0, 1));
            (_vLayer3, inplanes) = MakeLayer(inplanes, outChBlock[2], 5);
            _vConv3 = Conv2d(outChBlock[2], outChBlock[2], 3, stride: 1, padding: 1, bias: false);
            _vBn3 = BatchNorm2d(outChBlock[2]);
            (_vLayer4, inplanes) = MakeLayer(inplanes, outChBlock[3], 3);
            _vConv4_1 = Conv2d(outChBlock[3], outChBlock[3], (2L, 2L), stride: (2L, 1L), padding: (0L, 1L), bias: false);
            _vBn4_1 = BatchNorm2d(outChBlock[3]);
            _vConv4_2 = Conv2d(outChBlock[3], outChBlock[3], 2, stride: 1, padding: 0L, bias: false);
            _vBn4_2 = BatchNorm2d(outChBlock[3]);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Returns [visual_feature_3, x_3] as a concatenated tensor
        // The head separates them. For standard Module<Tensor, Tensor> interface, return x_3 (seq feature).
        var x1 = _backbone.call(input);

        if (_useSeq && _maxpool3 is not null)
        {
            var x = _maxpool3.call(x1);
            x = _layer3!.call(x);
            x = functional.relu(_bn3!.call(_conv3!.call(x)));
            x = _layer4!.call(x);
            x = functional.relu(_bn4_1!.call(_conv4_1!.call(x)));
            x = functional.relu(_bn4_2!.call(_conv4_2!.call(x)));
            return x;
        }

        return x1;
    }

    private static Module<Tensor, Tensor> BuildBaseResNet(int inChannels, int outChannels)
    {
        var outChBlock = new[] { outChannels / 4, outChannels / 2, outChannels, outChannels };
        var inplanes = outChannels / 8;

        var layers = new List<(string, Module<Tensor, Tensor>)>
        {
            ("conv0_1", Conv2d(inChannels, outChannels / 16, 3, stride: 1, padding: 1, bias: false)),
            ("bn0_1", BatchNorm2d(outChannels / 16)),
            ("relu0_1", ReLU()),
            ("conv0_2", Conv2d(outChannels / 16, inplanes, 3, stride: 1, padding: 1, bias: false)),
            ("bn0_2", BatchNorm2d(inplanes)),
            ("relu0_2", ReLU()),
            ("pool1", MaxPool2d(2, stride: 2))
        };

        // Layer 1
        var (layer1, newInplanes) = MakeLayer(inplanes, outChBlock[0], 1);
        layers.Add(("layer1", layer1));
        layers.Add(("conv1", Conv2d(outChBlock[0], outChBlock[0], 3, stride: 1, padding: 1, bias: false)));
        layers.Add(("bn1", BatchNorm2d(outChBlock[0])));
        layers.Add(("relu1", ReLU()));

        // Layer 2
        layers.Add(("pool2", MaxPool2d(2, stride: 2)));
        var (layer2, _) = MakeLayer(newInplanes, outChBlock[1], 2);
        layers.Add(("layer2", layer2));
        layers.Add(("conv2", Conv2d(outChBlock[1], outChBlock[1], 3, stride: 1, padding: 1, bias: false)));
        layers.Add(("bn2", BatchNorm2d(outChBlock[1])));
        layers.Add(("relu2", ReLU()));

        return Sequential(layers.ToArray());
    }

    private static (Module<Tensor, Tensor> layer, int outPlanes) MakeLayer(int inplanes, int planes, int blocks)
    {
        var layerList = new List<(string, Module<Tensor, Tensor>)>();
        Module<Tensor, Tensor>? downsample = null;
        if (inplanes != planes)
        {
            downsample = Sequential(
                Conv2d(inplanes, planes, 1, bias: false),
                BatchNorm2d(planes)
            );
        }
        layerList.Add(("0", new RFLBasicBlock(inplanes, planes, downsample)));
        for (var i = 1; i < blocks; i++)
        {
            layerList.Add(($"{i}", new RFLBasicBlock(planes, planes, null)));
        }
        return (Sequential(layerList.ToArray()), planes);
    }
}

internal sealed class RFLBasicBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly Module<Tensor, Tensor> _bn1;
    private readonly Module<Tensor, Tensor> _conv2;
    private readonly Module<Tensor, Tensor> _bn2;
    private readonly Module<Tensor, Tensor>? _downsample;

    public RFLBasicBlock(int inplanes, int planes, Module<Tensor, Tensor>? downsample) : base(nameof(RFLBasicBlock))
    {
        _conv1 = Conv2d(inplanes, planes, 3, stride: 1, padding: 1, bias: false);
        _bn1 = BatchNorm2d(planes);
        _conv2 = Conv2d(planes, planes, 3, stride: 1, padding: 1, bias: false);
        _bn2 = BatchNorm2d(planes);
        _downsample = downsample;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var residual = _downsample is not null ? _downsample.call(input) : input;
        var x = functional.relu(_bn1.call(_conv1.call(input)));
        x = _bn2.call(_conv2.call(x));
        return functional.relu(x + residual);
    }
}
