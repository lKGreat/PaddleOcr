using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ResNet_ASTER backbone：用于 ASTER 识别算法的 ResNet 变体。
/// 参考: ppocr/modeling/backbones/rec_resnet_aster.py
/// </summary>
public sealed class ResNetAster : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _layer0;
    private readonly Module<Tensor, Tensor> _layer1;
    private readonly Module<Tensor, Tensor> _layer2;
    private readonly Module<Tensor, Tensor> _layer3;
    private readonly Module<Tensor, Tensor> _layer4;
    private readonly Module<Tensor, Tensor> _layer5;
    private readonly bool _withLstm;
    private readonly TorchSharp.Modules.LSTM? _rnn;
    public int OutChannels { get; }

    public ResNetAster(int inChannels = 3, bool withLstm = true) : base(nameof(ResNetAster))
    {
        _withLstm = withLstm;

        _layer0 = Sequential(
            Conv2d(inChannels, 32, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(32),
            ReLU()
        );

        _layer1 = MakeLayer(32, 32, 3, (2, 2));
        _layer2 = MakeLayer(32, 64, 4, (2, 2));
        _layer3 = MakeLayer(64, 128, 6, (2, 1));
        _layer4 = MakeLayer(128, 256, 6, (2, 1));
        _layer5 = MakeLayer(256, 512, 3, (2, 1));

        if (withLstm)
        {
            _rnn = LSTM(512, 256, numLayers: 2, bidirectional: true, batchFirst: true);
            OutChannels = 512; // 2 * 256
        }
        else
        {
            OutChannels = 512;
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _layer0.call(input);
        x = _layer1.call(x);
        x = _layer2.call(x);
        x = _layer3.call(x);
        x = _layer4.call(x);
        x = _layer5.call(x);

        // [N, C, 1, W] -> [N, C, W] -> [N, W, C]
        x = x.squeeze(2);
        x = x.permute(0, 2, 1);

        if (_withLstm && _rnn is not null)
        {
            var (output, _, _) = _rnn.call(x);
            return output;
        }

        return x;
    }

    private Module<Tensor, Tensor> MakeLayer(int inplanes, int planes, int blocks, (int, int) stride)
    {
        var layers = new List<(string, Module<Tensor, Tensor>)>();

        // Downsample for first block
        Module<Tensor, Tensor>? downsample = null;
        if (stride != (1, 1) || inplanes != planes)
        {
            downsample = Sequential(
                Conv2d(inplanes, planes, (1L, 1L), stride: ((long)stride.Item1, (long)stride.Item2), bias: false),
                BatchNorm2d(planes)
            );
        }

        layers.Add(("block0", new AsterBlock(inplanes, planes, stride, downsample)));
        for (var i = 1; i < blocks; i++)
        {
            layers.Add(($"block{i}", new AsterBlock(planes, planes, (1, 1), null)));
        }

        return Sequential(layers.ToArray());
    }
}

internal sealed class AsterBlock : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv1;
    private readonly Module<Tensor, Tensor> _bn1;
    private readonly Module<Tensor, Tensor> _conv2;
    private readonly Module<Tensor, Tensor> _bn2;
    private readonly Module<Tensor, Tensor>? _downsample;

    public AsterBlock(int inplanes, int planes, (int, int) stride, Module<Tensor, Tensor>? downsample) : base(nameof(AsterBlock))
    {
        _conv1 = Conv2d(inplanes, planes, (1L, 1L), stride: ((long)stride.Item1, (long)stride.Item2), bias: false);
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
