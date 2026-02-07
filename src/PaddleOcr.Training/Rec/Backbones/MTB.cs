using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// MTB (Multi-Task Backbone)：NRTR 的多任务 backbone。
/// </summary>
public sealed class MTB : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public MTB(int inChannels = 3) : base(nameof(MTB))
    {
        OutChannels = 512;

        _features = Sequential(
            // Initial conv layers
            Conv2d(inChannels, 64, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 128, 3, stride: 2, padding: 1, bias: false),
            BatchNorm2d(128),
            ReLU(),
            // ResNet-like blocks
            ResBlock(128, 256, stride: 2),
            ResBlock(256, 256),
            ResBlock(256, 512, stride: 2),
            ResBlock(512, 512),
            ResBlock(512, 512)
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }

    private static Module<Tensor, Tensor> ResBlock(int inCh, int outCh, int stride = 1)
    {
        var layers = new List<(string, Module<Tensor, Tensor>)>
        {
            ("conv1", Conv2d(inCh, outCh, 3, stride: stride, padding: 1, bias: false)),
            ("bn1", BatchNorm2d(outCh)),
            ("relu1", ReLU()),
            ("conv2", Conv2d(outCh, outCh, 3, stride: 1, padding: 1, bias: false)),
            ("bn2", BatchNorm2d(outCh))
        };

        if (inCh != outCh || stride != 1)
        {
            layers.Insert(0, ("downsample", Sequential(
                Conv2d(inCh, outCh, 1, stride: stride, bias: false),
                BatchNorm2d(outCh)
            )));
        }

        return new ResidualBlock(layers);
    }

    private sealed class ResidualBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _main;
        private readonly Module<Tensor, Tensor>? _downsample;

        public ResidualBlock(List<(string, Module<Tensor, Tensor>)> layers) : base("ResBlock")
        {
            var mainLayers = new List<(string, Module<Tensor, Tensor>)>();
            Module<Tensor, Tensor>? downsample = null;

            foreach (var (name, layer) in layers)
            {
                if (name == "downsample")
                {
                    downsample = layer;
                }
                else
                {
                    mainLayers.Add((name, layer));
                }
            }

            _main = Sequential(mainLayers.ToArray());
            _downsample = downsample;
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var residual = _downsample?.call(input) ?? input;
            var x = _main.call(input);
            return functional.relu(x + residual);
        }
    }
}
