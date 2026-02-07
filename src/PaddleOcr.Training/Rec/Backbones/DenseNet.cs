using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// DenseNet backbone：用于 VisionLAN 等模型。
/// </summary>
public sealed class DenseNet : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public DenseNet(int inChannels = 3, int growthRate = 32, int[]? blockConfig = null) : base(nameof(DenseNet))
    {
        blockConfig ??= [6, 12, 24, 16];
        var numInitFeatures = 64;

        var blocks = new List<(string, Module<Tensor, Tensor>)>
        {
            ("conv0", Conv2d(inChannels, numInitFeatures, 7, stride: 2, padding: 3, bias: false)),
            ("bn0", BatchNorm2d(numInitFeatures)),
            ("relu0", ReLU()),
            ("pool0", MaxPool2d(3, stride: 2, padding: 1))
        };

        var numFeatures = numInitFeatures;
        for (var i = 0; i < blockConfig.Length; i++)
        {
            var numLayers = blockConfig[i];
            blocks.Add(($"denseblock{i}", new DenseBlock(numFeatures, growthRate, numLayers)));
            numFeatures += numLayers * growthRate;
            if (i != blockConfig.Length - 1)
            {
                blocks.Add(($"transition{i}", new Transition(numFeatures, numFeatures / 2)));
                numFeatures /= 2;
            }
        }

        OutChannels = numFeatures;
        _features = Sequential(blocks.ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }

    private sealed class DenseBlock : Module<Tensor, Tensor>
    {
        private readonly TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>> _layers;

        public DenseBlock(int numInputFeatures, int growthRate, int numLayers) : base("DenseBlock")
        {
            _layers = new TorchSharp.Modules.ModuleList<Module<Tensor, Tensor>>();
            for (var i = 0; i < numLayers; i++)
            {
                _layers.Add(new DenseLayer(numInputFeatures + i * growthRate, growthRate));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var features = new List<Tensor> { input };
            foreach (var layer in _layers)
            {
                var newFeatures = layer.call(cat(features.ToArray(), dim: 1));
                features.Add(newFeatures);
            }

            return cat(features.ToArray(), dim: 1);
        }
    }

    private sealed class DenseLayer : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _norm1;
        private readonly Module<Tensor, Tensor> _conv1;
        private readonly Module<Tensor, Tensor> _norm2;
        private readonly Module<Tensor, Tensor> _conv2;

        public DenseLayer(int numInputFeatures, int growthRate) : base("DenseLayer")
        {
            var bnSize = 4 * growthRate;
            _norm1 = BatchNorm2d(numInputFeatures);
            _conv1 = Conv2d(numInputFeatures, bnSize, 1, bias: false);
            _norm2 = BatchNorm2d(bnSize);
            _conv2 = Conv2d(bnSize, growthRate, 3, padding: 1, bias: false);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = functional.relu(_norm1.call(input));
            x = _conv1.call(x);
            var y = functional.relu(_norm2.call(x));
            return _conv2.call(y);
        }
    }

    private sealed class Transition : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _norm;
        private readonly Module<Tensor, Tensor> _conv;

        public Transition(int numInputFeatures, int numOutputFeatures) : base("Transition")
        {
            _norm = BatchNorm2d(numInputFeatures);
            _conv = Conv2d(numInputFeatures, numOutputFeatures, 1, bias: false);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = functional.relu(_norm.call(input));
            x = _conv.call(x);
            return functional.avg_pool2d(x, 2, stride: 2);
        }
    }
}
