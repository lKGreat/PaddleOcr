using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ResNet45 backbone：用于 SATRN 等模型。
/// </summary>
public sealed class ResNet45 : Module<Tensor, Tensor>, IRecBackbone
{
    private readonly Module<Tensor, Tensor> _features;
    public int OutChannels { get; }

    public ResNet45(int inChannels = 3, int[]? layers = null) : base(nameof(ResNet45))
    {
        layers ??= [3, 4, 6, 6, 3];
        var channels = new[] { 64, 128, 256, 256, 512, 512, 512 };

        var blocks = new List<(string, Module<Tensor, Tensor>)>
        {
            ("conv0", Conv2d(inChannels, channels[0], 3, stride: 1, padding: 1, bias: false)),
            ("bn0", BatchNorm2d(channels[0])),
            ("relu0", ReLU()),
            ("conv1", Conv2d(channels[0], channels[1], 3, stride: 1, padding: 1, bias: false)),
            ("bn1", BatchNorm2d(channels[1])),
            ("relu1", ReLU()),
            ("pool1", MaxPool2d(2, stride: 2))
        };

        var prevCh = channels[1];
        for (var stageIdx = 0; stageIdx < layers.Length; stageIdx++)
        {
            var outCh = channels[Math.Min(stageIdx + 2, channels.Length - 1)];
            for (var blockIdx = 0; blockIdx < layers[stageIdx]; blockIdx++)
            {
                var name = $"stage{stageIdx}_block{blockIdx}";
                var inCh = blockIdx == 0 ? prevCh : outCh;
                blocks.Add((name, BasicBlock(inCh, outCh)));
            }

            prevCh = outCh;
            if (stageIdx < layers.Length - 1)
            {
                blocks.Add(($"pool{stageIdx + 2}", MaxPool2d((2, 2), stride: (2, 1))));
            }
        }

        OutChannels = prevCh;
        _features = Sequential(blocks.ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return _features.call(input);
    }

    private static Module<Tensor, Tensor> BasicBlock(int inCh, int outCh)
    {
        return Sequential(
            Conv2d(inCh, outCh, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(outCh),
            ReLU(),
            Conv2d(outCh, outCh, 3, stride: 1, padding: 1, bias: false),
            BatchNorm2d(outCh),
            ReLU()
        );
    }
}
