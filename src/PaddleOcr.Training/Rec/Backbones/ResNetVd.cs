using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Backbones;

/// <summary>
/// ResNet_vd for Rec：PP-OCRv2 backbone。
/// 特征：vd stem (3个3x3 conv 替代 7x7), shortcut 使用 AvgPool + 1x1 Conv。
/// 支持 ResNet18_vd ~ ResNet200_vd。
/// 参考: ppocr/modeling/backbones/rec_resnet_vd.py
/// </summary>
public sealed class ResNetVd : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _conv1_1;
    private readonly Module<Tensor, Tensor> _conv1_2;
    private readonly Module<Tensor, Tensor> _conv1_3;
    private readonly MaxPool2d _pool;
    private readonly ModuleList<Module<Tensor, Tensor>> _blocks;
    private readonly MaxPool2d _outPool;

    public int OutChannels { get; }

    /// <param name="inChannels">输入通道数，默认3</param>
    /// <param name="layers">层数：18, 34, 50, 101, 152, 200</param>
    public ResNetVd(int inChannels = 3, int layers = 34) : base(nameof(ResNetVd))
    {
        var depth = layers switch
        {
            18 => new[] { 2, 2, 2, 2 },
            34 or 50 => new[] { 3, 4, 6, 3 },
            101 => new[] { 3, 4, 23, 3 },
            152 => new[] { 3, 8, 36, 3 },
            200 => new[] { 3, 12, 48, 3 },
            _ => throw new ArgumentException($"Unsupported layers: {layers}. Supported: 18, 34, 50, 101, 152, 200")
        };

        var numChannels = layers >= 50
            ? new[] { 64, 256, 512, 1024 }
            : new[] { 64, 64, 128, 256 };
        var numFilters = new[] { 64, 128, 256, 512 };

        // vd stem: 3 个 3x3 conv
        _conv1_1 = MakeConvBnRelu(inChannels, 32, 3, (1, 1), 1);
        _conv1_2 = MakeConvBnRelu(32, 32, 3, (1, 1), 1);
        _conv1_3 = MakeConvBnRelu(32, 64, 3, (1, 1), 1);
        _pool = MaxPool2d(kernel_size: 3, stride: 2, padding: 1);

        _blocks = new ModuleList<Module<Tensor, Tensor>>();

        if (layers >= 50)
        {
            // Bottleneck blocks
            for (var block = 0; block < depth.Length; block++)
            {
                var shortcut = false;
                for (var i = 0; i < depth[block]; i++)
                {
                    var strideH = (i == 0 && block != 0) ? 2 : 1;
                    var inCh = i == 0 ? numChannels[block] : numFilters[block] * 4;
                    var outCh = numFilters[block];
                    var ifFirst = block == 0 && i == 0;
                    _blocks.Add(new BottleneckBlock(inCh, outCh, (strideH, 1), shortcut, ifFirst));
                    shortcut = true;
                }
                OutChannels = numFilters[block] * 4;
            }
        }
        else
        {
            // Basic blocks
            for (var block = 0; block < depth.Length; block++)
            {
                var shortcut = false;
                for (var i = 0; i < depth[block]; i++)
                {
                    var strideH = (i == 0 && block != 0) ? 2 : 1;
                    var inCh = i == 0 ? numChannels[block] : numFilters[block];
                    var outCh = numFilters[block];
                    var ifFirst = block == 0 && i == 0;
                    _blocks.Add(new BasicBlock(inCh, outCh, (strideH, 1), shortcut, ifFirst));
                    shortcut = true;
                }
                OutChannels = numFilters[block];
            }
        }

        _outPool = MaxPool2d(kernel_size: 2, stride: 2);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _conv1_1.call(x);
        x = _conv1_2.call(x);
        x = _conv1_3.call(x);
        x = _pool.call(x);
        foreach (var block in _blocks)
        {
            x = block.call(x);
        }
        x = _outPool.call(x);
        return x;
    }

    private static Module<Tensor, Tensor> MakeConvBnRelu(int inCh, int outCh, int kernel, (int h, int w) stride, int padding)
    {
        return Sequential(
            Conv2d(inCh, outCh, (kernel, kernel), stride: (stride.h, stride.w), padding: (padding, padding), bias: false),
            BatchNorm2d(outCh),
            ReLU(inplace: true));
    }

    private static Module<Tensor, Tensor> MakeConvBn(int inCh, int outCh, int kernel, (int h, int w) stride, int padding)
    {
        return Sequential(
            Conv2d(inCh, outCh, (kernel, kernel), stride: (stride.h, stride.w), padding: (padding, padding), bias: false),
            BatchNorm2d(outCh));
    }

    /// <summary>
    /// vd shortcut：如果非 first 且 stride != 1，先 AvgPool 再 1x1 Conv。
    /// </summary>
    private static Module<Tensor, Tensor> MakeVdShortcut(int inCh, int outCh, (int h, int w) stride, bool ifFirst)
    {
        if (!ifFirst && (stride.h != 1 || stride.w != 1))
        {
            // AvgPool + Conv1x1
            return Sequential(
                AvgPool2d(kernel_size: (stride.h, stride.w), stride: (stride.h, stride.w), ceil_mode: true),
                Conv2d(inCh, outCh, (1, 1), stride: (1, 1), bias: false),
                BatchNorm2d(outCh));
        }
        else
        {
            return Sequential(
                Conv2d(inCh, outCh, (1, 1), stride: (stride.h, stride.w), bias: false),
                BatchNorm2d(outCh));
        }
    }

    /// <summary>
    /// BasicBlock: 2 层 3x3 conv (layers < 50)
    /// </summary>
    private sealed class BasicBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _conv0;
        private readonly Module<Tensor, Tensor> _conv1;
        private readonly Module<Tensor, Tensor>? _short;
        private readonly bool _useShortcut;

        public BasicBlock(int inCh, int outCh, (int h, int w) stride, bool shortcut, bool ifFirst)
            : base(nameof(BasicBlock))
        {
            _useShortcut = shortcut;
            _conv0 = MakeConvBnRelu(inCh, outCh, 3, stride, 1);
            _conv1 = MakeConvBn(outCh, outCh, 3, (1, 1), 1);
            _short = shortcut ? null : MakeVdShortcut(inCh, outCh, stride, ifFirst);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var y = _conv0.call(input);
            using var conv1 = _conv1.call(y);
            var shortVal = _useShortcut ? input : _short!.call(input);
            using var sum = conv1 + shortVal;
            if (!_useShortcut) shortVal.Dispose();
            return functional.relu(sum);
        }
    }

    /// <summary>
    /// BottleneckBlock: 1x1 + 3x3 + 1x1 (layers >= 50)
    /// </summary>
    private sealed class BottleneckBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _conv0;
        private readonly Module<Tensor, Tensor> _conv1;
        private readonly Module<Tensor, Tensor> _conv2;
        private readonly Module<Tensor, Tensor>? _short;
        private readonly bool _useShortcut;

        public BottleneckBlock(int inCh, int outCh, (int h, int w) stride, bool shortcut, bool ifFirst)
            : base(nameof(BottleneckBlock))
        {
            _useShortcut = shortcut;
            _conv0 = MakeConvBnRelu(inCh, outCh, 1, (1, 1), 0);
            _conv1 = MakeConvBnRelu(outCh, outCh, 3, stride, 1);
            _conv2 = MakeConvBn(outCh, outCh * 4, 1, (1, 1), 0);
            _short = shortcut ? null : MakeVdShortcut(inCh, outCh * 4, stride, ifFirst);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var y0 = _conv0.call(input);
            using var y1 = _conv1.call(y0);
            using var y2 = _conv2.call(y1);
            var shortVal = _useShortcut ? input : _short!.call(input);
            using var sum = y2 + shortVal;
            if (!_useShortcut) shortVal.Dispose();
            return functional.relu(sum);
        }
    }
}
