using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec.Transforms;

/// <summary>
/// STN_ON: Spatial Transformer Network + TPS (Thin Plate Spline)。
/// 用于 SVTR 配置中的 Transform。
/// 参考: ppocr/modeling/transforms/stn.py + tps_spatial_transformer.py
/// </summary>
public sealed class STN_ON : Module<Tensor, Tensor>
{
    private readonly STNHead _stnHead;
    private readonly TPSSpatialTransformer _tps;
    private readonly int[] _tpsInputSize;

    public int OutChannels { get; }

    public STN_ON(
        int inChannels,
        int[] tpsInputSize,
        int[] tpsOutputSize,
        int numControlPoints = 20,
        float[] tpsMargins = null!,
        string stnActivation = "none")
        : base(nameof(STN_ON))
    {
        tpsMargins ??= [0.05f, 0.05f];
        _tpsInputSize = tpsInputSize;

        _stnHead = new STNHead(inChannels, numControlPoints, stnActivation);
        _tps = new TPSSpatialTransformer(tpsOutputSize, numControlPoints, tpsMargins);

        OutChannels = inChannels;
        RegisterComponents();
    }

    public override Tensor forward(Tensor image)
    {
        // 插值到 STN 输入大小
        using var stnInput = functional.interpolate(image, size: [_tpsInputSize[0], _tpsInputSize[1]],
            mode: InterpolationMode.Bilinear, align_corners: true);
        var (_, ctrlPoints) = _stnHead.Forward(stnInput);
        using (ctrlPoints)
        {
            var (output, _) = _tps.Forward(image, ctrlPoints);
            return output;
        }
    }

    /// <summary>
    /// STN Head: CNN 网络预测 TPS 控制点。
    /// </summary>
    private sealed class STNHead : Module<Tensor, Tensor>
    {
        private readonly Sequential _convnet;
        private readonly Sequential _fc1;
        private readonly Linear _fc2;
        private readonly int _numCtrlPoints;
        private readonly string _activation;

        public STNHead(int inChannels, int numCtrlPoints, string activation) : base(nameof(STNHead))
        {
            _numCtrlPoints = numCtrlPoints;
            _activation = activation;

            _convnet = Sequential(
                MakeConv3x3Block(inChannels, 32),
                MaxPool2d(2, 2),
                MakeConv3x3Block(32, 64),
                MaxPool2d(2, 2),
                MakeConv3x3Block(64, 128),
                MaxPool2d(2, 2),
                MakeConv3x3Block(128, 256),
                MaxPool2d(2, 2),
                MakeConv3x3Block(256, 256),
                MaxPool2d(2, 2),
                MakeConv3x3Block(256, 256));

            _fc1 = Sequential(
                Linear(2 * 256, 512),
                BatchNorm1d(512),
                ReLU(inplace: true));

            // 初始化 fc2 bias 为均匀分布的控制点
            _fc2 = Linear(512, numCtrlPoints * 2);
            InitStn();

            RegisterComponents();
        }

        private void InitStn()
        {
            using var noGrad = torch.no_grad();
            var margin = 0.01f;
            var samplingNumPerSide = _numCtrlPoints / 2;
            var ctrlPts = new float[_numCtrlPoints * 2];

            for (var i = 0; i < samplingNumPerSide; i++)
            {
                var x = margin + i * (1f - 2 * margin) / (samplingNumPerSide - 1);
                // top row
                ctrlPts[i * 2] = x;
                ctrlPts[i * 2 + 1] = margin;
                // bottom row
                ctrlPts[(samplingNumPerSide + i) * 2] = x;
                ctrlPts[(samplingNumPerSide + i) * 2 + 1] = 1f - margin;
            }

            if (_activation == "sigmoid")
            {
                for (var i = 0; i < ctrlPts.Length; i++)
                {
                    ctrlPts[i] = -MathF.Log(1f / ctrlPts[i] - 1f);
                }
            }

            // 设置 fc2 weights=0, bias=ctrl_pts
            init.zeros_(_fc2.weight);
            using var biasTensor = tensor(ctrlPts, ScalarType.Float32);
            _fc2.bias!.copy_(biasTensor);
        }

        public (Tensor ImgFeat, Tensor CtrlPoints) Forward(Tensor x)
        {
            x = _convnet.call(x);
            var batchSize = x.shape[0];
            using var flat = x.reshape(batchSize, -1);
            var imgFeat = _fc1.call(flat);
            using var scaled = 0.1f * imgFeat;
            var pts = _fc2.call(scaled);
            if (_activation == "sigmoid")
            {
                pts = functional.sigmoid(pts);
            }
            var ctrlPoints = pts.reshape(-1, _numCtrlPoints, 2);
            return (imgFeat, ctrlPoints);
        }

        public override Tensor forward(Tensor x)
        {
            var (_, ctrlPoints) = Forward(x);
            return ctrlPoints;
        }

        private static Sequential MakeConv3x3Block(int inCh, int outCh)
        {
            return Sequential(
                Conv2d(inCh, outCh, 3, stride: 1, padding: 1),
                BatchNorm2d(outCh),
                ReLU(inplace: true));
        }
    }

    /// <summary>
    /// TPS Spatial Transformer: 用 TPS 变换对输入图像进行空间变形。
    /// </summary>
    private sealed class TPSSpatialTransformer : Module<Tensor, Tensor>
    {
        private readonly int _targetH;
        private readonly int _targetW;
        private readonly int _numControlPoints;
        private Tensor _inverseKernel;
        private Tensor _targetCoordinateRepr;
        private Tensor _paddingMatrix;

        public TPSSpatialTransformer(int[] outputImageSize, int numControlPoints, float[] margins)
            : base(nameof(TPSSpatialTransformer))
        {
            _targetH = outputImageSize[0];
            _targetW = outputImageSize[1];
            _numControlPoints = numControlPoints;

            // 构建目标控制点
            var targetCtrlPts = BuildOutputControlPoints(numControlPoints, margins[0], margins[1]);

            var n = numControlPoints;

            // 构建 forward kernel
            var forwardKernel = new float[(n + 3) * (n + 3)];
            var partialRepr = ComputePartialRepr(targetCtrlPts, targetCtrlPts, n, n);

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    forwardKernel[i * (n + 3) + j] = partialRepr[i * n + j];
                }
                forwardKernel[i * (n + 3) + n] = 1;
                forwardKernel[n * (n + 3) + i] = 1;
                forwardKernel[i * (n + 3) + n + 1] = targetCtrlPts[i * 2];
                forwardKernel[i * (n + 3) + n + 2] = targetCtrlPts[i * 2 + 1];
                forwardKernel[(n + 1) * (n + 3) + i] = targetCtrlPts[i * 2];
                forwardKernel[(n + 2) * (n + 3) + i] = targetCtrlPts[i * 2 + 1];
            }

            using var fk = tensor(forwardKernel, ScalarType.Float32).reshape(n + 3, n + 3);
            _inverseKernel = fk.inverse();

            // 构建目标坐标矩阵
            var hw = _targetH * _targetW;
            var targetCoords = new float[hw * 2];
            for (var y = 0; y < _targetH; y++)
            {
                for (var x = 0; x < _targetW; x++)
                {
                    var idx = y * _targetW + x;
                    targetCoords[idx * 2] = (float)x / (_targetW - 1); // X
                    targetCoords[idx * 2 + 1] = (float)y / (_targetH - 1); // Y
                }
            }

            var targetCoordPartialRepr = ComputePartialRepr(targetCoords, targetCtrlPts, hw, n);

            // target_coordinate_repr = [partial_repr | ones | coords]
            var reprData = new float[hw * (n + 3)];
            for (var i = 0; i < hw; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    reprData[i * (n + 3) + j] = targetCoordPartialRepr[i * n + j];
                }
                reprData[i * (n + 3) + n] = 1f;
                reprData[i * (n + 3) + n + 1] = targetCoords[i * 2];
                reprData[i * (n + 3) + n + 2] = targetCoords[i * 2 + 1];
            }

            _targetCoordinateRepr = tensor(reprData, ScalarType.Float32).reshape(hw, n + 3);
            _paddingMatrix = zeros(3, 2);

            RegisterComponents();
        }

        public (Tensor Output, Tensor SourceCoordinate) Forward(Tensor input, Tensor sourceControlPoints)
        {
            var batchSize = sourceControlPoints.shape[0];

            using var padding = _paddingMatrix.expand(batchSize, 3, 2);
            using var Y = cat([sourceControlPoints.to(padding.dtype), padding], dim: 1);
            using var invKernel = _inverseKernel.to(Y.device);
            using var mapping = matmul(invKernel, Y);
            using var coordRepr = _targetCoordinateRepr.to(Y.device);
            using var sourceCoordinate = matmul(coordRepr, mapping);
            using var grid = sourceCoordinate.reshape(-1, _targetH, _targetW, 2);
            using var gridClamped = grid.clamp(0, 1);
            using var gridNorm = 2f * gridClamped - 1f;
            var output = functional.grid_sample(input, gridNorm, mode: GridSampleMode.Bilinear,
                align_corners: true);
            return (output, sourceCoordinate.clone());
        }

        public override Tensor forward(Tensor x) => x;

        private static float[] BuildOutputControlPoints(int numControlPoints, float marginX, float marginY)
        {
            var perSide = numControlPoints / 2;
            var pts = new float[numControlPoints * 2];
            for (var i = 0; i < perSide; i++)
            {
                var x = marginX + i * (1f - 2 * marginX) / (perSide - 1);
                pts[i * 2] = x;
                pts[i * 2 + 1] = marginY;
                pts[(perSide + i) * 2] = x;
                pts[(perSide + i) * 2 + 1] = 1f - marginY;
            }
            return pts;
        }

        /// <summary>
        /// phi(x1, x2) = r^2 * log(r), r = ||x1-x2||
        /// </summary>
        private static float[] ComputePartialRepr(float[] inputPts, float[] ctrlPts, int n, int m)
        {
            var repr = new float[n * m];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < m; j++)
                {
                    var dx = inputPts[i * 2] - ctrlPts[j * 2];
                    var dy = inputPts[i * 2 + 1] - ctrlPts[j * 2 + 1];
                    var dist = dx * dx + dy * dy;
                    repr[i * m + j] = dist > 0 ? 0.5f * dist * MathF.Log(dist) : 0f;
                }
            }
            return repr;
        }
    }
}
