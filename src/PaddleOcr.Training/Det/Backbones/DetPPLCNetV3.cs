using PaddleOcr.Training.Rec.Backbones;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det.Backbones;

/// <summary>
/// PPLCNetV3 backbone for detection (det mode).
/// 1:1 port of ppocr/modeling/backbones/rec_lcnetv3.py PPLCNetV3(det=True).
///
/// Architecture: conv1 -> blocks2 -> blocks3 -> blocks4 -> blocks5 -> blocks6
/// Outputs 4 multi-scale feature maps [blocks3, blocks4, blocks5, blocks6]
/// with 1x1 projection convolutions (mv_c = [16, 24, 56, 480] * scale).
///
/// All strides in det mode are symmetric (int, not tuple).
/// </summary>
public sealed class DetPPLCNetV3 : Module<Tensor, Tensor[]>, IDetBackbone
{
    // NET_CONFIG_det: k, in_c, out_c, stride, use_se
    private static readonly (int K, int InC, int OutC, int S, bool UseSe)[][] NetConfigDet =
    [
        // blocks2
        [(3, 16, 32, 1, false)],
        // blocks3
        [(3, 32, 64, 2, false), (3, 64, 64, 1, false)],
        // blocks4
        [(3, 64, 128, 2, false), (3, 128, 128, 1, false)],
        // blocks5
        [(3, 128, 256, 2, false), (5, 256, 256, 1, false), (5, 256, 256, 1, false), (5, 256, 256, 1, false), (5, 256, 256, 1, false)],
        // blocks6
        [(5, 256, 512, 2, true), (5, 512, 512, 1, true), (5, 512, 512, 1, false), (5, 512, 512, 1, false)],
    ];

    private readonly LCNetV3ConvBNLayer _conv1;
    private readonly Sequential _blocks2;
    private readonly Sequential _blocks3;
    private readonly Sequential _blocks4;
    private readonly Sequential _blocks5;
    private readonly Sequential _blocks6;

    // 1x1 projection convolutions for each output stage
    private readonly Conv2d _proj3;
    private readonly Conv2d _proj4;
    private readonly Conv2d _proj5;
    private readonly Conv2d _proj6;

    public int[] OutChannels { get; }

    public DetPPLCNetV3(
        int inChannels = 3,
        float scale = 0.75f,
        int convKxkNum = 4,
        float[]? lrMultList = null,
        float labLr = 0.1f) : base(nameof(DetPPLCNetV3))
    {
        lrMultList ??= [1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f];
        if (lrMultList.Length != 6)
            throw new ArgumentException($"lrMultList length should be 6 but got {lrMultList.Length}");

        _conv1 = new LCNetV3ConvBNLayer(
            inChannels, PPLCNetV3.MakeDivisible((int)(16 * scale)), 3, (2, 2));

        _blocks2 = BuildBlockGroup(NetConfigDet[0], scale, convKxkNum, lrMultList[1], labLr);
        _blocks3 = BuildBlockGroup(NetConfigDet[1], scale, convKxkNum, lrMultList[2], labLr);
        _blocks4 = BuildBlockGroup(NetConfigDet[2], scale, convKxkNum, lrMultList[3], labLr);
        _blocks5 = BuildBlockGroup(NetConfigDet[3], scale, convKxkNum, lrMultList[4], labLr);
        _blocks6 = BuildBlockGroup(NetConfigDet[4], scale, convKxkNum, lrMultList[5], labLr);

        // Compute backbone output channels before projection
        var blocks3OutCh = PPLCNetV3.MakeDivisible((int)(NetConfigDet[1][^1].OutC * scale));
        var blocks4OutCh = PPLCNetV3.MakeDivisible((int)(NetConfigDet[2][^1].OutC * scale));
        var blocks5OutCh = PPLCNetV3.MakeDivisible((int)(NetConfigDet[3][^1].OutC * scale));
        var blocks6OutCh = PPLCNetV3.MakeDivisible((int)(NetConfigDet[4][^1].OutC * scale));

        // mv_c projection target channels
        int[] mvC = [16, 24, 56, 480];
        var proj3Ch = (int)(mvC[0] * scale);
        var proj4Ch = (int)(mvC[1] * scale);
        var proj5Ch = (int)(mvC[2] * scale);
        var proj6Ch = (int)(mvC[3] * scale);

        // 1x1 projection convolutions (padding=0 is default for kernel=1)
        _proj3 = nn.Conv2d(blocks3OutCh, proj3Ch, 1, stride: 1L);
        _proj4 = nn.Conv2d(blocks4OutCh, proj4Ch, 1, stride: 1L);
        _proj5 = nn.Conv2d(blocks5OutCh, proj5Ch, 1, stride: 1L);
        _proj6 = nn.Conv2d(blocks6OutCh, proj6Ch, 1, stride: 1L);

        OutChannels = [proj3Ch, proj4Ch, proj5Ch, proj6Ch];

        RegisterComponents();
    }

    public override Tensor[] forward(Tensor input)
    {
        var x = _conv1.call(input);
        x = _blocks2.call(x);

        x = _blocks3.call(x);
        var out3 = x;

        x = _blocks4.call(x);
        var out4 = x;

        x = _blocks5.call(x);
        var out5 = x;

        x = _blocks6.call(x);
        var out6 = x;

        // Apply 1x1 projections
        return
        [
            _proj3.call(out3),
            _proj4.call(out4),
            _proj5.call(out5),
            _proj6.call(out6),
        ];
    }

    private static Sequential BuildBlockGroup(
        (int K, int InC, int OutC, int S, bool UseSe)[] blockConfigs,
        float scale, int convKxkNum, float lrMult, float labLr)
    {
        var blocks = new List<Module<Tensor, Tensor>>();
        foreach (var (k, inC, outC, s, useSe) in blockConfigs)
        {
            blocks.Add(new LCNetV3Block(
                PPLCNetV3.MakeDivisible((int)(inC * scale)),
                PPLCNetV3.MakeDivisible((int)(outC * scale)),
                dwSize: k,
                stride: (s, s), // det mode uses symmetric strides
                useSe: useSe,
                convKxkNum: convKxkNum,
                lrMult: lrMult,
                labLr: labLr));
        }
        return Sequential(blocks);
    }
}
