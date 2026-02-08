using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Det;

/// <summary>
/// Det 完整模型：Backbone (多尺度) + Neck (FPN) + Head (DB)。
/// </summary>
public sealed class DetModel : Module<Tensor, Dictionary<string, Tensor>>
{
    private readonly Module<Tensor, Tensor[]> _backbone;
    private readonly Module<Tensor[], Tensor> _neck;
    private readonly Module<Tensor, Dictionary<string, Tensor>> _head;

    public string BackboneName { get; }
    public string NeckName { get; }
    public string HeadName { get; }

    public DetModel(
        Module<Tensor, Tensor[]> backbone,
        Module<Tensor[], Tensor> neck,
        Module<Tensor, Dictionary<string, Tensor>> head,
        string backboneName,
        string neckName,
        string headName) : base(nameof(DetModel))
    {
        _backbone = backbone;
        _neck = neck;
        _head = head;
        BackboneName = backboneName;
        NeckName = neckName;
        HeadName = headName;
        RegisterComponents();
    }

    public override Dictionary<string, Tensor> forward(Tensor x)
    {
        return ForwardDict(x, training: true);
    }

    public Dictionary<string, Tensor> ForwardDict(Tensor x, bool training)
    {
        var features = _backbone.call(x);
        var fuse = _neck.call(features);

        if (_head is Heads.DBHead dbHead)
        {
            return dbHead.Forward(fuse, training);
        }

        return _head.call(fuse);
    }
}
