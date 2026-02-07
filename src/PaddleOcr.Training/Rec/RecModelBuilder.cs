using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using PaddleOcr.Training.Rec.Backbones;
using PaddleOcr.Training.Rec.Necks;
using PaddleOcr.Training.Rec.Heads;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// Rec 模型构建器：从配置构建完整的 Backbone + Neck + Head 模型。
/// </summary>
public static class RecModelBuilder
{
    /// <summary>
    /// 构建 backbone。
    /// </summary>
    public static (Module<Tensor, Tensor> Module, int OutChannels) BuildBackbone(
        string name, int inChannels = 3)
    {
        return name.ToLowerInvariant() switch
        {
            "mobilenetv1enhance" or "mobilenetv1_enhance" => BuildMobileNetV1Enhance(inChannels),
            "pphgnet_small" or "pphgnetsmall" => BuildPPHGNetSmall(inChannels),
            "resnet31" => BuildResNet31(inChannels),
            "svtrnet" or "svtr" => BuildSVTRNet(inChannels),
            _ => BuildMobileNetV1Enhance(inChannels)
        };
    }

    /// <summary>
    /// 构建 neck。
    /// </summary>
    public static (Module<Tensor, Tensor> Module, int OutChannels) BuildNeck(
        string name, int inChannels, int hiddenSize = 48)
    {
        var encoder = new SequenceEncoder(inChannels, name.ToLowerInvariant(), hiddenSize);
        return (encoder, encoder.OutChannels);
    }

    /// <summary>
    /// 构建 head。
    /// </summary>
    public static Module<Tensor, Tensor> BuildHead(
        string name, int inChannels, int outChannels, int hiddenSize = 48)
    {
        return name.ToLowerInvariant() switch
        {
            "ctc" or "ctchead" => new CTCHead(inChannels, outChannels),
            "attn" or "attention" or "attentionhead" => new AttentionHead(inChannels, outChannels, hiddenSize),
            "multi" or "multihead" => new MultiHead(inChannels, outChannels, outChannels, hiddenSize),
            _ => new CTCHead(inChannels, outChannels)
        };
    }

    /// <summary>
    /// 从配置构建完整的 Rec 模型（Backbone + Neck + Head）。
    /// </summary>
    public static RecModel Build(
        string backboneName,
        string neckName,
        string headName,
        int numClasses,
        int inChannels = 3,
        int hiddenSize = 48)
    {
        var (backbone, backboneOutCh) = BuildBackbone(backboneName, inChannels);
        var (neck, neckOutCh) = BuildNeck(neckName, backboneOutCh, hiddenSize);
        var head = BuildHead(headName, neckOutCh, numClasses, hiddenSize);
        return new RecModel(backbone, neck, head, backboneName, neckName, headName);
    }

    private static (Module<Tensor, Tensor>, int) BuildMobileNetV1Enhance(int inChannels)
    {
        var m = new MobileNetV1Enhance(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildPPHGNetSmall(int inChannels)
    {
        var m = new PPHGNetSmall(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildResNet31(int inChannels)
    {
        var m = new ResNet31(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildSVTRNet(int inChannels)
    {
        var m = new SVTRNet(inChannels);
        return (m, m.OutChannels);
    }
}

/// <summary>
/// 完整的 Rec 模型：Backbone + Neck + Head。
/// </summary>
public sealed class RecModel : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _neck;
    private readonly Module<Tensor, Tensor> _head;
    public string BackboneName { get; }
    public string NeckName { get; }
    public string HeadName { get; }

    public RecModel(
        Module<Tensor, Tensor> backbone,
        Module<Tensor, Tensor> neck,
        Module<Tensor, Tensor> head,
        string backboneName,
        string neckName,
        string headName) : base(nameof(RecModel))
    {
        _backbone = backbone;
        _neck = neck;
        _head = head;
        BackboneName = backboneName;
        NeckName = neckName;
        HeadName = headName;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var feat = _backbone.call(input);
        using var seq = _neck.call(feat);
        return _head.call(seq);
    }

    /// <summary>
    /// 支持 IRecHead 接口的前向传播，返回预测字典。
    /// </summary>
    public Dictionary<string, Tensor> ForwardDict(Tensor input, Dictionary<string, Tensor>? targets = null)
    {
        using var feat = _backbone.call(input);
        using var seq = _neck.call(feat);
        if (_head is IRecHead recHead)
        {
            return recHead.Forward(seq, targets);
        }

        var logits = _head.call(seq);
        return new Dictionary<string, Tensor> { ["predict"] = logits };
    }
}
