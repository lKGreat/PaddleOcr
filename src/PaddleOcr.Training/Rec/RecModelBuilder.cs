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
            "resnet32" => BuildResNet32(inChannels),
            "resnet45" => BuildResNet45(inChannels),
            "svtrnet" or "svtr" => BuildSVTRNet(inChannels),
            "vitstr" => BuildViTSTR(inChannels),
            "mtb" => BuildMTB(inChannels),
            "densenet" => BuildDenseNet(inChannels),
            "efficientnetb3" or "efficientnet_b3" => BuildEfficientNetB3(inChannels),
            "shallowcnn" => BuildShallowCNN(inChannels),
            "pplcnetsmall" or "pplcnetsmall_v3" => BuildPPLCNetV3(inChannels),
            "micronet" => BuildMicroNet(inChannels),
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
        string name, int inChannels, int outChannels, int hiddenSize = 48, int maxLen = 25)
    {
        return name.ToLowerInvariant() switch
        {
            "ctc" or "ctchead" => new CTCHead(inChannels, outChannels),
            "attn" or "attention" or "attentionhead" => new AttentionHead(inChannels, outChannels, hiddenSize, maxLen),
            "multi" or "multihead" => new MultiHead(inChannels, outChannels, outChannels, hiddenSize),
            "sar" or "sarhead" => new SARHead(inChannels, outChannels, hiddenSize, maxLen),
            "nrtr" or "nrtrhead" => new NRTRHead(inChannels, outChannels, hiddenSize, maxLen: maxLen),
            "srn" or "srnhead" => new SRNHead(inChannels, outChannels, hiddenSize, maxLen),
            "satrn" or "satrnhead" => new SATRNHead(inChannels, outChannels, hiddenSize, maxLen),
            "robustscanner" or "robustscannerhead" => new RobustScannerHead(inChannels, outChannels, hiddenSize, maxLen),
            "spin" or "spinattentionhead" => new SPINAttentionHead(inChannels, outChannels, hiddenSize, maxLen),
            "abinet" or "abinethead" => new ABINetHead(inChannels, outChannels, hiddenSize),
            "vl" or "vlhead" => new VLHead(inChannels, outChannels, maxLen),
            "rfl" or "rflhead" => new RFLHead(inChannels, outChannels, maxLen),
            "can" or "canhead" => new CANHead(inChannels, outChannels),
            "latexocr" or "latexocrhead" => new LaTeXOCRHead(inChannels, outChannels, hiddenSize, maxLen),
            "parseq" or "parseqhead" => new ParseQHead(inChannels, outChannels, maxLen),
            "cppd" or "cppdhead" => new CPPDHead(inChannels, outChannels, hiddenSize, maxLen),
            "pren" or "prenhead" => new PRENHead(inChannels, outChannels),
            "unimernet" or "unimernethead" => new UniMERNetHead(inChannels, outChannels, hiddenSize, maxLen),
            "ppformulanet" or "ppformulanethead" => new PPFormulaNetHead(inChannels, outChannels, hiddenSize, maxLen),
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
        int hiddenSize = 48,
        int maxLen = 25)
    {
        var (backbone, backboneOutCh) = BuildBackbone(backboneName, inChannels);
        var (neck, neckOutCh) = BuildNeck(neckName, backboneOutCh, hiddenSize);
        var head = BuildHead(headName, neckOutCh, numClasses, hiddenSize, maxLen);
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

    private static (Module<Tensor, Tensor>, int) BuildResNet32(int inChannels)
    {
        var m = new ResNet32(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildResNet45(int inChannels)
    {
        var m = new ResNet45(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildViTSTR(int inChannels)
    {
        var m = new ViTSTR(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildMTB(int inChannels)
    {
        var m = new MTB(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildDenseNet(int inChannels)
    {
        var m = new DenseNet(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildEfficientNetB3(int inChannels)
    {
        var m = new EfficientNetB3(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildShallowCNN(int inChannels)
    {
        var m = new ShallowCNN(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildPPLCNetV3(int inChannels)
    {
        var m = new PPLCNetV3(inChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor>, int) BuildMicroNet(int inChannels)
    {
        var m = new MicroNet(inChannels);
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
