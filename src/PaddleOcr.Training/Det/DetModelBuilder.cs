using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using PaddleOcr.Training.Det.Backbones;
using PaddleOcr.Training.Det.Necks;
using PaddleOcr.Training.Det.Heads;

namespace PaddleOcr.Training.Det;

/// <summary>
/// Det 模型构建器：从配置构建完整的 Backbone + Neck + Head 模型。
/// </summary>
public static class DetModelBuilder
{
    /// <summary>
    /// 构建 backbone。
    /// </summary>
    public static (Module<Tensor, Tensor[]> Module, int[] OutChannels) BuildBackbone(
        string name, int inChannels = 3, float scale = 0.5f, string modelName = "large")
    {
        return name.ToLowerInvariant() switch
        {
            "mobilenetv3" or "mobilenet_v3" => BuildDetMobileNetV3(inChannels, modelName, scale),
            "mobilenetv3_large" => BuildDetMobileNetV3(inChannels, "large", scale),
            "mobilenetv3_small" => BuildDetMobileNetV3(inChannels, "small", scale),
            "pplcnetv3" or "pplcnet_v3" or "pplcnetv3_det" => BuildDetPPLCNetV3(inChannels, scale),
            "resnet_vd" or "resnetvd" or "resnet18_vd" => BuildDetResNetVd(inChannels, 18),
            "resnet34_vd" => BuildDetResNetVd(inChannels, 34),
            "resnet50_vd" => BuildDetResNetVd(inChannels, 50),
            _ => BuildDetMobileNetV3(inChannels, modelName, scale)
        };
    }

    /// <summary>
    /// 构建 neck。
    /// </summary>
    public static (Module<Tensor[], Tensor> Module, int OutChannels) BuildNeck(
        string name, int[] inChannels, int outChannels = 256)
    {
        return name.ToLowerInvariant() switch
        {
            "dbfpn" or "db_fpn" => BuildDBFPN(inChannels, outChannels),
            "rsefpn" or "rse_fpn" => BuildRSEFPN(inChannels, outChannels),
            _ => BuildDBFPN(inChannels, outChannels)
        };
    }

    /// <summary>
    /// 构建 head。
    /// </summary>
    public static Module<Tensor, Dictionary<string, Tensor>> BuildHead(
        string name, int inChannels, int k = 50)
    {
        return name.ToLowerInvariant() switch
        {
            "dbhead" or "db" => new DBHead(inChannels, k),
            _ => new DBHead(inChannels, k)
        };
    }

    /// <summary>
    /// 从配置构建完整的 Det 模型 (Backbone + Neck + Head)。
    /// </summary>
    public static DetModel Build(
        string backboneName,
        string neckName,
        string headName,
        int inChannels = 3,
        float scale = 0.5f,
        string modelName = "large",
        int neckOutChannels = 256,
        int k = 50)
    {
        var (backbone, backboneOutChs) = BuildBackbone(backboneName, inChannels, scale, modelName);
        var (neck, neckOutCh) = BuildNeck(neckName, backboneOutChs, neckOutChannels);
        var head = BuildHead(headName, neckOutCh, k);

        return new DetModel(backbone, neck, head, backboneName, neckName, headName);
    }

    private static (Module<Tensor, Tensor[]>, int[]) BuildDetMobileNetV3(
        int inChannels, string modelName, float scale)
    {
        var m = new DetMobileNetV3(inChannels, modelName, scale);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor[]>, int[]) BuildDetResNetVd(
        int inChannels, int layers)
    {
        var m = new DetResNetVd(inChannels, layers);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor, Tensor[]>, int[]) BuildDetPPLCNetV3(
        int inChannels, float scale)
    {
        var m = new DetPPLCNetV3(inChannels, scale);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor[], Tensor>, int) BuildDBFPN(int[] inChannels, int outChannels)
    {
        var m = new DBFPN(inChannels, outChannels);
        return (m, m.OutChannels);
    }

    private static (Module<Tensor[], Tensor>, int) BuildRSEFPN(int[] inChannels, int outChannels)
    {
        var m = new RSEFPN(inChannels, outChannels);
        return (m, m.OutChannels);
    }
}
