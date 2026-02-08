using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Det;

/// <summary>
/// Det Backbone 组件接口。
/// 输出多尺度特征图列表 (c2, c3, c4, c5)。
/// </summary>
public interface IDetBackbone
{
    /// <summary>
    /// 各阶段输出通道数列表。
    /// </summary>
    int[] OutChannels { get; }
}

/// <summary>
/// Det Neck 组件接口 (如 DBFPN)。
/// 接收多尺度特征列表，输出融合特征。
/// </summary>
public interface IDetNeck
{
    /// <summary>
    /// 输出通道数。
    /// </summary>
    int OutChannels { get; }
}

/// <summary>
/// Det Head 组件接口 (如 DBHead)。
/// </summary>
public interface IDetHead
{
    /// <summary>
    /// 前向传播。
    /// </summary>
    Dictionary<string, Tensor> Forward(Tensor input, bool training);
}

/// <summary>
/// Det 损失函数接口。
/// </summary>
public interface IDetLoss
{
    /// <summary>
    /// 计算损失。
    /// </summary>
    Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch);
}
