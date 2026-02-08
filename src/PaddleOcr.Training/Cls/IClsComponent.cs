using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Cls;

/// <summary>
/// Cls Backbone 组件接口（单输出形式）。
/// </summary>
public interface IClsBackbone
{
    /// <summary>
    /// 输出通道数。
    /// </summary>
    int OutChannels { get; }
}

/// <summary>
/// Cls Head 组件接口。
/// </summary>
public interface IClsHead
{
    /// <summary>
    /// 前向传播，返回分类 logits。
    /// </summary>
    Tensor Forward(Tensor input);
}

/// <summary>
/// Cls 损失函数接口。
/// </summary>
public interface IClsLoss
{
    /// <summary>
    /// 计算分类损失。
    /// </summary>
    Dictionary<string, Tensor> Forward(Tensor predictions, Tensor labels);
}
