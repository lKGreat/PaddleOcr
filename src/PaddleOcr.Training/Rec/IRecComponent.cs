using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// Rec Backbone 组件接口。
/// </summary>
public interface IRecBackbone
{
    /// <summary>
    /// 输出通道数，供下游 Neck/Head 使用。
    /// </summary>
    int OutChannels { get; }

    /// <summary>
    /// 前向传播。
    /// </summary>
    Tensor Forward(Tensor input);
}

/// <summary>
/// Rec Neck 组件接口。
/// </summary>
public interface IRecNeck
{
    /// <summary>
    /// 输出通道数，供下游 Head 使用。
    /// </summary>
    int OutChannels { get; }

    /// <summary>
    /// 前向传播。
    /// </summary>
    Tensor Forward(Tensor input);
}

/// <summary>
/// Rec Head 组件接口。
/// </summary>
public interface IRecHead
{
    /// <summary>
    /// 前向传播。
    /// </summary>
    /// <param name="input">来自 Neck 或 Backbone 的特征</param>
    /// <param name="targets">训练时的 label 信息（推理时为 null）</param>
    /// <returns>预测结果字典，通常包含 "predict" 键</returns>
    Dictionary<string, Tensor> Forward(Tensor input, Dictionary<string, Tensor>? targets = null);
}

/// <summary>
/// Rec 损失函数接口。
/// </summary>
public interface IRecLoss
{
    /// <summary>
    /// 计算损失。
    /// </summary>
    /// <param name="predictions">模型预测输出</param>
    /// <param name="batch">标签 batch 信息</param>
    /// <returns>损失值字典，包含 "loss" 键和可能的子损失</returns>
    Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> predictions, Dictionary<string, Tensor> batch);
}

/// <summary>
/// 学习率调度器接口。
/// </summary>
public interface ILRScheduler
{
    /// <summary>
    /// 获取当前学习率。
    /// </summary>
    double CurrentLR { get; }

    /// <summary>
    /// 根据当前步骤/epoch 更新学习率。
    /// </summary>
    /// <param name="step">当前全局步骤</param>
    /// <param name="epoch">当前 epoch</param>
    void Step(int step, int epoch);
}
