using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// ModelAverager：模型参数平均器，用于 SRN 等模型。
/// </summary>
public sealed class ModelAverager
{
    private readonly Dictionary<string, Tensor> _averagedParams;
    private int _count;

    public ModelAverager()
    {
        _averagedParams = new Dictionary<string, Tensor>();
        _count = 0;
    }

    /// <summary>
    /// 更新平均参数。
    /// </summary>
    public void Update(Module<Tensor, Tensor> model)
    {
        _count++;
        var stateDict = model.state_dict();

        foreach (var (name, param) in stateDict)
        {
            if (!_averagedParams.ContainsKey(name))
            {
                _averagedParams[name] = param.clone();
            }
            else
            {
                // 累积平均
                var avg = _averagedParams[name];
                _averagedParams[name] = (avg * (_count - 1) + param) / _count;
            }
        }
    }

    /// <summary>
    /// 将平均参数应用到模型。
    /// </summary>
    public void Apply(Module<Tensor, Tensor> model)
    {
        var stateDict = model.state_dict();
        foreach (var (name, avgParam) in _averagedParams)
        {
            if (stateDict.ContainsKey(name))
            {
                stateDict[name].copy_(avgParam);
            }
        }
    }

    /// <summary>
    /// 重置平均器。
    /// </summary>
    public void Reset()
    {
        _averagedParams.Clear();
        _count = 0;
    }
}
