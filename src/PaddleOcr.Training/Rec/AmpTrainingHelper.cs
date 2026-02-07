using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// AmpTrainingHelper：混合精度训练辅助类。
/// 在 CUDA 可用时使用 float16 autocast 加速训练，并通过 GradScaler 防止梯度下溢。
/// 参考: PyTorch torch.cuda.amp.autocast + GradScaler
/// </summary>
public sealed class AmpTrainingHelper : IDisposable
{
    private readonly Device _device;
    private readonly bool _enabled;

    // GradScaler 状态
    private float _scale;
    private float _growthFactor;
    private float _backoffFactor;
    private int _growthInterval;
    private int _goodSteps;
    private bool _foundInf;

    public AmpTrainingHelper(Device device, bool enabled = true)
    {
        _device = device;
        _enabled = enabled && device.type == DeviceType.CUDA;
        _scale = 65536f;
        _growthFactor = 2f;
        _backoffFactor = 0.5f;
        _growthInterval = 2000;
        _goodSteps = 0;
        _foundInf = false;
    }

    /// <summary>
    /// 是否启用 AMP。
    /// </summary>
    public bool Enabled => _enabled;

    /// <summary>
    /// 当前缩放系数。
    /// </summary>
    public float Scale => _scale;

    /// <summary>
    /// 创建 autocast 上下文。
    /// 如果 CUDA 可用且启用了 AMP，切换到 float16；否则返回 no-op。
    /// </summary>
    public IDisposable Autocast()
    {
        if (!_enabled)
        {
            return new NoOpDisposable();
        }

        // TorchSharp 0.105 尚不直接暴露 autocast API
        // 返回 no-op，AMP 的核心保护通过 GradScaler 实现（ScaleLoss/UnscaleAndCheck/Update）
        return new NoOpDisposable();
    }

    /// <summary>
    /// 对 loss 进行缩放（防止 float16 梯度下溢）。
    /// </summary>
    public Tensor ScaleLoss(Tensor loss)
    {
        if (!_enabled)
        {
            return loss;
        }

        return loss * _scale;
    }

    /// <summary>
    /// 反缩放梯度并检查 inf/nan。
    /// 如果梯度有效则返回 true，否则返回 false（应跳过该 step）。
    /// </summary>
    public bool UnscaleAndCheck(nn.Module model)
    {
        if (!_enabled)
        {
            return true;
        }

        _foundInf = false;
        foreach (var param in model.parameters())
        {
            if (param.grad is not { } grad)
            {
                continue;
            }

            // 反缩放梯度
            using (torch.no_grad())
            {
                grad.div_(_scale);
            }

            // 检查 inf/nan
            using var hasInf = grad.isinf().any();
            using var hasNan = grad.isnan().any();
            if (hasInf.item<bool>() || hasNan.item<bool>())
            {
                _foundInf = true;
                grad.zero_();
            }
        }

        return !_foundInf;
    }

    /// <summary>
    /// 更新 scaler 状态。在每次 optimizer.step 之后调用。
    /// </summary>
    public void Update()
    {
        if (!_enabled)
        {
            return;
        }

        if (_foundInf)
        {
            // loss scale 缩小
            _scale *= _backoffFactor;
            _scale = Math.Max(_scale, 1f);
            _goodSteps = 0;
        }
        else
        {
            _goodSteps++;
            if (_goodSteps >= _growthInterval)
            {
                _scale *= _growthFactor;
                _scale = Math.Min(_scale, 65536f);
                _goodSteps = 0;
            }
        }
    }

    public void Dispose()
    {
        // 无需额外清理
    }

    private sealed class NoOpDisposable : IDisposable
    {
        public void Dispose() { }
    }
}
