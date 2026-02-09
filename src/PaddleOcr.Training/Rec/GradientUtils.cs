using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// GradientUtils：梯度裁剪、梯度累积等工具。
/// </summary>
public static class GradientUtils
{
    /// <summary>
    /// 按参数独立裁剪梯度范数（ClipGradByNorm）。
    /// 每个参数的梯度独立裁剪到 maxNorm。
    /// 参考: paddle.nn.ClipGradByNorm
    /// </summary>
    public static void ClipGradNorm(Module<Tensor, Tensor> model, float maxNorm)
    {
        var parameters = model.parameters();
        var totalNorm = 0.0;

        foreach (var param in parameters)
        {
            var grad = param.grad;
            if (grad is not null)
            {
                var norm = grad.norm().item<float>();
                totalNorm += norm * norm;
            }
        }

        totalNorm = Math.Sqrt(totalNorm);

        if (totalNorm > maxNorm)
        {
            var clipCoeff = maxNorm / (totalNorm + 1e-6);
            foreach (var param in parameters)
            {
                var grad = param.grad;
                if (grad is not null)
                {
                    grad.mul_(clipCoeff);
                }
            }
        }
    }

    /// <summary>
    /// 全局梯度范数裁剪（ClipGradByGlobalNorm）。
    /// 计算所有参数梯度的全局 L2 范数，如果超过 maxNorm 则按比例缩放所有梯度。
    /// 参考: paddle.nn.ClipGradByGlobalNorm
    /// </summary>
    /// <returns>裁剪前的全局梯度范数。</returns>
    public static double ClipGradGlobalNorm(Module<Tensor, Tensor> model, float maxNorm)
    {
        var parameters = model.parameters().ToList();
        var totalNorm = 0.0;

        foreach (var param in parameters)
        {
            var grad = param.grad;
            if (grad is not null)
            {
                var paramNorm = grad.norm(2).item<double>();
                totalNorm += paramNorm * paramNorm;
            }
        }

        totalNorm = Math.Sqrt(totalNorm);

        if (totalNorm > maxNorm)
        {
            var clipCoeff = maxNorm / (totalNorm + 1e-6);
            foreach (var param in parameters)
            {
                var grad = param.grad;
                if (grad is not null)
                {
                    grad.mul_(clipCoeff);
                }
            }
        }

        return totalNorm;
    }

    /// <summary>
    /// 按指定模式裁剪梯度。
    /// </summary>
    /// <param name="model">模型</param>
    /// <param name="maxNorm">最大范数</param>
    /// <param name="useGlobalNorm">true 使用全局范数裁剪，false 使用参数独立裁剪</param>
    public static void ClipGrad(Module<Tensor, Tensor> model, float maxNorm, bool useGlobalNorm = false)
    {
        if (useGlobalNorm)
        {
            ClipGradGlobalNorm(model, maxNorm);
        }
        else
        {
            ClipGradNorm(model, maxNorm);
        }
    }

    /// <summary>
    /// 梯度累积辅助类。
    /// </summary>
    public sealed class GradientAccumulator
    {
        private readonly int _accumulationSteps;
        private int _currentStep;

        public GradientAccumulator(int accumulationSteps)
        {
            _accumulationSteps = Math.Max(1, accumulationSteps);
            _currentStep = 0;
        }

        public bool ShouldUpdate()
        {
            _currentStep++;
            if (_currentStep >= _accumulationSteps)
            {
                _currentStep = 0;
                return true;
            }

            return false;
        }

        public void Reset()
        {
            _currentStep = 0;
        }
    }
}
