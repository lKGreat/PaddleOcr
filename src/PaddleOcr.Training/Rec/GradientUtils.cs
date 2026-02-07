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
    /// 梯度裁剪。
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
