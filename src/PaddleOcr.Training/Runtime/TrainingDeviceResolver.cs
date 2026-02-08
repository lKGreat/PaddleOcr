using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Runtime;

public readonly record struct CudaRuntimeInfo(bool Available, int DeviceCount);

public readonly record struct ResolvedTrainingRuntime(
    Device Device,
    bool UseCuda,
    bool UseAmp,
    string RequestedDevice,
    string Reason);

public static class TrainingDeviceResolver
{
    internal static ResolvedTrainingRuntime Resolve(TrainingConfigView cfg)
    {
        return Resolve(
            cfg.Device,
            cfg.UseGpu,
            cfg.UseAmp,
            () => new CudaRuntimeInfo(cuda.is_available(), cuda.is_available() ? cuda.device_count() : 0));
    }

    public static ResolvedTrainingRuntime Resolve(
        string deviceRaw,
        bool useGpu,
        bool useAmp,
        Func<CudaRuntimeInfo> cudaProbe)
    {
        var cuda = cudaProbe();
        var requested = ResolveRequestedDevice(deviceRaw, useGpu);

        if (requested.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            if (useAmp)
            {
                throw new InvalidOperationException("training runtime invalid: use_amp=true requires cuda device");
            }

            return new ResolvedTrainingRuntime(CPU, false, false, requested, "cpu requested");
        }

        if (requested.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            if (!cuda.Available)
            {
                return new ResolvedTrainingRuntime(CPU, false, false, requested, "auto fallback to cpu");
            }

            return new ResolvedTrainingRuntime(CUDA, true, useAmp, requested, "auto selected cuda");
        }

        if (!requested.StartsWith("cuda", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"training runtime invalid: unsupported device '{requested}', expected cpu|auto|cuda|cuda:N");
        }

        if (!cuda.Available)
        {
            throw new InvalidOperationException($"training runtime invalid: requested device={requested} but cuda is not available");
        }

        var idx = ParseCudaIndex(requested);
        if (idx >= 0 && idx >= Math.Max(1, cuda.DeviceCount))
        {
            throw new InvalidOperationException($"training runtime invalid: requested device={requested} but cuda device_count={cuda.DeviceCount}");
        }

        var device = CUDA;
        return new ResolvedTrainingRuntime(device, true, useAmp, requested, idx >= 0 ? $"cuda device {idx}" : "cuda default device");
    }

    private static string ResolveRequestedDevice(string deviceRaw, bool useGpu)
    {
        if (!string.IsNullOrWhiteSpace(deviceRaw))
        {
            return deviceRaw.Trim().ToLowerInvariant();
        }

        return useGpu ? "cuda" : "cpu";
    }

    private static int ParseCudaIndex(string requested)
    {
        var sep = requested.IndexOf(':');
        if (sep < 0 || sep == requested.Length - 1)
        {
            return -1;
        }

        return int.TryParse(requested[(sep + 1)..], out var idx) && idx >= 0 ? idx : -1;
    }
}
