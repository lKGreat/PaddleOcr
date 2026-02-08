using FluentAssertions;
using PaddleOcr.Training.Runtime;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Tests;

public sealed class TrainingDeviceResolverTests
{
    [Fact]
    public void Resolve_Should_Use_Cpu_When_Requested_Cpu()
    {
        var resolved = TrainingDeviceResolver.Resolve("cpu", useGpu: false, useAmp: false, () => new CudaRuntimeInfo(false, 0));
        resolved.UseCuda.Should().BeFalse();
        resolved.Device.type.Should().Be(DeviceType.CPU);
    }

    [Fact]
    public void Resolve_Should_Use_Cuda_When_Auto_And_Cuda_Available()
    {
        if (!cuda.is_available())
        {
            return;
        }

        var resolved = TrainingDeviceResolver.Resolve("auto", useGpu: false, useAmp: true, () => new CudaRuntimeInfo(true, 1));
        resolved.UseCuda.Should().BeTrue();
        resolved.UseAmp.Should().BeTrue();
        resolved.Device.type.Should().Be(DeviceType.CUDA);
    }

    [Fact]
    public void Resolve_Should_Fallback_To_Cpu_When_Auto_And_No_Cuda()
    {
        var resolved = TrainingDeviceResolver.Resolve("auto", useGpu: false, useAmp: false, () => new CudaRuntimeInfo(false, 0));
        resolved.UseCuda.Should().BeFalse();
        resolved.Device.type.Should().Be(DeviceType.CPU);
        resolved.Reason.Should().Contain("fallback");
    }

    [Fact]
    public void Resolve_Should_Throw_When_Cuda_Requested_But_Unavailable()
    {
        var act = () => TrainingDeviceResolver.Resolve("cuda", useGpu: false, useAmp: false, () => new CudaRuntimeInfo(false, 0));
        act.Should().Throw<InvalidOperationException>().WithMessage("*cuda is not available*");
    }

    [Fact]
    public void Resolve_Should_Throw_When_Amp_Enabled_On_Cpu()
    {
        var act = () => TrainingDeviceResolver.Resolve("cpu", useGpu: false, useAmp: true, () => new CudaRuntimeInfo(false, 0));
        act.Should().Throw<InvalidOperationException>().WithMessage("*use_amp=true requires cuda*");
    }

    [Fact]
    public void Resolve_Should_Throw_When_Cuda_Index_Out_Of_Range()
    {
        var act = () => TrainingDeviceResolver.Resolve("cuda:2", useGpu: false, useAmp: false, () => new CudaRuntimeInfo(true, 1));
        act.Should().Throw<InvalidOperationException>().WithMessage("*device_count=1*");
    }

    [Fact]
    public void Resolve_Should_Map_UseGpu_To_Cuda_When_Device_Missing()
    {
        if (!cuda.is_available())
        {
            return;
        }

        var resolved = TrainingDeviceResolver.Resolve("", useGpu: true, useAmp: false, () => new CudaRuntimeInfo(true, 1));
        resolved.RequestedDevice.Should().Be("cuda");
        resolved.UseCuda.Should().BeTrue();
    }
}
