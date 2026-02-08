using FluentAssertions;

namespace PaddleOcr.Tests;

public sealed class CtcLengthSanitizerTests
{
    [Fact]
    public void Sanitize_Should_Clamp_Target_Length_To_Input_Length()
    {
        var result = InvokeSanitize(
            rawTargetLengths: [6],
            validRatios: [0.5f],
            flatLabelCtc: [1L, 2, 3, 4, 5, 6],
            ctcTimeSteps: 8,
            maxTextLength: 6,
            useValidRatio: true);

        GetLongArray(result, "InputLengths").Should().Equal(4L);
        GetLongArray(result, "TargetLengths").Should().Equal(4L);
        GetInt(result, "TruncatedByInput").Should().BeGreaterThan(0);
    }

    [Fact]
    public void Sanitize_Should_Respect_Repeat_Constraint_For_Ctc()
    {
        var result = InvokeSanitize(
            rawTargetLengths: [4],
            validRatios: [0.5f],
            flatLabelCtc: [7L, 7, 7, 7],
            ctcTimeSteps: 6,
            maxTextLength: 4,
            useValidRatio: true);

        GetLongArray(result, "InputLengths").Should().Equal(3L);
        GetLongArray(result, "TargetLengths").Should().Equal(2L);
        GetInt(result, "TruncatedByRepeatConstraint").Should().BeGreaterThan(0);
    }

    private static object InvokeSanitize(
        int[] rawTargetLengths,
        float[] validRatios,
        long[] flatLabelCtc,
        int ctcTimeSteps,
        int maxTextLength,
        bool useValidRatio)
    {
        var asm = typeof(PaddleOcr.Training.TrainingExecutor).Assembly;
        var type = asm.GetType("PaddleOcr.Training.Rec.CtcLengthSanitizer", throwOnError: true)!;
        var method = type.GetMethod("Sanitize", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        return method.Invoke(
            null,
            [rawTargetLengths, validRatios, flatLabelCtc, ctcTimeSteps, maxTextLength, useValidRatio])!;
    }

    private static long[] GetLongArray(object result, string propertyName)
    {
        var value = result.GetType().GetProperty(propertyName)!.GetValue(result);
        return value switch
        {
            long[] arr => arr,
            IEnumerable<long> seq => seq.ToArray(),
            _ => []
        };
    }

    private static int GetInt(object result, string propertyName)
    {
        return (int)(result.GetType().GetProperty(propertyName)!.GetValue(result) ?? 0);
    }
}
