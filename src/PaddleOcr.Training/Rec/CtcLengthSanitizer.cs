namespace PaddleOcr.Training.Rec;

internal static class CtcLengthSanitizer
{
    public static CtcLengthSanitizeResult Sanitize(
        IReadOnlyList<int> rawTargetLengths,
        IReadOnlyList<float> validRatios,
        IReadOnlyList<long> flatLabelCtc,
        int ctcTimeSteps,
        int maxTextLength,
        bool useValidRatio)
    {
        if (ctcTimeSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ctcTimeSteps), "ctcTimeSteps must be positive.");
        }

        var batch = rawTargetLengths.Count;
        var inputLengths = new long[batch];
        var targetLengths = new long[batch];
        var truncatedByTime = 0;
        var truncatedByInput = 0;
        var truncatedByRepeatConstraint = 0;
        var emptyTargets = 0;

        for (var i = 0; i < batch; i++)
        {
            var inputLen = ctcTimeSteps;
            if (useValidRatio)
            {
                var ratio = i < validRatios.Count ? validRatios[i] : 1f;
                inputLen = (int)Math.Round(ctcTimeSteps * Math.Clamp(ratio, 0f, 1f));
            }

            inputLen = Math.Clamp(inputLen, 1, ctcTimeSteps);
            inputLengths[i] = inputLen;

            var targetLen = i < rawTargetLengths.Count ? rawTargetLengths[i] : 0;
            targetLen = Math.Clamp(targetLen, 0, maxTextLength);
            if (targetLen > ctcTimeSteps)
            {
                targetLen = ctcTimeSteps;
                truncatedByTime++;
            }

            if (targetLen > inputLen)
            {
                targetLen = inputLen;
                truncatedByInput++;
            }

            while (targetLen > 0 && RequiredCtcTimeForPrefix(flatLabelCtc, i, targetLen, maxTextLength) > inputLen)
            {
                targetLen--;
                truncatedByRepeatConstraint++;
            }

            if (targetLen <= 0)
            {
                emptyTargets++;
            }

            targetLengths[i] = targetLen;
        }

        return new CtcLengthSanitizeResult(
            inputLengths,
            targetLengths,
            truncatedByTime,
            truncatedByInput,
            truncatedByRepeatConstraint,
            emptyTargets);
    }

    private static int RequiredCtcTimeForPrefix(IReadOnlyList<long> flatLabelCtc, int sampleIndex, int targetLen, int maxTextLength)
    {
        if (targetLen <= 0)
        {
            return 0;
        }

        var baseOffset = sampleIndex * maxTextLength;
        var repeats = 0;
        long prev = -1;
        for (var j = 0; j < targetLen; j++)
        {
            var idx = baseOffset + j;
            if (idx < 0 || idx >= flatLabelCtc.Count)
            {
                break;
            }

            var cur = flatLabelCtc[idx];
            if (j > 0 && cur > 0 && cur == prev)
            {
                repeats++;
            }

            prev = cur;
        }

        return targetLen + repeats;
    }
}

internal sealed record CtcLengthSanitizeResult(
    long[] InputLengths,
    long[] TargetLengths,
    int TruncatedByTime,
    int TruncatedByInput,
    int TruncatedByRepeatConstraint,
    int EmptyTargets);
