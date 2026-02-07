namespace PaddleOcr.Training;

public sealed record TrainingSummary(int Epochs, float BestAccuracy, string SaveDir);
public sealed record EvaluationSummary(float Accuracy, int Samples);

public sealed record TrainingRunSummary(
    string ModelType,
    int EpochsRequested,
    int EpochsCompleted,
    string BestMetricName,
    float BestMetricValue,
    bool EarlyStopped,
    string SaveDir,
    string? ResumeCheckpoint,
    DateTime GeneratedAtUtc,
    int? Seed = null,
    string? Device = null,
    string? EarlyStopReason = null,
    bool? NanDetected = null);
