using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training;

internal sealed class SimpleDetTrainer
{
    private readonly ILogger _logger;

    public SimpleDetTrainer(ILogger logger)
    {
        _logger = logger;
    }

    public TrainingSummary Train(TrainingConfigView cfg)
    {
        SeedEverything(cfg.Seed);
        var size = cfg.DetInputSize;
        var trainSet = new SimpleDetDataset(
            cfg.TrainLabelFile,
            cfg.DataDir,
            size,
            cfg.InvalidSamplePolicy,
            cfg.MinValidSamples,
            cfg.DetShrinkRatio,
            cfg.DetThreshMin,
            cfg.DetThreshMax);
        var evalSet = new SimpleDetDataset(
            cfg.EvalLabelFile,
            cfg.EvalDataDir,
            size,
            cfg.InvalidSamplePolicy,
            cfg.MinValidSamples,
            cfg.DetShrinkRatio,
            cfg.DetThreshMin,
            cfg.DetThreshMax);

        var dev = ResolveDevice(cfg);
        _logger.LogInformation("Training(det) device: {Device}", dev.type);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}", trainSet.Count, evalSet.Count);
        _logger.LogInformation("deterministic={Deterministic}, seed={Seed}", cfg.Deterministic, cfg.Seed);

        Directory.CreateDirectory(cfg.SaveModelDir);
        var configFingerprint = BuildDetConfigFingerprint(cfg);
        var ckptMeta = new DetCheckpointMeta(
            ModelType: "det",
            InputSize: size,
            Seed: cfg.Seed,
            Deterministic: cfg.Deterministic,
            ConfigFingerprint: configFingerprint,
            GeneratedAtUtc: DateTime.UtcNow);
        WriteDataAudit(cfg.SaveModelDir, trainSet.Audit, evalSet.Audit);
        ResetHistory(cfg.SaveModelDir);

        using var model = new SimpleDetNet();
        model.to(dev);
        var lr = cfg.LearningRate;
        var optimizer = torch.optim.Adam(model.parameters(), lr: lr);
        var rng = new Random(cfg.Seed);
        var evalRng = new Random(cfg.Seed + 17);
        var resumeCkpt = cfg.ResumeTraining ? ResolveEvalCheckpoint(cfg) : null;
        if (!string.IsNullOrWhiteSpace(resumeCkpt) && File.Exists(resumeCkpt))
        {
            ValidateCheckpointMetaOrThrow(resumeCkpt, ckptMeta);
            _logger.LogInformation("Loading checkpoint: {Path}", resumeCkpt);
            model.load(resumeCkpt);
        }

        float bestFscore = -1f;
        var epochsCompleted = 0;
        var staleEpochs = 0;
        var earlyStopped = false;
        var earlyStopReason = string.Empty;
        var nanDetected = false;
        var stopTraining = false;
        for (var epoch = 1; epoch <= cfg.EpochNum; epoch++)
        {
            var sw = Stopwatch.StartNew();
            if (cfg.LrDecayStep > 0 && epoch > 1 && (epoch - 1) % cfg.LrDecayStep == 0)
            {
                lr *= cfg.LrDecayGamma;
                optimizer.Dispose();
                optimizer = torch.optim.Adam(model.parameters(), lr: lr);
                _logger.LogInformation("lr decayed to {LearningRate:F6} at epoch {Epoch}", lr, epoch);
            }

            model.train();
            var lossSum = 0f;
            var samples = 0;
            foreach (var (images, shrinkMaps, thresholdMaps, batch) in trainSet.GetBatches(cfg.BatchSize, true, rng))
            {
                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, size, size).to(dev);
                using var shrinkGt = torch.tensor(shrinkMaps, dtype: ScalarType.Float32).reshape(batch, 1, size, size).to(dev);
                using var thresholdGt = torch.tensor(thresholdMaps, dtype: ScalarType.Float32).reshape(batch, 1, size, size).to(dev);
                optimizer.zero_grad();
                using var pred = model.call(x);
                using var shrinkPred = pred.narrow(1, 0, 1);
                using var thresholdPred = pred.narrow(1, 1, 1);
                using var shrinkLoss = functional.binary_cross_entropy_with_logits(shrinkPred, shrinkGt);
                using var thresholdLoss = functional.l1_loss(thresholdPred.sigmoid(), thresholdGt);
                using var loss = shrinkLoss * cfg.DetShrinkLossWeight + thresholdLoss * cfg.DetThresholdLossWeight;
                var lossValue = loss.ToSingle();
                if (cfg.NanGuard && !IsFinite(lossValue))
                {
                    nanDetected = true;
                    earlyStopReason = "nan_guard";
                    stopTraining = true;
                    _logger.LogError("det nan guard triggered at epoch={Epoch}: loss is NaN/Inf", epoch);
                    break;
                }

                loss.backward();
                if (cfg.GradClipNorm > 0f)
                {
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GradClipNorm);
                }

                optimizer.step();
                lossSum += lossValue * batch;
                samples += batch;
            }

            if (stopTraining)
            {
                break;
            }

            var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, size, dev, evalRng, cfg.DetEvalIouThresh);
            if (metrics.Fscore > bestFscore + cfg.MinImproveDelta)
            {
                bestFscore = metrics.Fscore;
                staleEpochs = 0;
                SaveCheckpointWithMeta(cfg.SaveModelDir, model, "best.pt", ckptMeta with { GeneratedAtUtc = DateTime.UtcNow });
            }
            else
            {
                staleEpochs++;
            }

            SaveCheckpointWithMeta(cfg.SaveModelDir, model, "latest.pt", ckptMeta with { GeneratedAtUtc = DateTime.UtcNow });
            sw.Stop();

            var trainLoss = lossSum / Math.Max(1, samples);
            _logger.LogInformation(
                "epoch={Epoch}/{Total} train_loss={Loss:F4} eval_p={P:F4} eval_r={R:F4} eval_f={F:F4} eval_iou={IoU:F4}",
                epoch, cfg.EpochNum, trainLoss, metrics.Precision, metrics.Recall, metrics.Fscore, metrics.Iou);
            AppendHistory(cfg.SaveModelDir, new DetTrainHistoryEntry(
                Epoch: epoch,
                TrainLoss: trainLoss,
                EvalPrecision: metrics.Precision,
                EvalRecall: metrics.Recall,
                EvalFscore: metrics.Fscore,
                EvalIou: metrics.Iou,
                LearningRate: lr,
                EpochTimeMs: sw.Elapsed.TotalMilliseconds));
            epochsCompleted = epoch;
            if (cfg.EarlyStopPatience > 0 && staleEpochs >= cfg.EarlyStopPatience)
            {
                earlyStopped = true;
                earlyStopReason = "patience";
                _logger.LogInformation("early stop triggered at epoch {Epoch} (patience={Patience})", epoch, cfg.EarlyStopPatience);
                break;
            }
        }

        optimizer.Dispose();
        var best = bestFscore < 0f ? 0f : bestFscore;
        var summary = new TrainingSummary(epochsCompleted, best, cfg.SaveModelDir);
        var result = new DetTrainingResult(
            Epochs: summary.Epochs,
            BestAccuracy: summary.BestAccuracy,
            SaveDir: summary.SaveDir,
            Seed: cfg.Seed,
            Device: dev.type.ToString(),
            EarlyStopReason: string.IsNullOrWhiteSpace(earlyStopReason) ? null : earlyStopReason,
            NanDetected: nanDetected);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "train_result.json"),
            JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
        var run = new TrainingRunSummary(
            ModelType: "det",
            EpochsRequested: cfg.EpochNum,
            EpochsCompleted: summary.Epochs,
            BestMetricName: "fscore",
            BestMetricValue: summary.BestAccuracy,
            EarlyStopped: earlyStopped || nanDetected,
            SaveDir: cfg.SaveModelDir,
            ResumeCheckpoint: resumeCkpt,
            GeneratedAtUtc: DateTime.UtcNow,
            Seed: cfg.Seed,
            Device: dev.type.ToString(),
            EarlyStopReason: string.IsNullOrWhiteSpace(earlyStopReason) ? null : earlyStopReason,
            NanDetected: nanDetected);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "train_run_summary.json"),
            JsonSerializer.Serialize(run, new JsonSerializerOptions { WriteIndented = true }));
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        SeedEverything(cfg.Seed);
        var size = cfg.DetInputSize;
        var evalSet = new SimpleDetDataset(
            cfg.EvalLabelFile,
            cfg.EvalDataDir,
            size,
            cfg.InvalidSamplePolicy,
            cfg.MinValidSamples,
            cfg.DetShrinkRatio,
            cfg.DetThreshMin,
            cfg.DetThreshMax);
        var dev = ResolveDevice(cfg);
        using var model = new SimpleDetNet();
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt) && File.Exists(ckpt))
        {
            _logger.LogInformation("Loading checkpoint: {Path}", ckpt);
            model.load(ckpt);
        }

        var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, size, dev, new Random(cfg.Seed + 23), cfg.DetEvalIouThresh);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "eval_result.json"),
            JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true }));
        _logger.LogInformation(
            "det eval metrics: precision={P:F4}, recall={R:F4}, fscore={F:F4}, iou={IoU:F4}, iou_thresh={T:F2}, tp={TP}, fp={FP}, fn={FN}",
            metrics.Precision, metrics.Recall, metrics.Fscore, metrics.Iou, metrics.IouThreshold, metrics.Tp, metrics.Fp, metrics.Fn);
        return new EvaluationSummary(metrics.Iou, evalSet.Count);
    }

    private static DetEvalMetrics Evaluate(SimpleDetNet model, SimpleDetDataset evalSet, int batchSize, int size, Device dev, Random evalRng, float evalIouThresh)
    {
        model.eval();
        var summary = new DetMatchSummary(0, 0, 0, 0f);
        using var noGrad = torch.no_grad();
        foreach (var (images, shrinkMaps, _, batch) in evalSet.GetBatches(batchSize, false, evalRng))
        {
            using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, size, size).to(dev);
            using var predAll = model.call(x);
            using var pred = predAll.narrow(1, 0, 1).sigmoid().cpu();
            var predData = pred.data<float>().ToArray();
            var area = size * size;
            for (var bi = 0; bi < batch; bi++)
            {
                var predMask = new bool[area];
                var gtMask = new bool[area];
                var offset = bi * area;
                for (var i = 0; i < area; i++)
                {
                    predMask[i] = predData[offset + i] > 0.5f;
                    gtMask[i] = shrinkMaps[offset + i] > 0.5f;
                }

                summary += DetMetricEvaluator.EvaluateSingle(predMask, gtMask, size, size, evalIouThresh);
            }
        }

        return new DetEvalMetrics(
            Precision: summary.Precision,
            Recall: summary.Recall,
            Fscore: summary.Fscore,
            Iou: summary.MeanIou,
            IouThreshold: evalIouThresh,
            Tp: summary.TruePositive,
            Fp: summary.FalsePositive,
            Fn: summary.FalseNegative);
    }

    private static void SeedEverything(int seed)
    {
        torch.random.manual_seed(seed);
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static Device ResolveDevice(TrainingConfigView cfg)
    {
        if (cfg.Device.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            return CPU;
        }

        if (cfg.Device.Equals("auto", StringComparison.OrdinalIgnoreCase) && cuda.is_available())
        {
            return CUDA;
        }

        return CPU;
    }

    private static string? ResolveEvalCheckpoint(TrainingConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints))
        {
            return cfg.Checkpoints;
        }

        var best = Path.Combine(cfg.SaveModelDir, "best.pt");
        if (File.Exists(best))
        {
            return best;
        }

        var latest = Path.Combine(cfg.SaveModelDir, "latest.pt");
        if (File.Exists(latest))
        {
            return latest;
        }

        return null;
    }

    private static string BuildDetConfigFingerprint(TrainingConfigView cfg)
    {
        var raw = string.Join('|', new[]
        {
            cfg.ModelType,
            cfg.DetInputSize.ToString(),
            cfg.BatchSize.ToString(),
            cfg.EvalBatchSize.ToString(),
            cfg.LearningRate.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.DetShrinkRatio.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.DetThreshMin.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.DetThreshMax.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.DetShrinkLossWeight.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.DetThresholdLossWeight.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.DetEvalIouThresh.ToString(System.Globalization.CultureInfo.InvariantCulture),
            cfg.TrainLabelFile,
            cfg.EvalLabelFile,
            cfg.DataDir,
            cfg.EvalDataDir
        });
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(raw))).ToLowerInvariant();
    }

    private static void SaveCheckpointWithMeta(string saveDir, SimpleDetNet model, string fileName, DetCheckpointMeta meta)
    {
        Directory.CreateDirectory(saveDir);
        var path = Path.Combine(saveDir, fileName);
        model.save(path);
        File.WriteAllText(path + ".meta.json", JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void ValidateCheckpointMetaOrThrow(string checkpointPath, DetCheckpointMeta expected)
    {
        var metaPath = checkpointPath + ".meta.json";
        if (!File.Exists(metaPath))
        {
            return;
        }

        var meta = JsonSerializer.Deserialize<DetCheckpointMeta>(File.ReadAllText(metaPath));
        if (meta is null)
        {
            throw new InvalidOperationException($"checkpoint metadata parse failed: {metaPath}");
        }

        if (!meta.ModelType.Equals(expected.ModelType, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"checkpoint model_type mismatch: {meta.ModelType} != {expected.ModelType}");
        }

        if (meta.InputSize != expected.InputSize)
        {
            throw new InvalidOperationException($"checkpoint input_size mismatch: {meta.InputSize} != {expected.InputSize}");
        }

        if (!meta.ConfigFingerprint.Equals(expected.ConfigFingerprint, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("checkpoint config fingerprint mismatch");
        }
    }

    private static void WriteDataAudit(string saveDir, DetDataAudit trainAudit, DetDataAudit evalAudit)
    {
        var payload = new
        {
            train = trainAudit,
            eval = evalAudit,
            generated_at_utc = DateTime.UtcNow
        };
        File.WriteAllText(
            Path.Combine(saveDir, "det_data_audit.json"),
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void ResetHistory(string saveDir)
    {
        File.WriteAllText(Path.Combine(saveDir, "det_train_history.jsonl"), string.Empty);
    }

    private static void AppendHistory(string saveDir, DetTrainHistoryEntry entry)
    {
        File.AppendAllText(
            Path.Combine(saveDir, "det_train_history.jsonl"),
            JsonSerializer.Serialize(entry) + Environment.NewLine);
    }
}

internal sealed class SimpleDetNet : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _head;

    public SimpleDetNet() : base(nameof(SimpleDetNet))
    {
        _backbone = Sequential(
            ("conv1", Conv2d(3, 16, 3, stride: 1, padding: 1)),
            ("relu1", ReLU()),
            ("conv2", Conv2d(16, 32, 3, stride: 1, padding: 1)),
            ("relu2", ReLU()),
            ("conv3", Conv2d(32, 32, 3, stride: 1, padding: 1)),
            ("relu3", ReLU())
        );
        _head = Conv2d(32, 2, 1);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var feat = _backbone.call(input);
        return _head.call(feat);
    }
}

internal sealed record DetEvalMetrics(float Precision, float Recall, float Fscore, float Iou, float IouThreshold, int Tp, int Fp, int Fn);
internal sealed record DetCheckpointMeta(string ModelType, int InputSize, int Seed, bool Deterministic, string ConfigFingerprint, DateTime GeneratedAtUtc);
internal sealed record DetTrainHistoryEntry(
    int Epoch,
    float TrainLoss,
    float EvalPrecision,
    float EvalRecall,
    float EvalFscore,
    float EvalIou,
    float LearningRate,
    double EpochTimeMs);
internal sealed record DetTrainingResult(
    int Epochs,
    float BestAccuracy,
    string SaveDir,
    int Seed,
    string Device,
    string? EarlyStopReason,
    bool NanDetected);
