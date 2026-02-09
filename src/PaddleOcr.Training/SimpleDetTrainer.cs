using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Training.Det.Losses;
using PaddleOcr.Training.Runtime;
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

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        _logger.LogInformation("Training(det) device: {Device}", dev.type);
        _logger.LogInformation("runtime: requested={Requested}, cuda={Cuda}, amp={Amp}, reason={Reason}", runtime.RequestedDevice, runtime.UseCuda, runtime.UseAmp, runtime.Reason);
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
        var rng = new Random(cfg.Seed);
        var evalRng = new Random(cfg.Seed + 17);
        var resumeCkpt = cfg.ResumeTraining ? ResolveEvalCheckpoint(cfg) : null;
        if (!string.IsNullOrWhiteSpace(resumeCkpt) && File.Exists(resumeCkpt))
        {
            ValidateCheckpointMetaOrThrow(resumeCkpt, ckptMeta);
            _logger.LogInformation("Loading checkpoint: {Path}", resumeCkpt);
            model.load(resumeCkpt);
        }
        var optimizer = torch.optim.Adam(model.parameters(), lr: lr);

        // Initialize DBLoss with configured weights
        using var dbLoss = new DBLoss(
            alpha: cfg.DetShrinkLossWeight,
            beta: cfg.DetThresholdLossWeight,
            balanceLoss: true,
            ohemRatio: 3f,
            eps: 1e-6f);
        dbLoss.to(dev);

        // Training stats with median smoothing (matching Python PaddleOCR)
        var trainStats = new TrainingStats(cfg.LogSmoothWindow, ["lr"]);
        var iterTimer = new IterationTimer(cfg.LogSmoothWindow, cfg.PrintBatchStep);
        var tracePath = Path.Combine(cfg.SaveModelDir, "det_train_trace.jsonl");
        var epochTracePath = Path.Combine(cfg.SaveModelDir, "det_train_history.jsonl");
        File.WriteAllText(tracePath, string.Empty); // reset trace

        float bestFscore = -1f;
        var epochsCompleted = 0;
        var staleEpochs = 0;
        var earlyStopped = false;
        var earlyStopReason = string.Empty;
        var nanDetected = false;
        var stopTraining = false;
        var globalStep = 0;
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
            var batchIdx = 0;
            var totalBatches = (trainSet.Count + cfg.BatchSize - 1) / cfg.BatchSize;
            foreach (var (images, shrinkMaps, shrinkMasks, thresholdMaps, thresholdMasks, batch) in trainSet.GetBatches(cfg.BatchSize, true, rng))
            {
                iterTimer.StartReader();
                // Data is already loaded in GetBatches
                iterTimer.EndReader();

                iterTimer.StartBatch();

                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, size, size).to(dev);
                using var gtShrinkMap = torch.tensor(shrinkMaps, dtype: ScalarType.Float32).reshape(batch, size, size).to(dev);
                using var gtShrinkMask = torch.tensor(shrinkMasks, dtype: ScalarType.Float32).reshape(batch, size, size).to(dev);
                using var gtThresholdMap = torch.tensor(thresholdMaps, dtype: ScalarType.Float32).reshape(batch, size, size).to(dev);
                using var gtThresholdMask = torch.tensor(thresholdMasks, dtype: ScalarType.Float32).reshape(batch, size, size).to(dev);

                optimizer.zero_grad();
                using var pred = model.call(x);  // [B, 3, H, W]

                // Prepare predictions and batch dictionaries for DBLoss
                var predictions = new Dictionary<string, Tensor>
                {
                    ["maps"] = pred
                };
                var batchDict = new Dictionary<string, Tensor>
                {
                    ["shrink_map"] = gtShrinkMap,
                    ["shrink_mask"] = gtShrinkMask,
                    ["threshold_map"] = gtThresholdMap,
                    ["threshold_mask"] = gtThresholdMask
                };

                // Compute loss using DBLoss
                var losses = dbLoss.Forward(predictions, batchDict);
                using var loss = losses["loss"];
                var lossValue = loss.ToSingle();

                // Extract individual loss components
                float lossShrinkVal = 0f, lossThreshVal = 0f, lossBinaryVal = 0f;
                if (losses.TryGetValue("loss_shrink_maps", out var lossShrink))
                {
                    lossShrinkVal = lossShrink.ToSingle();
                    lossShrink.Dispose();
                }
                if (losses.TryGetValue("loss_threshold_maps", out var lossThresh))
                {
                    lossThreshVal = lossThresh.ToSingle();
                    lossThresh.Dispose();
                }
                if (losses.TryGetValue("loss_binary_maps", out var lossBinary))
                {
                    lossBinaryVal = lossBinary.ToSingle();
                    lossBinary.Dispose();
                }

                if (cfg.NanGuard && !IsFinite(lossValue))
                {
                    nanDetected = true;
                    earlyStopReason = "nan_guard";
                    stopTraining = true;
                    _logger.LogError("det nan guard triggered at epoch={Epoch} step={Step}: loss is NaN/Inf", epoch, globalStep);
                    AppendJsonLine(tracePath, new
                    {
                        event_type = "nan_guard",
                        epoch,
                        global_step = globalStep,
                        loss = lossValue,
                        loss_shrink_maps = lossShrinkVal,
                        loss_threshold_maps = lossThreshVal,
                        loss_binary_maps = lossBinaryVal
                    });
                    break;
                }

                loss.backward();

                // Compute gradient norm for diagnostics
                var gradNorm = EstimateGradNorm(model);
                if (float.IsNaN(gradNorm) || float.IsInfinity(gradNorm))
                {
                    _logger.LogWarning("det gradient norm is NaN/Inf at step {Step}", globalStep);
                }

                if (cfg.GradClipNorm > 0f)
                {
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GradClipNorm);
                }

                optimizer.step();
                lossSum += lossValue * batch;
                samples += batch;
                globalStep++;

                iterTimer.EndBatch(batch);

                // Update training stats with all loss components (matching Python PaddleOCR)
                trainStats.Update(new Dictionary<string, float>
                {
                    ["loss"] = lossValue,
                    ["loss_shrink_maps"] = lossShrinkVal,
                    ["loss_threshold_maps"] = lossThreshVal,
                    ["loss_binary_maps"] = lossBinaryVal,
                    ["lr"] = lr,
                    ["grad_norm"] = gradNorm
                });

                // Periodic sample prediction vs ground truth logging (for debugging)
                if (globalStep % (cfg.PrintBatchStep * 10) == 0)
                {
                    LogDetSamplePredictions(pred, shrinkMaps, size, batch, globalStep, epoch);
                }

                // Per-iteration logging (matching Python PaddleOCR format)
                if (globalStep % cfg.PrintBatchStep == 0 || batchIdx >= totalBatches - 1)
                {
                    var smoothed = trainStats.Get();
                    var eta = iterTimer.GetEta(epoch, cfg.EpochNum, batchIdx, totalBatches);
                    _logger.LogInformation(
                        "epoch: [{Epoch}/{Total}], global_step: {Step}, {StatsLog}, avg_reader_cost: {ReaderCost:F5} s, avg_batch_cost: {BatchCost:F5} s, avg_samples: {AvgSamples}, ips: {Ips:F2} samples/s, eta: {Eta}",
                        epoch, cfg.EpochNum, globalStep,
                        trainStats.Log(),
                        iterTimer.AvgReaderCost,
                        iterTimer.AvgBatchCost,
                        batch,
                        iterTimer.AvgBatchCost > 0 ? batch / iterTimer.AvgBatchCost : 0,
                        eta);

                    // Per-iteration JSONL trace
                    AppendJsonLine(tracePath, new
                    {
                        epoch,
                        global_step = globalStep,
                        loss = lossValue,
                        loss_shrink_maps = lossShrinkVal,
                        loss_threshold_maps = lossThreshVal,
                        loss_binary_maps = lossBinaryVal,
                        smooth_loss = smoothed.GetValueOrDefault("loss"),
                        smooth_loss_shrink = smoothed.GetValueOrDefault("loss_shrink_maps"),
                        smooth_loss_threshold = smoothed.GetValueOrDefault("loss_threshold_maps"),
                        smooth_loss_binary = smoothed.GetValueOrDefault("loss_binary_maps"),
                        lr,
                        grad_norm = gradNorm,
                        batch,
                        avg_reader_cost = iterTimer.AvgReaderCost,
                        avg_batch_cost = iterTimer.AvgBatchCost,
                        ips = iterTimer.AvgBatchCost > 0 ? batch / iterTimer.AvgBatchCost : 0,
                        eta
                    });
                }

                batchIdx++;
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
                "epoch={Epoch}/{Total} train_loss={Loss:F4} eval_p={P:F4} eval_r={R:F4} eval_f={F:F4} eval_iou={IoU:F4} lr={Lr:F6} time={Time:F1}s",
                epoch, cfg.EpochNum, trainLoss, metrics.Precision, metrics.Recall, metrics.Fscore, metrics.Iou, lr, sw.Elapsed.TotalSeconds);

            // Log eval metrics to JSONL
            AppendJsonLine(epochTracePath, new
            {
                event_type = "epoch_end",
                epoch,
                global_step = globalStep,
                train_loss = trainLoss,
                eval_precision = metrics.Precision,
                eval_recall = metrics.Recall,
                eval_fscore = metrics.Fscore,
                eval_iou = metrics.Iou,
                best_fscore = bestFscore,
                lr,
                epoch_time_ms = sw.Elapsed.TotalMilliseconds,
                stale_epochs = staleEpochs
            });

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
        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
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
        foreach (var (images, shrinkMaps, _, _, _, batch) in evalSet.GetBatches(batchSize, false, evalRng))
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

    private static string? ResolveEvalCheckpoint(TrainingConfigView cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.Checkpoints))
        {
            return cfg.Checkpoints;
        }

        if (!string.IsNullOrWhiteSpace(cfg.PretrainedModel))
        {
            return cfg.PretrainedModel;
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

    /// <summary>
    /// Estimate gradient L2 norm across all parameters.
    /// Critical for detecting vanishing/exploding gradients.
    /// </summary>
    /// <summary>
    /// Log detection sample predictions vs ground truth for debugging.
    /// Shows predicted positive pixel ratio vs GT positive pixel ratio per sample.
    /// </summary>
    private void LogDetSamplePredictions(Tensor pred, float[] gtShrinkMaps, int size, int batch, int globalStep, int epoch)
    {
        try
        {
            using var predSigmoid = pred.narrow(1, 0, 1).sigmoid().cpu();
            var predData = predSigmoid.data<float>().ToArray();
            var area = size * size;
            var numSamples = Math.Min(3, batch);
            for (var i = 0; i < numSamples; i++)
            {
                var offset = i * area;
                var predPositive = 0;
                var gtPositive = 0;
                for (var j = 0; j < area; j++)
                {
                    if (predData[offset + j] > 0.5f) predPositive++;
                    if (gtShrinkMaps[offset + j] > 0.5f) gtPositive++;
                }
                var predRatio = (float)predPositive / area;
                var gtRatio = (float)gtPositive / area;
                _logger.LogInformation(
                    "sample_pred(det) epoch={Epoch} step={Step} sample={SampleIdx} pred_positive_ratio={PredRatio:F4} gt_positive_ratio={GtRatio:F4} pred_pixels={PredPix} gt_pixels={GtPix}",
                    epoch, globalStep, i, predRatio, gtRatio, predPositive, gtPositive);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to log det sample predictions at step {Step}", globalStep);
        }
    }

    private static float EstimateGradNorm(Module model)
    {
        double totalNorm = 0;
        foreach (var param in model.parameters())
        {
            var grad = param.grad;
            if (grad is not null)
            {
                using var gradCpu = grad.cpu().to_type(ScalarType.Float32);
                var values = gradCpu.data<float>().ToArray();
                for (var i = 0; i < values.Length; i++)
                {
                    var v = values[i];
                    totalNorm += v * v;
                }
            }
        }
        return (float)Math.Sqrt(totalNorm);
    }

    private static void AppendJsonLine(string filePath, object data)
    {
        File.AppendAllText(filePath, JsonSerializer.Serialize(data) + Environment.NewLine);
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
