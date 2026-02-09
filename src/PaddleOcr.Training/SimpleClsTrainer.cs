using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Training.Runtime;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training;

internal sealed class SimpleClsTrainer
{
    private readonly ILogger _logger;

    public SimpleClsTrainer(ILogger logger)
    {
        _logger = logger;
    }

    public TrainingSummary Train(TrainingConfigView cfg)
    {
        var shape = cfg.ImageShape;
        var trainSet = new SimpleClsDataset(cfg.TrainLabelFile, cfg.DataDir, shape.H, shape.W);
        var evalSet = new SimpleClsDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W);
        var numClasses = Math.Max(2, EstimateNumClasses(trainSet, cfg.BatchSize));

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        _logger.LogInformation("Training(cls) device: {Device}", dev.type);
        _logger.LogInformation("runtime: requested={Requested}, cuda={Cuda}, amp={Amp}, reason={Reason}", runtime.RequestedDevice, runtime.UseCuda, runtime.UseAmp, runtime.Reason);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}, Classes: {Classes}", trainSet.Count, evalSet.Count, numClasses);

        Directory.CreateDirectory(cfg.SaveModelDir);

        // Training stats with median smoothing (matching Python PaddleOCR)
        var trainStats = new TrainingStats(cfg.LogSmoothWindow, ["lr"]);
        var iterTimer = new IterationTimer(cfg.LogSmoothWindow, cfg.PrintBatchStep);
        var tracePath = Path.Combine(cfg.SaveModelDir, "cls_train_trace.jsonl");
        var epochTracePath = Path.Combine(cfg.SaveModelDir, "cls_epoch_summary.jsonl");
        File.WriteAllText(tracePath, string.Empty);
        File.WriteAllText(epochTracePath, string.Empty);

        using var model = new SimpleClsNet(numClasses);
        model.to(dev);
        var lr = cfg.LearningRate;
        var resumeCkpt = cfg.ResumeTraining ? ResolveEvalCheckpoint(cfg) : null;
        if (!string.IsNullOrWhiteSpace(resumeCkpt))
        {
            TryLoadCheckpoint(model, resumeCkpt);
        }
        var optimizer = torch.optim.Adam(model.parameters(), lr: lr);

        var rng = new Random(cfg.Seed);
        float bestAcc = -1f;
        var epochsCompleted = 0;
        var staleEpochs = 0;
        var earlyStopped = false;
        var globalStep = 0;
        for (var epoch = 1; epoch <= cfg.EpochNum; epoch++)
        {
            var epochSw = Stopwatch.StartNew();
            if (cfg.LrDecayStep > 0 && epoch > 1 && (epoch - 1) % cfg.LrDecayStep == 0)
            {
                lr *= cfg.LrDecayGamma;
                optimizer.Dispose();
                optimizer = torch.optim.Adam(model.parameters(), lr: lr);
                _logger.LogInformation("lr decayed to {LearningRate:F6} at epoch {Epoch}", lr, epoch);
            }

            model.train();
            var lossSum = 0f;
            var sampleCount = 0;
            var correct = 0L;
            var batchIdx = 0;
            var totalBatches = (trainSet.Count + cfg.BatchSize - 1) / cfg.BatchSize;
            foreach (var (images, labels, batch) in trainSet.GetBatches(cfg.BatchSize, shuffle: true, rng))
            {
                iterTimer.StartReader();
                iterTimer.EndReader();

                iterTimer.StartBatch();

                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, shape.H, shape.W).to(dev);
                using var y = torch.tensor(labels, dtype: ScalarType.Int64).to(dev);
                optimizer.zero_grad();
                using var logits = model.call(x);
                using var loss = functional.cross_entropy(logits, y);
                var lossVal = loss.ToSingle();

                loss.backward();

                // Compute gradient norm for diagnostics
                var gradNorm = EstimateGradNorm(model);
                if (float.IsNaN(gradNorm) || float.IsInfinity(gradNorm))
                {
                    _logger.LogWarning("cls gradient norm is NaN/Inf at step {Step}", globalStep);
                }

                if (cfg.GradClipNorm > 0f)
                {
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GradClipNorm);
                }
                optimizer.step();

                lossSum += lossVal * batch;
                using var pred = logits.argmax(1);
                using var eq = pred.eq(y);
                correct += eq.sum().ToInt64();
                sampleCount += batch;
                globalStep++;

                iterTimer.EndBatch(batch);

                // Compute batch accuracy
                var batchAcc = batch == 0 ? 0f : (float)eq.sum().ToInt64() / batch;

                // Update training stats with median smoothing
                trainStats.Update(new Dictionary<string, float>
                {
                    ["loss"] = lossVal,
                    ["acc"] = batchAcc,
                    ["lr"] = lr,
                    ["grad_norm"] = gradNorm
                });

                // Periodic sample prediction vs ground truth logging (for debugging)
                if (globalStep % (cfg.PrintBatchStep * 10) == 0)
                {
                    LogClsSamplePredictions(logits, labels, batch, globalStep, epoch);
                }

                // Per-iteration logging (matching Python PaddleOCR format)
                if (globalStep % cfg.PrintBatchStep == 0 || batchIdx >= totalBatches - 1)
                {
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
                        loss = lossVal,
                        acc = batchAcc,
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

            var trainLoss = sampleCount == 0 ? 0f : lossSum / sampleCount;
            var trainAcc = sampleCount == 0 ? 0f : (float)correct / sampleCount;
            var evalAcc = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, dev);

            if (evalAcc > bestAcc)
            {
                bestAcc = evalAcc;
                staleEpochs = 0;
                SaveCheckpoint(cfg.SaveModelDir, model, "best.pt");
            }
            else
            {
                staleEpochs++;
            }

            SaveCheckpoint(cfg.SaveModelDir, model, "latest.pt");
            epochSw.Stop();
            _logger.LogInformation(
                "epoch={Epoch}/{Total} train_loss={Loss:F4} train_acc={TrainAcc:F4} eval_acc={EvalAcc:F4} best_acc={BestAcc:F4} lr={Lr:F6} time={Time:F1}s",
                epoch, cfg.EpochNum, trainLoss, trainAcc, evalAcc, bestAcc, lr, epochSw.Elapsed.TotalSeconds);

            // Epoch summary JSONL
            AppendJsonLine(epochTracePath, new
            {
                event_type = "epoch_end",
                epoch,
                global_step = globalStep,
                train_loss = trainLoss,
                train_acc = trainAcc,
                eval_acc = evalAcc,
                best_acc = bestAcc,
                lr,
                epoch_time_ms = epochSw.Elapsed.TotalMilliseconds,
                stale_epochs = staleEpochs
            });

            epochsCompleted = epoch;
            if (cfg.EarlyStopPatience > 0 && staleEpochs >= cfg.EarlyStopPatience)
            {
                earlyStopped = true;
                _logger.LogInformation("early stop triggered at epoch {Epoch} (patience={Patience})", epoch, cfg.EarlyStopPatience);
                break;
            }
        }

        optimizer.Dispose();
        var summary = new TrainingSummary(epochsCompleted, bestAcc, cfg.SaveModelDir);
        SaveSummary(cfg, "cls", "accuracy", summary, earlyStopped, resumeCkpt);
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        var shape = cfg.ImageShape;
        var evalSet = new SimpleClsDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W);
        var numClasses = 2;

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        using var model = new SimpleClsNet(numClasses);
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt))
        {
            TryLoadCheckpoint(model, ckpt);
        }

        var acc = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, dev);
        var summary = new EvaluationSummary(acc, evalSet.Count);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "eval_result.json"),
            JsonSerializer.Serialize(new { accuracy = acc, samples = evalSet.Count }, new JsonSerializerOptions { WriteIndented = true }));
        _logger.LogInformation("eval_acc={EvalAcc:F4} samples={Samples}", summary.Accuracy, summary.Samples);
        return summary;
    }

    private static int EstimateNumClasses(SimpleClsDataset dataSet, int batchSize)
    {
        var labels = new HashSet<long>();
        foreach (var (_, y, _) in dataSet.GetBatches(Math.Max(batchSize, 128), shuffle: false, new Random(1)))
        {
            foreach (var l in y)
            {
                labels.Add(l);
            }
        }

        return labels.Count;
    }

    private static float Evaluate(SimpleClsNet model, SimpleClsDataset evalSet, int batchSize, int h, int w, Device dev)
    {
        model.eval();
        long correct = 0;
        var total = 0;
        using var noGrad = torch.no_grad();
        foreach (var (images, labels, batch) in evalSet.GetBatches(batchSize, shuffle: false, new Random(7)))
        {
            using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, h, w).to(dev);
            using var y = torch.tensor(labels, dtype: ScalarType.Int64).to(dev);
            using var logits = model.call(x);
            using var pred = logits.argmax(1);
            using var eq = pred.eq(y);
            correct += eq.sum().ToInt64();
            total += batch;
        }

        return total == 0 ? 0f : (float)correct / total;
    }

    private void SaveCheckpoint(string saveDir, SimpleClsNet model, string fileName)
    {
        Directory.CreateDirectory(saveDir);
        var path = Path.Combine(saveDir, fileName);
        try
        {
            model.save(path);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save torch checkpoint {Path}", path);
        }
    }

    private void TryLoadCheckpoint(SimpleClsNet model, string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
        {
            _logger.LogWarning("Checkpoint not found: {Path}", checkpointPath);
            return;
        }

        try
        {
            _logger.LogInformation("Loading checkpoint: {Path}", checkpointPath);
            model.load(checkpointPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load checkpoint {Path}", checkpointPath);
        }
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

    private static void SaveSummary(TrainingConfigView cfg, string modelType, string metricName, TrainingSummary summary, bool earlyStopped, string? resumeCheckpoint)
    {
        Directory.CreateDirectory(cfg.SaveModelDir);
        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(cfg.SaveModelDir, "train_result.json"), json);
        var run = new TrainingRunSummary(
            ModelType: modelType,
            EpochsRequested: cfg.EpochNum,
            EpochsCompleted: summary.Epochs,
            BestMetricName: metricName,
            BestMetricValue: summary.BestAccuracy,
            EarlyStopped: earlyStopped,
            SaveDir: cfg.SaveModelDir,
            ResumeCheckpoint: resumeCheckpoint,
            GeneratedAtUtc: DateTime.UtcNow);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "train_run_summary.json"),
            JsonSerializer.Serialize(run, new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>
    /// Estimate gradient L2 norm across all parameters.
    /// </summary>
    /// <summary>
    /// Log classification sample predictions vs ground truth for debugging.
    /// </summary>
    private void LogClsSamplePredictions(Tensor logits, long[] labels, int batch, int globalStep, int epoch)
    {
        try
        {
            using var predCpu = logits.argmax(1).cpu();
            var predData = predCpu.data<long>().ToArray();
            var numSamples = Math.Min(3, batch);
            for (var i = 0; i < numSamples; i++)
            {
                var predClass = predData[i];
                var gtClass = labels[i];
                var isMatch = predClass == gtClass;
                _logger.LogInformation(
                    "sample_pred(cls) epoch={Epoch} step={Step} sample={SampleIdx} pred={Pred} gt={Gt} match={Match}",
                    epoch, globalStep, i, predClass, gtClass, isMatch);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to log cls sample predictions at step {Step}", globalStep);
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

internal sealed class SimpleClsNet : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _features;
    private readonly Module<Tensor, Tensor> _classifier;

    public SimpleClsNet(int numClasses) : base(nameof(SimpleClsNet))
    {
        _features = Sequential(
            ("conv1", Conv2d(3, 16, 3, stride: 1, padding: 1)),
            ("relu1", ReLU()),
            ("pool1", MaxPool2d(2)),
            ("conv2", Conv2d(16, 32, 3, stride: 1, padding: 1)),
            ("relu2", ReLU()),
            ("pool2", MaxPool2d(2)),
            ("conv3", Conv2d(32, 64, 3, stride: 1, padding: 1)),
            ("relu3", ReLU()),
            ("pool3", AdaptiveAvgPool2d([1, 1]))
        );
        _classifier = Linear(64, numClasses);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var feat = _features.call(input);
        using var flat = feat.flatten(1);
        return _classifier.call(flat);
    }
}
