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
        _logger.LogInformation("Training device: {Device}", dev.type);
        _logger.LogInformation("runtime: requested={Requested}, cuda={Cuda}, amp={Amp}, reason={Reason}", runtime.RequestedDevice, runtime.UseCuda, runtime.UseAmp, runtime.Reason);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}", trainSet.Count, evalSet.Count);

        using var model = new SimpleClsNet(numClasses);
        model.to(dev);
        var lr = cfg.LearningRate;
        var resumeCkpt = cfg.ResumeTraining ? ResolveEvalCheckpoint(cfg) : null;
        if (!string.IsNullOrWhiteSpace(resumeCkpt))
        {
            TryLoadCheckpoint(model, resumeCkpt);
        }
        var optimizer = torch.optim.Adam(model.parameters(), lr: lr);

        var rng = new Random(1024);
        float bestAcc = -1f;
        var epochsCompleted = 0;
        var staleEpochs = 0;
        var earlyStopped = false;
        for (var epoch = 1; epoch <= cfg.EpochNum; epoch++)
        {
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
            foreach (var (images, labels, batch) in trainSet.GetBatches(cfg.BatchSize, shuffle: true, rng))
            {
                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, shape.H, shape.W).to(dev);
                using var y = torch.tensor(labels, dtype: ScalarType.Int64).to(dev);
                optimizer.zero_grad();
                using var logits = model.call(x);
                using var loss = functional.cross_entropy(logits, y);
                loss.backward();
                if (cfg.GradClipNorm > 0f)
                {
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GradClipNorm);
                }
                optimizer.step();

                lossSum += loss.ToSingle() * batch;
                using var pred = logits.argmax(1);
                using var eq = pred.eq(y);
                correct += eq.sum().ToInt64();
                sampleCount += batch;
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
            _logger.LogInformation("epoch={Epoch}/{Total} train_loss={Loss:F4} train_acc={TrainAcc:F4} eval_acc={EvalAcc:F4}", epoch, cfg.EpochNum, trainLoss, trainAcc, evalAcc);
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
