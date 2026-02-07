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
        var size = cfg.DetInputSize;
        var trainSet = new SimpleDetDataset(cfg.TrainLabelFile, cfg.DataDir, size);
        var evalSet = new SimpleDetDataset(cfg.EvalLabelFile, cfg.EvalDataDir, size);

        var dev = cuda.is_available() ? CUDA : CPU;
        using var model = new SimpleDetNet();
        model.to(dev);
        var lr = cfg.LearningRate;
        var optimizer = torch.optim.Adam(model.parameters(), lr: lr);
        var rng = new Random(1024);
        Directory.CreateDirectory(cfg.SaveModelDir);
        var resumeCkpt = cfg.ResumeTraining ? ResolveEvalCheckpoint(cfg) : null;
        if (!string.IsNullOrWhiteSpace(resumeCkpt))
        {
            _logger.LogInformation("Loading checkpoint: {Path}", resumeCkpt);
            model.load(resumeCkpt);
        }

        float bestFscore = -1f;
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
            var samples = 0;
            foreach (var (images, masks, batch) in trainSet.GetBatches(cfg.BatchSize, true, rng))
            {
                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, size, size).to(dev);
                using var y = torch.tensor(masks, dtype: ScalarType.Float32).reshape(batch, 1, size, size).to(dev);
                optimizer.zero_grad();
                using var pred = model.call(x);
                using var loss = functional.binary_cross_entropy_with_logits(pred, y);
                loss.backward();
                optimizer.step();
                lossSum += loss.ToSingle() * batch;
                samples += batch;
            }

            var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, size, dev);
            if (metrics.Fscore > bestFscore)
            {
                bestFscore = metrics.Fscore;
                staleEpochs = 0;
                model.save(Path.Combine(cfg.SaveModelDir, "best.pt"));
            }
            else
            {
                staleEpochs++;
            }

            model.save(Path.Combine(cfg.SaveModelDir, "latest.pt"));
            _logger.LogInformation(
                "epoch={Epoch}/{Total} train_loss={Loss:F4} eval_p={P:F4} eval_r={R:F4} eval_f={F:F4} eval_iou={IoU:F4}",
                epoch, cfg.EpochNum, lossSum / Math.Max(1, samples), metrics.Precision, metrics.Recall, metrics.Fscore, metrics.Iou);
            epochsCompleted = epoch;
            if (cfg.EarlyStopPatience > 0 && staleEpochs >= cfg.EarlyStopPatience)
            {
                earlyStopped = true;
                _logger.LogInformation("early stop triggered at epoch {Epoch} (patience={Patience})", epoch, cfg.EarlyStopPatience);
                break;
            }
        }

        optimizer.Dispose();
        var summary = new TrainingSummary(epochsCompleted, bestFscore, cfg.SaveModelDir);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(Path.Combine(cfg.SaveModelDir, "train_result.json"), System.Text.Json.JsonSerializer.Serialize(summary));
        var run = new TrainingRunSummary(
            ModelType: "det",
            EpochsRequested: cfg.EpochNum,
            EpochsCompleted: summary.Epochs,
            BestMetricName: "fscore",
            BestMetricValue: summary.BestAccuracy,
            EarlyStopped: earlyStopped,
            SaveDir: cfg.SaveModelDir,
            ResumeCheckpoint: resumeCkpt,
            GeneratedAtUtc: DateTime.UtcNow);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "train_run_summary.json"),
            System.Text.Json.JsonSerializer.Serialize(run, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        var size = cfg.DetInputSize;
        var evalSet = new SimpleDetDataset(cfg.EvalLabelFile, cfg.EvalDataDir, size);
        var dev = cuda.is_available() ? CUDA : CPU;
        using var model = new SimpleDetNet();
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (File.Exists(ckpt))
        {
            _logger.LogInformation("Loading checkpoint: {Path}", ckpt);
            model.load(ckpt);
        }

        var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, size, dev);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "eval_result.json"),
            System.Text.Json.JsonSerializer.Serialize(metrics, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        _logger.LogInformation(
            "det eval metrics: precision={P:F4}, recall={R:F4}, fscore={F:F4}, iou={IoU:F4}",
            metrics.Precision, metrics.Recall, metrics.Fscore, metrics.Iou);
        return new EvaluationSummary(metrics.Iou, evalSet.Count);
    }

    private static DetEvalMetrics Evaluate(SimpleDetNet model, SimpleDetDataset evalSet, int batchSize, int size, Device dev)
    {
        model.eval();
        var interSum = 0f;
        var unionSum = 0f;
        var predSum = 0f;
        var gtSum = 0f;
        using var noGrad = torch.no_grad();
        foreach (var (images, masks, batch) in evalSet.GetBatches(batchSize, false, new Random(7)))
        {
            using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, size, size).to(dev);
            using var y = torch.tensor(masks, dtype: ScalarType.Float32).reshape(batch, 1, size, size).to(dev);
            using var pred = model.call(x).sigmoid();
            using var pb = pred.gt(0.5);
            using var yb = y.gt(0.5);
            using var inter = pb.logical_and(yb).sum();
            using var union = pb.logical_or(yb).sum();
            using var predArea = pb.sum();
            using var gtArea = yb.sum();
            interSum += inter.ToSingle();
            unionSum += union.ToSingle();
            predSum += predArea.ToSingle();
            gtSum += gtArea.ToSingle();
        }

        var iou = unionSum <= 0f ? 0f : interSum / unionSum;
        var precision = predSum <= 0f ? 0f : interSum / predSum;
        var recall = gtSum <= 0f ? 0f : interSum / gtSum;
        var f = precision + recall <= 0f ? 0f : 2f * precision * recall / (precision + recall);
        return new DetEvalMetrics(precision, recall, f, iou);
    }

    private static string ResolveEvalCheckpoint(TrainingConfigView cfg)
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

        return Path.Combine(cfg.SaveModelDir, "latest.pt");
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
        _head = Conv2d(32, 1, 1);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var feat = _backbone.call(input);
        return _head.call(feat);
    }
}

internal sealed record DetEvalMetrics(float Precision, float Recall, float Fscore, float Iou);
