using System.Text.Json;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PaddleOcr.Training;

internal sealed class SimpleRecTrainer
{
    private readonly ILogger _logger;

    public SimpleRecTrainer(ILogger logger)
    {
        _logger = logger;
    }

    public TrainingSummary Train(TrainingConfigView cfg)
    {
        var shape = cfg.RecImageShape;
        var (charToId, vocab) = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        var trainSet = new SimpleRecDataset(cfg.TrainLabelFile, cfg.DataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);
        var evalSet = new SimpleRecDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);

        var dev = cuda.is_available() ? CUDA : CPU;
        _logger.LogInformation("Training(rec) device: {Device}", dev.type);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}, Vocab: {Vocab}", trainSet.Count, evalSet.Count, vocab.Count);

        using var model = new SimpleRecNet(vocab.Count + 1, cfg.MaxTextLength);
        model.to(dev);
        using var optimizer = torch.optim.Adam(model.parameters(), lr: cfg.LearningRate);

        var rng = new Random(1024);
        float bestAcc = -1f;
        for (var epoch = 1; epoch <= cfg.EpochNum; epoch++)
        {
            model.train();
            var lossSum = 0f;
            var sampleCount = 0;
            foreach (var (images, labels, batch) in trainSet.GetBatches(cfg.BatchSize, shuffle: true, rng))
            {
                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, shape.H, shape.W).to(dev);
                using var y = torch.tensor(labels, dtype: ScalarType.Int64).reshape(batch, cfg.MaxTextLength).to(dev);
                optimizer.zero_grad();
                using var logits = model.call(x); // [B,T,V]
                using var loss = functional.cross_entropy(logits.reshape(batch * cfg.MaxTextLength, vocab.Count + 1), y.reshape(batch * cfg.MaxTextLength));
                loss.backward();
                optimizer.step();

                lossSum += loss.ToSingle() * batch;
                sampleCount += batch;
            }

            var trainLoss = sampleCount == 0 ? 0f : lossSum / sampleCount;
            var evalAcc = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, cfg.MaxTextLength, dev, vocab);

            if (evalAcc > bestAcc)
            {
                bestAcc = evalAcc;
                SaveCheckpoint(cfg.SaveModelDir, model, "best.pt");
            }

            SaveCheckpoint(cfg.SaveModelDir, model, "latest.pt");
            _logger.LogInformation("epoch={Epoch}/{Total} train_loss={Loss:F4} eval_acc={EvalAcc:F4}", epoch, cfg.EpochNum, trainLoss, evalAcc);
        }

        var summary = new TrainingSummary(cfg.EpochNum, bestAcc, cfg.SaveModelDir);
        SaveSummary(cfg.SaveModelDir, summary);
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        var shape = cfg.RecImageShape;
        var (charToId, vocab) = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        var evalSet = new SimpleRecDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);

        var dev = cuda.is_available() ? CUDA : CPU;
        using var model = new SimpleRecNet(vocab.Count + 1, cfg.MaxTextLength);
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt))
        {
            TryLoadCheckpoint(model, ckpt);
        }

        var acc = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, cfg.MaxTextLength, dev, vocab);
        var summary = new EvaluationSummary(acc, evalSet.Count);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "eval_result.json"),
            JsonSerializer.Serialize(new { accuracy = acc, samples = evalSet.Count }, new JsonSerializerOptions { WriteIndented = true }));
        _logger.LogInformation("rec eval_acc={EvalAcc:F4} samples={Samples}", summary.Accuracy, summary.Samples);
        return summary;
    }

    private static float Evaluate(SimpleRecNet model, SimpleRecDataset evalSet, int batchSize, int h, int w, int maxTextLength, Device dev, IReadOnlyList<char> vocab)
    {
        model.eval();
        long correct = 0;
        var total = 0;
        using var noGrad = torch.no_grad();
        foreach (var (images, labels, batch) in evalSet.GetBatches(batchSize, shuffle: false, new Random(7)))
        {
            using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, h, w).to(dev);
            using var y = torch.tensor(labels, dtype: ScalarType.Int64).reshape(batch, maxTextLength).to(dev);
            using var logits = model.call(x);
            using var pred = logits.argmax(2).to_type(ScalarType.Int64).cpu();
            using var gt = y.cpu();

            var predFlat = pred.data<long>().ToArray();
            var gtFlat = gt.data<long>().ToArray();
            for (var i = 0; i < batch; i++)
            {
                var predSeq = predFlat.Skip(i * maxTextLength).Take(maxTextLength).ToArray();
                var gtSeq = gtFlat.Skip(i * maxTextLength).Take(maxTextLength).ToArray();
                var predText = SimpleRecDataset.Decode(predSeq, vocab);
                var gtText = SimpleRecDataset.Decode(gtSeq, vocab);
                if (string.Equals(predText, gtText, StringComparison.Ordinal))
                {
                    correct++;
                }
                total++;
            }
        }

        return total == 0 ? 0f : (float)correct / total;
    }

    private void SaveCheckpoint(string saveDir, SimpleRecNet model, string fileName)
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

    private void TryLoadCheckpoint(SimpleRecNet model, string checkpointPath)
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

    private static void SaveSummary(string saveDir, TrainingSummary summary)
    {
        Directory.CreateDirectory(saveDir);
        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(saveDir, "train_result.json"), json);
    }
}

internal sealed class SimpleRecNet : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _features;
    private readonly Module<Tensor, Tensor> _head;
    private readonly int _maxTextLength;
    private readonly int _numClasses;

    public SimpleRecNet(int numClasses, int maxTextLength) : base(nameof(SimpleRecNet))
    {
        _numClasses = numClasses;
        _maxTextLength = maxTextLength;
        _features = Sequential(
            ("conv1", Conv2d(3, 32, 3, stride: 1, padding: 1)),
            ("relu1", ReLU()),
            ("pool1", MaxPool2d(2)),
            ("conv2", Conv2d(32, 64, 3, stride: 1, padding: 1)),
            ("relu2", ReLU()),
            ("pool2", MaxPool2d(2)),
            ("conv3", Conv2d(64, 128, 3, stride: 1, padding: 1)),
            ("relu3", ReLU()),
            ("pool3", AdaptiveAvgPool2d([1, maxTextLength]))
        );
        _head = Linear(128, numClasses);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var feat = _features.call(input); // [B,128,1,T]
        using var squeezed = feat.squeeze(2).transpose(1, 2); // [B,T,128]
        var flat = squeezed.reshape(-1, 128);
        using var logitsFlat = _head.call(flat);
        return logitsFlat.reshape(input.shape[0], _maxTextLength, _numClasses);
    }
}
