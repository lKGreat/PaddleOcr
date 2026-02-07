using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Training.Rec.Losses;
using PaddleOcr.Training.Rec.Schedulers;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// ConfigDrivenRecTrainer：配置驱动的 Rec 训练器，支持完整的模型架构、损失函数、学习率调度等。
/// </summary>
internal sealed class ConfigDrivenRecTrainer
{
    private readonly ILogger _logger;

    public ConfigDrivenRecTrainer(ILogger logger)
    {
        _logger = logger;
    }

    public TrainingSummary Train(TrainingConfigView cfg)
    {
        var shape = cfg.RecImageShape;
        var (charToId, vocab) = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        EnsureCharsetCoverage(cfg.TrainLabelFile, cfg.EvalLabelFile, charToId, _logger);

        var useMultiScale = cfg.UseMultiScale;
        SimpleRecDataset? trainSet = null;
        MultiScaleRecDataset? trainSetMs = null;
        if (useMultiScale)
        {
            trainSetMs = new MultiScaleRecDataset(cfg.TrainLabelFile, cfg.DataDir, shape.H, cfg.MultiScaleWidths, cfg.MaxTextLength, charToId, enableAugmentation: true);
            _logger.LogInformation("Using MultiScaleDataSet with widths: [{Widths}]", string.Join(", ", cfg.MultiScaleWidths));
        }
        else
        {
            trainSet = new SimpleRecDataset(cfg.TrainLabelFile, cfg.DataDir, shape.H, shape.W, cfg.MaxTextLength, charToId, enableAugmentation: true);
        }
        var trainCount = useMultiScale ? trainSetMs!.Count : trainSet!.Count;
        var evalSet = new SimpleRecDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);

        var dev = ResolveDevice(cfg);
        _logger.LogInformation("Training(rec) device: {Device}", dev.type);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}, Vocab: {Vocab}", trainCount, evalSet.Count, vocab.Count);

        // 从配置构建模型
        var model = BuildModel(cfg, vocab.Count + 1);
        model.to(dev);

        // 从配置构建优化器
        var optimizer = BuildOptimizer(cfg, model);

        // 从配置构建学习率调度器
        var lrScheduler = BuildLRScheduler(cfg);

        // 从配置构建损失函数
        var lossFn = BuildLoss(cfg);

        // 训练辅助工具
        var ckptManager = new CheckpointManager(_logger);
        var ampHelper = cfg.Device.Contains("cuda", StringComparison.OrdinalIgnoreCase) ? new AmpTrainingHelper(dev) : null;
        var modelAverager = new ModelAverager();
        var gradAccumulator = new GradientUtils.GradientAccumulator(GetGradAccumulationSteps(cfg));

        var rng = new Random(cfg.Seed);
        float bestAcc = -1f;
        var epochsCompleted = 0;
        var staleEpochs = 0;
        var earlyStopped = false;
        var globalStep = 0;
        var startEpoch = 1;

        // 加载 checkpoint（完整恢复：模型 + 优化器 + 调度器 + 元信息）
        if (cfg.ResumeTraining)
        {
            var meta = ckptManager.LoadFull(cfg.SaveModelDir, "latest", model, optimizer, lrScheduler);
            if (meta is not null)
            {
                startEpoch = meta.Epoch + 1;
                globalStep = meta.GlobalStep;
                bestAcc = meta.BestAcc;
                _logger.LogInformation("Resumed training from epoch {Epoch}, step {Step}, best_acc {Acc:F4}",
                    startEpoch, globalStep, bestAcc);
            }
            else
            {
                // 兼容旧 checkpoint 格式（仅模型）
                var resumeCkpt = ResolveEvalCheckpoint(cfg);
                if (!string.IsNullOrWhiteSpace(resumeCkpt))
                {
                    TryLoadCheckpoint(model, resumeCkpt);
                }
            }
        }

        Directory.CreateDirectory(cfg.SaveModelDir);

        for (var epoch = startEpoch; epoch <= cfg.EpochNum; epoch++)
        {
            model.train();
            var lossSum = 0f;
            var sampleCount = 0;

            // 统一的 batch 迭代：支持 SimpleRecDataset 和 MultiScaleRecDataset
            IEnumerable<(float[] Images, long[] Labels, int Batch, int Width)> batchIter;
            if (useMultiScale)
            {
                batchIter = trainSetMs!.GetBatches(cfg.BatchSize, shuffle: true, rng)
                    .Select(b => (b.Images, b.Labels, b.Batch, b.Width));
            }
            else
            {
                batchIter = trainSet!.GetBatches(cfg.BatchSize, shuffle: true, rng)
                    .Select(b => (b.Images, b.Labels, b.Batch, shape.W));
            }

            foreach (var (images, labels, batch, batchW) in batchIter)
            {
                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, shape.H, batchW).to(dev);
                using var y = torch.tensor(labels, dtype: ScalarType.Int64).reshape(batch, cfg.MaxTextLength).to(dev);

                optimizer.zero_grad();

                // 混合精度前向传播
                using var autocast = ampHelper?.Autocast();
                var predictions = model.ForwardDict(x, new Dictionary<string, Tensor> { ["label"] = y });
                var batchDict = new Dictionary<string, Tensor> { ["label"] = y };
                var lossDict = lossFn.Forward(predictions, batchDict);
                var loss = lossDict["loss"];

                // NaN/Inf 守卫：检查 loss 是否有效
                var lossVal = loss.ToSingle();
                if (float.IsNaN(lossVal) || float.IsInfinity(lossVal))
                {
                    _logger.LogWarning("NaN/Inf loss detected at step {Step}, skipping this batch", globalStep);
                    globalStep++;
                    foreach (var t in predictions.Values) t.Dispose();
                    foreach (var t in lossDict.Values) t.Dispose();
                    continue;
                }

                // AMP: 缩放 loss 防止 float16 梯度下溢
                var scaledLoss = ampHelper is not null ? ampHelper.ScaleLoss(loss) : loss;
                scaledLoss.backward();
                if (!ReferenceEquals(scaledLoss, loss)) scaledLoss.Dispose();

                // AMP: 反缩放梯度并检查 inf/nan
                var gradsOk = ampHelper?.UnscaleAndCheck(model) ?? true;

                // 梯度裁剪
                if (gradsOk && cfg.GradClipNorm > 0f)
                {
                    GradientUtils.ClipGradNorm(model, cfg.GradClipNorm);
                }

                // 梯度累积 + 优化器更新
                if (gradsOk && gradAccumulator.ShouldUpdate())
                {
                    optimizer.step();
                    optimizer.zero_grad();
                }

                // AMP: 更新 scaler 状态
                ampHelper?.Update();

                lossSum += lossVal * batch;
                sampleCount += batch;
                globalStep++;

                // 更新学习率（仅在梯度更新时）
                if (gradAccumulator.ShouldUpdate() || globalStep == 1)
                {
                    lrScheduler.Step(globalStep, epoch);
                    // 通过 param_groups 更新优化器学习率
                    ApplyLearningRate(optimizer, lrScheduler.CurrentLR);
                }
            }

            // 模型平均（用于 SRN 等）
            if (ShouldUseModelAveraging(cfg))
            {
                modelAverager.Update(model);
            }

            var trainLoss = sampleCount == 0 ? 0f : lossSum / sampleCount;
            var evalMetrics = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, cfg.MaxTextLength, dev, vocab);

            if (evalMetrics.Accuracy > bestAcc + cfg.MinImproveDelta)
            {
                bestAcc = evalMetrics.Accuracy;
                staleEpochs = 0;
                ckptManager.SaveFull(cfg.SaveModelDir, "best", model, optimizer, lrScheduler, epoch, globalStep, bestAcc);
            }
            else
            {
                staleEpochs++;
            }

            ckptManager.SaveFull(cfg.SaveModelDir, "latest", model, optimizer, lrScheduler, epoch, globalStep, bestAcc);
            _logger.LogInformation(
                "epoch={Epoch}/{Total} train_loss={Loss:F4} eval_acc={EvalAcc:F4} eval_char_acc={CharAcc:F4} eval_edit={Edit:F4} lr={Lr:F6}",
                epoch, cfg.EpochNum, trainLoss, evalMetrics.Accuracy, evalMetrics.CharacterAccuracy, evalMetrics.AvgEditDistance, lrScheduler.CurrentLR);

            epochsCompleted = epoch;

            if (cfg.EarlyStopPatience > 0 && staleEpochs >= cfg.EarlyStopPatience)
            {
                earlyStopped = true;
                _logger.LogInformation("early stop triggered at epoch {Epoch} (patience={Patience})", epoch, cfg.EarlyStopPatience);
                break;
            }
        }

        // 应用模型平均
        if (ShouldUseModelAveraging(cfg))
        {
            modelAverager.Apply(model);
            ckptManager.SaveModel(cfg.SaveModelDir, model, "best_averaged.pt");
        }

        ampHelper?.Dispose();
        optimizer.Dispose();
        model.Dispose();
        var summary = new TrainingSummary(epochsCompleted, bestAcc, cfg.SaveModelDir);
        SaveSummary(cfg, summary, earlyStopped, cfg.ResumeTraining ? cfg.SaveModelDir : null);
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        var shape = cfg.RecImageShape;
        var (charToId, vocab) = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        var evalSet = new SimpleRecDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);

        var dev = ResolveDevice(cfg);
        var model = BuildModel(cfg, vocab.Count + 1);
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt))
        {
            TryLoadCheckpoint(model, ckpt);
        }

        var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, cfg.MaxTextLength, dev, vocab);
        model.Dispose();

        var summary = new EvaluationSummary(metrics.Accuracy, evalSet.Count);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "eval_result.json"),
            JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true }));
        _logger.LogInformation(
            "rec eval_acc={EvalAcc:F4} char_acc={CharAcc:F4} avg_edit={Edit:F4} samples={Samples}",
            metrics.Accuracy, metrics.CharacterAccuracy, metrics.AvgEditDistance, summary.Samples);
        return summary;
    }

    private RecModel BuildModel(TrainingConfigView cfg, int numClasses)
    {
        var backboneName = cfg.GetArchitectureString("Backbone.name", "MobileNetV1Enhance");
        var neckName = cfg.GetArchitectureString("Neck.name", "SequenceEncoder");
        var headName = cfg.GetArchitectureString("Head.name", "CTCHead");
        var hiddenSize = cfg.GetArchitectureInt("Head.hidden_size", 48);
        var maxLen = cfg.MaxTextLength;
        var inChannels = cfg.GetArchitectureInt("in_channels", 3);

        _logger.LogInformation("Building model: backbone={Backbone}, neck={Neck}, head={Head}, num_classes={Classes}, hidden_size={Hidden}, max_len={MaxLen}",
            backboneName, neckName, headName, numClasses, hiddenSize, maxLen);

        return RecModelBuilder.Build(backboneName, neckName, headName, numClasses, inChannels, hiddenSize, maxLen);
    }

    private Optimizer BuildOptimizer(TrainingConfigView cfg, RecModel model)
    {
        var optName = cfg.GetOptimizerString("name", "Adam");
        var lr = cfg.LearningRate;
        var beta1 = cfg.GetOptimizerFloat("beta1", 0.9f);
        var beta2 = cfg.GetOptimizerFloat("beta2", 0.999f);
        var weightDecay = cfg.GetOptimizerFloat("weight_decay", 0f);

        return optName.ToLowerInvariant() switch
        {
            "adam" => Adam(model.parameters(), lr: lr, beta1: beta1, beta2: beta2, weight_decay: weightDecay),
            "adamw" => AdamW(model.parameters(), lr: lr, beta1: beta1, beta2: beta2, weight_decay: weightDecay),
            "sgd" => SGD(model.parameters(), lr, momentum: 0.9f, weight_decay: weightDecay),
            _ => Adam(model.parameters(), lr: lr)
        };
    }

    private ILRScheduler BuildLRScheduler(TrainingConfigView cfg)
    {
        var lrConfig = cfg.GetOptimizerLrConfig();
        var lrName = lrConfig.TryGetValue("name", out var name) ? name.ToString() ?? "Cosine" : "Cosine";
        var baseLr = cfg.LearningRate;

        return LRSchedulerBuilder.Build(lrName, lrConfig);
    }

    private IRecLoss BuildLoss(TrainingConfigView cfg)
    {
        var lossName = cfg.GetLossString("name", "CTCLoss");
        var lossConfig = cfg.GetLossConfig();
        return RecLossBuilder.Build(lossName, lossConfig);
    }

    private static RecEvalMetrics Evaluate(RecModel model, SimpleRecDataset evalSet, int batchSize, int h, int w, int maxTextLength, Device dev, IReadOnlyList<char> vocab)
    {
        model.eval();
        long correct = 0;
        var total = 0L;
        var charTotal = 0L;
        var charErrors = 0L;
        var editSum = 0L;
        using var noGrad = torch.no_grad();
        foreach (var (images, labels, batch) in evalSet.GetBatches(batchSize, shuffle: false, new Random(7)))
        {
            using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, h, w).to(dev);
            using var y = torch.tensor(labels, dtype: ScalarType.Int64).reshape(batch, maxTextLength).to(dev);
            var predictions = model.ForwardDict(x);
            var logits = predictions.ContainsKey("predict") ? predictions["predict"] : predictions.Values.First();

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
                var edit = Levenshtein(predText, gtText);
                editSum += edit;
                charErrors += edit;
                charTotal += gtText.Length;
                total++;
            }
        }

        var acc = total == 0 ? 0f : (float)correct / total;
        var charAcc = charTotal == 0 ? 0f : 1f - (float)charErrors / charTotal;
        var avgEdit = total == 0 ? 0f : (float)editSum / total;
        return new RecEvalMetrics(acc, charAcc, avgEdit);
    }

    private static int Levenshtein(string left, string right)
    {
        if (left.Length == 0)
        {
            return right.Length;
        }

        if (right.Length == 0)
        {
            return left.Length;
        }

        var prev = new int[right.Length + 1];
        var curr = new int[right.Length + 1];
        for (var j = 0; j <= right.Length; j++)
        {
            prev[j] = j;
        }

        for (var i = 1; i <= left.Length; i++)
        {
            curr[0] = i;
            for (var j = 1; j <= right.Length; j++)
            {
                var cost = left[i - 1] == right[j - 1] ? 0 : 1;
                curr[j] = Math.Min(
                    Math.Min(curr[j - 1] + 1, prev[j] + 1),
                    prev[j - 1] + cost);
            }

            (prev, curr) = (curr, prev);
        }

        return prev[right.Length];
    }

    private void SaveCheckpoint(string saveDir, RecModel model, string fileName)
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

    private void TryLoadCheckpoint(RecModel model, string checkpointPath)
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

    private static void SaveSummary(TrainingConfigView cfg, TrainingSummary summary, bool earlyStopped, string? resumeCheckpoint)
    {
        Directory.CreateDirectory(cfg.SaveModelDir);
        var json = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(cfg.SaveModelDir, "train_result.json"), json);
        var run = new TrainingRunSummary(
            ModelType: "rec",
            EpochsRequested: cfg.EpochNum,
            EpochsCompleted: summary.Epochs,
            BestMetricName: "accuracy",
            BestMetricValue: summary.BestAccuracy,
            EarlyStopped: earlyStopped,
            SaveDir: cfg.SaveModelDir,
            ResumeCheckpoint: resumeCheckpoint,
            GeneratedAtUtc: DateTime.UtcNow);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "train_run_summary.json"),
            JsonSerializer.Serialize(run, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void EnsureCharsetCoverage(string trainLabelFile, string evalLabelFile, IReadOnlyDictionary<char, int> charToId, ILogger logger)
    {
        var missing = new HashSet<char>();
        foreach (var labelFile in new[] { trainLabelFile, evalLabelFile })
        {
            if (!File.Exists(labelFile))
            {
                continue;
            }

            foreach (var line in File.ReadLines(labelFile))
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var split = line.Split('\t', 2);
                if (split.Length < 2)
                {
                    continue;
                }

                foreach (var ch in split[1])
                {
                    if (!charToId.ContainsKey(ch))
                    {
                        missing.Add(ch);
                    }
                }
            }
        }

        if (missing.Count > 0)
        {
            var preview = new string(missing.Take(32).ToArray());
            logger.LogWarning("rec charset missing {Count} chars from labels. example: {Chars}", missing.Count, preview);
        }
    }

    private static Device ResolveDevice(TrainingConfigView cfg)
    {
        if (cfg.Device.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            return CPU;
        }

        return cuda.is_available() ? CUDA : CPU;
    }

    /// <summary>
    /// 通过 TorchSharp 优化器的 param_groups 更新学习率。
    /// </summary>
    private static void ApplyLearningRate(Optimizer optimizer, double lr)
    {
        foreach (var pg in optimizer.ParamGroups)
        {
            pg.LearningRate = lr;
        }
    }

    private static int GetGradAccumulationSteps(TrainingConfigView cfg)
    {
        // 从配置读取梯度累积步数，默认 1
        return cfg.GetConfigInt("Optimizer.grad_accumulation_steps", 1);
    }

    private static bool ShouldUseModelAveraging(TrainingConfigView cfg)
    {
        // 从配置读取是否使用模型平均，默认 false
        return cfg.GetConfigBool("Global.use_model_averaging", false);
    }
}
