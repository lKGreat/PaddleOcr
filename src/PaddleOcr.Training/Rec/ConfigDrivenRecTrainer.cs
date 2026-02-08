using System.Text.Json;
using System.Text;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using PaddleOcr.Data;
using PaddleOcr.Data.LabelEncoders;
using PaddleOcr.Training.Runtime;
using PaddleOcr.Training.Rec.Losses;
using PaddleOcr.Training.Rec.Schedulers;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// Config-driven rec trainer aligned with PaddleOCR tools training flow.
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
        var (charToId, _) = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        var trainLabelFiles = GetLabelFilesOrFallback(cfg.TrainLabelFiles, cfg.TrainLabelFile);
        var evalLabelFiles = GetLabelFilesOrFallback(cfg.EvalLabelFiles, cfg.EvalLabelFile);
        EnsureCharsetCoverage(trainLabelFiles, evalLabelFiles, charToId, _logger);

        var gtcEncodeType = ResolveGtcEncodeType(cfg);
        var ctcEncoder = new CTCLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar);
        var gtcEncoder = CreateGtcEncoder(gtcEncodeType, cfg);
        var resizeStrategy = RecTrainingResizeFactory.Create(cfg.GetArchitectureString("algorithm", "SVTR_LCNet"));

        var trainSet = new ConfigRecDataset(
            trainLabelFiles,
            cfg.DataDir,
            shape.H,
            shape.W,
            cfg.MaxTextLength,
            ctcEncoder,
            gtcEncoder,
            resizeStrategy,
            enableAugmentation: true,
            useMultiScale: cfg.UseMultiScale,
            multiScaleWidths: cfg.MultiScaleWidths);

        var evalSet = new ConfigRecDataset(
            evalLabelFiles,
            cfg.EvalDataDir,
            shape.H,
            shape.W,
            cfg.MaxTextLength,
            ctcEncoder,
            gtcEncoder,
            resizeStrategy,
            enableAugmentation: false,
            useMultiScale: false);

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        _logger.LogInformation("Training(rec) device: {Device}", dev.type);
        _logger.LogInformation("runtime: requested={Requested}, cuda={Cuda}, amp={Amp}, reason={Reason}", runtime.RequestedDevice, runtime.UseCuda, runtime.UseAmp, runtime.Reason);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}, Vocab: {Vocab}", trainSet.Count, evalSet.Count, ctcEncoder.NumClasses);

        var model = BuildModel(cfg, ctcEncoder.NumClasses, gtcEncodeType);
        model.to(dev);

        var optimizer = BuildOptimizer(cfg, model);
        var lrScheduler = BuildLRScheduler(cfg);
        var lossFn = BuildLoss(cfg);

        var ckptManager = new CheckpointManager(_logger);
        var ampHelper = runtime.UseAmp ? new AmpTrainingHelper(dev, enabled: true) : null;
        var modelAverager = new ModelAverager();
        var gradAccumulator = new GradientUtils.GradientAccumulator(GetGradAccumulationSteps(cfg));

        var rng = new Random(cfg.Seed);
        float bestAcc = -1f;
        var epochsCompleted = 0;
        var staleEpochs = 0;
        var earlyStopped = false;
        var globalStep = 0;
        var startEpoch = 1;
        var optimizerStepCount = 0;
        var nonZeroGradSteps = 0;
        var tracePath = Path.Combine(cfg.SaveModelDir, "train_trace.jsonl");
        var epochTracePath = Path.Combine(cfg.SaveModelDir, "train_epoch_summary.jsonl");
        var recentLoss = new Queue<float>(cfg.LogSmoothWindow);
        var stepWatch = Stopwatch.StartNew();

        if (cfg.ResumeTraining)
        {
            var meta = ckptManager.LoadFull(cfg.SaveModelDir, "latest", model, optimizer, lrScheduler);
            if (meta is not null)
            {
                startEpoch = meta.Epoch + 1;
                globalStep = meta.GlobalStep;
                bestAcc = meta.BestAcc;
                _logger.LogInformation("Resumed training from epoch {Epoch}, step {Step}, best_acc {Acc:F4}", startEpoch, globalStep, bestAcc);
            }
            else
            {
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
            optimizer.zero_grad();
            var lossSum = 0f;
            var sampleCount = 0;

            foreach (var batchData in trainSet.GetBatches(cfg.BatchSize, shuffle: true, rng))
            {
                using var x = torch.tensor(batchData.Images, dtype: ScalarType.Float32).reshape(batchData.Batch, 3, shape.H, batchData.Width).to(dev);
                using var yCtc = torch.tensor(batchData.LabelCtc, dtype: ScalarType.Int64).reshape(batchData.Batch, cfg.MaxTextLength).to(dev);
                using var yGtc = torch.tensor(batchData.LabelGtc, dtype: ScalarType.Int64).reshape(batchData.Batch, cfg.MaxTextLength).to(dev);
                using var targetLengths = torch.tensor(batchData.Lengths.Select(x => (long)x).ToArray(), dtype: ScalarType.Int64).to(dev);
                using var validRatio = torch.tensor(batchData.ValidRatios, dtype: ScalarType.Float32).to(dev);

                using var autocast = ampHelper?.Autocast();
                var predictions = model.ForwardDict(
                    x,
                    new Dictionary<string, Tensor>
                    {
                        ["label"] = yCtc,
                        ["label_ctc"] = yCtc,
                        ["label_gtc"] = yGtc,
                        ["valid_ratio"] = validRatio
                    });

                var ctcLogits = predictions.TryGetValue("ctc", out var ctcValue)
                    ? ctcValue
                    : (predictions.TryGetValue("predict", out var predValue) ? predValue : predictions.Values.First());
                var ctcTime = ctcLogits.shape.Length >= 2 ? ctcLogits.shape[1] : cfg.MaxTextLength;
                using var inputLengths = torch.full(new long[] { batchData.Batch }, ctcTime, dtype: ScalarType.Int64, device: dev);

                var lossDict = lossFn.Forward(
                    predictions,
                    new Dictionary<string, Tensor>
                    {
                        ["label"] = yCtc,
                        ["label_ctc"] = yCtc,
                        ["label_gtc"] = yGtc,
                        ["target_lengths"] = targetLengths,
                        ["input_lengths"] = inputLengths,
                        ["length"] = targetLengths,
                        ["valid_ratio"] = validRatio
                    });

                var loss = lossDict["loss"];
                var lossVal = loss.ToSingle();
                if (float.IsNaN(lossVal) || float.IsInfinity(lossVal))
                {
                    _logger.LogWarning("NaN/Inf loss detected at step {Step}, skipping this batch", globalStep);
                    globalStep++;
                    DisposeTensorDictionary(predictions);
                    DisposeTensorDictionary(lossDict);
                    continue;
                }

                var scaledLoss = ampHelper is not null ? ampHelper.ScaleLoss(loss) : loss;
                scaledLoss.backward();
                if (!ReferenceEquals(scaledLoss, loss))
                {
                    scaledLoss.Dispose();
                }

                var gradsOk = ampHelper?.UnscaleAndCheck(model) ?? true;
                if (gradsOk && cfg.GradClipNorm > 0f)
                {
                    GradientUtils.ClipGradNorm(model, cfg.GradClipNorm);
                }

                var shouldUpdate = gradAccumulator.ShouldUpdate();
                if (gradsOk && shouldUpdate)
                {
                    var gradNorm = EstimateGradNorm(model);
                    if (gradNorm > 0f)
                    {
                        nonZeroGradSteps++;
                    }
                    optimizer.step();
                    optimizerStepCount++;
                    optimizer.zero_grad();
                }

                ampHelper?.Update();

                lossSum += lossVal * batchData.Batch;
                sampleCount += batchData.Batch;
                globalStep++;

                if (cfg.SaveBatchModel)
                {
                    ckptManager.SaveModel(cfg.SaveModelDir, model, $"iter_step_{globalStep}.pt");
                }

                recentLoss.Enqueue(lossVal);
                while (recentLoss.Count > cfg.LogSmoothWindow)
                {
                    _ = recentLoss.Dequeue();
                }

                if (globalStep % cfg.PrintBatchStep == 0)
                {
                    var smoothedLoss = recentLoss.Count == 0 ? lossVal : recentLoss.Average();
                    var stepMs = stepWatch.Elapsed.TotalMilliseconds;
                    stepWatch.Restart();
                    AppendJsonLine(
                        tracePath,
                        new
                        {
                            epoch,
                            global_step = globalStep,
                            loss = lossVal,
                            smooth_loss = smoothedLoss,
                            lr = lrScheduler.CurrentLR,
                            batch = batchData.Batch,
                            width = batchData.Width,
                            step_ms = stepMs,
                            optimizer_step_count = optimizerStepCount,
                            non_zero_grad_steps = nonZeroGradSteps
                        });
                    _logger.LogInformation(
                        "train step epoch={Epoch} step={Step} loss={Loss:F4} smooth_loss={Smooth:F4} lr={Lr:F6} step_ms={StepMs:F1}",
                        epoch,
                        globalStep,
                        lossVal,
                        smoothedLoss,
                        lrScheduler.CurrentLR,
                        stepMs);
                }

                if (shouldUpdate || globalStep == 1)
                {
                    lrScheduler.Step(globalStep, epoch);
                    ApplyLearningRate(optimizer, lrScheduler.CurrentLR);
                }

                if (cfg.CalMetricDuringTrain && ShouldEvalByStep(cfg, globalStep))
                {
                    var stepEval = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, cfg.MaxTextLength, dev, ctcEncoder.Characters);
                    if (stepEval.Accuracy > bestAcc + cfg.MinImproveDelta)
                    {
                        bestAcc = stepEval.Accuracy;
                        ckptManager.SaveFull(cfg.SaveModelDir, "best", model, optimizer, lrScheduler, epoch, globalStep, bestAcc);
                    }

                    AppendJsonLine(
                        epochTracePath,
                        new
                        {
                            event_type = "step_eval",
                            epoch,
                            global_step = globalStep,
                            eval_acc = stepEval.Accuracy,
                            eval_char_acc = stepEval.CharacterAccuracy,
                            eval_edit = stepEval.AvgEditDistance,
                            best_acc = bestAcc
                        });
                    _logger.LogInformation(
                        "step eval epoch={Epoch} step={Step} eval_acc={EvalAcc:F4} best_acc={BestAcc:F4}",
                        epoch,
                        globalStep,
                        stepEval.Accuracy,
                        bestAcc);
                    model.train();
                }

                DisposeTensorDictionary(predictions);
                DisposeTensorDictionary(lossDict);
            }

            if (ShouldUseModelAveraging(cfg))
            {
                modelAverager.Update(model);
            }

            var trainLoss = sampleCount == 0 ? 0f : lossSum / sampleCount;
            var evalMetrics = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, cfg.MaxTextLength, dev, ctcEncoder.Characters);

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
            if (epoch % cfg.SaveEpochStep == 0)
            {
                ckptManager.SaveFull(cfg.SaveModelDir, $"epoch_{epoch}", model, optimizer, lrScheduler, epoch, globalStep, bestAcc);
            }
            _logger.LogInformation(
                "epoch={Epoch}/{Total} train_loss={Loss:F4} eval_acc={EvalAcc:F4} eval_char_acc={CharAcc:F4} eval_edit={Edit:F4} lr={Lr:F6}",
                epoch,
                cfg.EpochNum,
                trainLoss,
                evalMetrics.Accuracy,
                evalMetrics.CharacterAccuracy,
                evalMetrics.AvgEditDistance,
                lrScheduler.CurrentLR);
            AppendJsonLine(
                epochTracePath,
                new
                {
                    epoch,
                    total_epoch = cfg.EpochNum,
                    train_loss = trainLoss,
                    eval_acc = evalMetrics.Accuracy,
                    eval_char_acc = evalMetrics.CharacterAccuracy,
                    eval_edit = evalMetrics.AvgEditDistance,
                    lr = lrScheduler.CurrentLR,
                    optimizer_step_count = optimizerStepCount,
                    non_zero_grad_steps = nonZeroGradSteps
                });

            epochsCompleted = epoch;

            if (cfg.EarlyStopPatience > 0 && staleEpochs >= cfg.EarlyStopPatience)
            {
                earlyStopped = true;
                _logger.LogInformation("early stop triggered at epoch {Epoch} (patience={Patience})", epoch, cfg.EarlyStopPatience);
                break;
            }
        }

        if (ShouldUseModelAveraging(cfg))
        {
            modelAverager.Apply(model);
            ckptManager.SaveModel(cfg.SaveModelDir, model, "best_averaged.pt");
        }

        ampHelper?.Dispose();
        optimizer.Dispose();
        model.Dispose();

        var summary = new TrainingSummary(epochsCompleted, bestAcc, cfg.SaveModelDir);
        SaveSummary(cfg, summary, earlyStopped, cfg.ResumeTraining ? cfg.SaveModelDir : null, optimizerStepCount, nonZeroGradSteps);
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        var shape = cfg.RecImageShape;
        _ = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        var gtcEncodeType = ResolveGtcEncodeType(cfg);
        var ctcEncoder = new CTCLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar);
        var gtcEncoder = CreateGtcEncoder(gtcEncodeType, cfg);
        var resizeStrategy = RecTrainingResizeFactory.Create(cfg.GetArchitectureString("algorithm", "SVTR_LCNet"));

        var evalSet = new ConfigRecDataset(
            GetLabelFilesOrFallback(cfg.EvalLabelFiles, cfg.EvalLabelFile),
            cfg.EvalDataDir,
            shape.H,
            shape.W,
            cfg.MaxTextLength,
            ctcEncoder,
            gtcEncoder,
            resizeStrategy,
            enableAugmentation: false);

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        var model = BuildModel(cfg, ctcEncoder.NumClasses, gtcEncodeType);
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt))
        {
            TryLoadCheckpoint(model, ckpt);
        }

        var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, cfg.MaxTextLength, dev, ctcEncoder.Characters);
        model.Dispose();

        var summary = new EvaluationSummary(metrics.Accuracy, evalSet.Count);
        Directory.CreateDirectory(cfg.SaveModelDir);
        File.WriteAllText(
            Path.Combine(cfg.SaveModelDir, "eval_result.json"),
            JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true }));
        _logger.LogInformation(
            "rec eval_acc={EvalAcc:F4} char_acc={CharAcc:F4} avg_edit={Edit:F4} samples={Samples}",
            metrics.Accuracy,
            metrics.CharacterAccuracy,
            metrics.AvgEditDistance,
            summary.Samples);
        return summary;
    }

    private RecModel BuildModel(TrainingConfigView cfg, int numClasses, string? gtcEncodeType)
    {
        var backboneName = cfg.GetArchitectureString("Backbone.name", "MobileNetV1Enhance");
        var resolvedBackboneName = ResolveBackboneAlias(backboneName);
        var neckName = cfg.GetArchitectureString("Neck.name", "SequenceEncoder");
        var neckEncoderType = ResolveNeckEncoderType(cfg);
        var headName = cfg.GetArchitectureString("Head.name", "CTCHead");
        var hiddenSize = cfg.GetArchitectureInt("Head.hidden_size", 48);
        hiddenSize = ResolveGtcHiddenSize(cfg, hiddenSize);
        var maxLen = cfg.MaxTextLength;
        var inChannels = cfg.GetArchitectureInt("in_channels", 3);

        var gtcHeadName = ResolveGtcHeadName(cfg);
        var gtcOutChannels = ResolveGtcOutChannels(numClasses, gtcHeadName, gtcEncodeType);

        _logger.LogInformation(
                "Building model: backbone={Backbone}, neck={Neck}({NeckEncoder}), head={Head}, num_classes={Classes}, gtc_head={GtcHead}, gtc_classes={GtcClasses}",
                resolvedBackboneName,
                neckName,
                neckEncoderType,
                headName,
                numClasses,
                gtcHeadName ?? "none",
                gtcOutChannels);

        if (!string.Equals(backboneName, resolvedBackboneName, StringComparison.OrdinalIgnoreCase))
        {
            _logger.LogWarning("Backbone '{Backbone}' is not implemented, using compatible fallback '{Fallback}'", backboneName, resolvedBackboneName);
        }

        return RecModelBuilder.Build(
            resolvedBackboneName,
            neckName,
            headName,
            numClasses,
            inChannels,
            hiddenSize,
            maxLen,
            neckEncoderType,
            gtcHeadName,
            gtcOutChannels);
    }

    private Optimizer BuildOptimizer(TrainingConfigView cfg, RecModel model)
    {
        var optName = cfg.GetOptimizerString("name", "Adam");
        var lr = cfg.LearningRate;
        var beta1 = cfg.GetOptimizerFloat("beta1", 0.9f);
        var beta2 = cfg.GetOptimizerFloat("beta2", 0.999f);
        var weightDecay = cfg.GetOptimizerFloat("weight_decay", 0f);
        if (weightDecay <= 0f)
        {
            weightDecay = cfg.GetOptimizerFloat("regularizer.factor", 0f);
        }

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
        var lrName = lrConfig.TryGetValue("name", out var rawName) ? rawName?.ToString() ?? "Cosine" : "Cosine";
        if (!lrConfig.ContainsKey("initial_lr"))
        {
            lrConfig["initial_lr"] = cfg.LearningRate;
        }

        if (!lrConfig.ContainsKey("learning_rate"))
        {
            lrConfig["learning_rate"] = cfg.LearningRate;
        }

        if (!lrConfig.ContainsKey("max_epochs"))
        {
            lrConfig["max_epochs"] = cfg.EpochNum;
        }

        if (lrConfig.TryGetValue("warmup_epoch", out var warmupEpoch) && !lrConfig.ContainsKey("warmup_epochs"))
        {
            lrConfig["warmup_epochs"] = warmupEpoch ?? 0;
        }

        return LRSchedulerBuilder.Build(lrName, lrConfig);
    }

    private IRecLoss BuildLoss(TrainingConfigView cfg)
    {
        var lossName = cfg.GetLossString("name", "CTCLoss");
        var lossConfig = cfg.GetLossConfig();
        return RecLossBuilder.Build(lossName, lossConfig);
    }

    private static RecEvalMetrics Evaluate(
        RecModel model,
        ConfigRecDataset evalSet,
        int batchSize,
        int h,
        int maxTextLength,
        Device dev,
        IReadOnlyList<string> vocab)
    {
        model.eval();
        long correct = 0;
        var total = 0L;
        var charTotal = 0L;
        var charErrors = 0L;
        var editSum = 0L;
        using var noGrad = torch.no_grad();
        foreach (var batchData in evalSet.GetBatches(batchSize, shuffle: false, new Random(7)))
        {
            using var x = torch.tensor(batchData.Images, dtype: ScalarType.Float32).reshape(batchData.Batch, 3, h, batchData.Width).to(dev);
            using var y = torch.tensor(batchData.LabelCtc, dtype: ScalarType.Int64).reshape(batchData.Batch, maxTextLength).to(dev);
            var predictions = model.ForwardDict(x);
            var logits = predictions.TryGetValue("ctc", out var ctcValue)
                ? ctcValue
                : (predictions.TryGetValue("predict", out var predValue) ? predValue : predictions.Values.First());

            using var pred = logits.argmax(2).to_type(ScalarType.Int64).cpu();
            using var gt = y.cpu();

            var predFlat = pred.data<long>().ToArray();
            var gtFlat = gt.data<long>().ToArray();
            for (var i = 0; i < batchData.Batch; i++)
            {
                var predSeq = predFlat.Skip(i * maxTextLength).Take(maxTextLength).ToArray();
                var gtSeq = gtFlat.Skip(i * maxTextLength).Take(maxTextLength).ToArray();
                var predText = DecodeCtcPrediction(predSeq, vocab);
                var gtText = DecodeLabel(gtSeq, vocab);
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

            DisposeTensorDictionary(predictions);
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

    private static string DecodeCtcPrediction(long[] ids, IReadOnlyList<string> vocab)
    {
        if (ids.Length == 0)
        {
            return string.Empty;
        }

        var sb = new StringBuilder(ids.Length);
        long prev = 0;
        foreach (var id in ids)
        {
            if (id <= 0)
            {
                prev = 0;
                continue;
            }

            if (id == prev)
            {
                continue;
            }

            if (id < vocab.Count)
            {
                sb.Append(vocab[(int)id]);
            }

            prev = id;
        }

        return sb.ToString();
    }

    private static string DecodeLabel(long[] ids, IReadOnlyList<string> vocab)
    {
        if (ids.Length == 0)
        {
            return string.Empty;
        }

        var sb = new StringBuilder(ids.Length);
        foreach (var id in ids)
        {
            if (id <= 0 || id >= vocab.Count)
            {
                continue;
            }

            sb.Append(vocab[(int)id]);
        }

        return sb.ToString();
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

    private static void SaveSummary(
        TrainingConfigView cfg,
        TrainingSummary summary,
        bool earlyStopped,
        string? resumeCheckpoint,
        int optimizerStepCount,
        int nonZeroGradSteps)
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
        AppendJsonLine(
            Path.Combine(cfg.SaveModelDir, "train_trace.jsonl"),
            new
            {
                event_type = "train_completed",
                optimizer_step_count = optimizerStepCount,
                non_zero_grad_steps = nonZeroGradSteps,
                generated_at_utc = DateTime.UtcNow
            });
    }

    private static void AppendJsonLine(string path, object payload)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        var line = JsonSerializer.Serialize(payload);
        File.AppendAllText(path, line + Environment.NewLine);
    }

    private static float EstimateGradNorm(RecModel model)
    {
        double sum = 0d;
        foreach (var parameter in model.parameters())
        {
            var grad = parameter.grad;
            if (grad is null)
            {
                continue;
            }

            using var gradCpu = grad.cpu().to_type(ScalarType.Float32);
            var values = gradCpu.data<float>().ToArray();
            for (var i = 0; i < values.Length; i++)
            {
                var v = values[i];
                sum += v * v;
            }
        }

        return (float)Math.Sqrt(sum);
    }

    private static void EnsureCharsetCoverage(IReadOnlyList<string> trainLabelFiles, IReadOnlyList<string> evalLabelFiles, IReadOnlyDictionary<char, int> charToId, ILogger logger)
    {
        var missing = new HashSet<char>();
        foreach (var labelFile in trainLabelFiles.Concat(evalLabelFiles))
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

    private static void ApplyLearningRate(Optimizer optimizer, double lr)
    {
        foreach (var pg in optimizer.ParamGroups)
        {
            pg.LearningRate = lr;
        }
    }

    private static int GetGradAccumulationSteps(TrainingConfigView cfg)
    {
        return cfg.GetConfigInt("Optimizer.grad_accumulation_steps", 1);
    }

    private static bool ShouldUseModelAveraging(TrainingConfigView cfg)
    {
        return cfg.GetConfigBool("Global.use_model_averaging", false);
    }

    private static bool ShouldEvalByStep(TrainingConfigView cfg, int globalStep)
    {
        var (start, interval) = cfg.EvalBatchStep;
        if (globalStep < start)
        {
            return false;
        }

        return (globalStep - start) % interval == 0;
    }

    private static IReadOnlyList<string> GetLabelFilesOrFallback(IReadOnlyList<string> labelFiles, string fallback)
    {
        if (labelFiles.Count > 0)
        {
            return labelFiles;
        }

        return string.IsNullOrWhiteSpace(fallback) ? [] : [fallback];
    }

    private static string? ResolveGtcEncodeType(TrainingConfigView cfg)
    {
        var transforms = cfg.GetByPathPublic("Train.dataset.transforms");
        if (transforms is not IList<object?> list)
        {
            return null;
        }

        foreach (var item in list)
        {
            if (item is not Dictionary<string, object?> op || !op.TryGetValue("MultiLabelEncode", out var cfgObj))
            {
                continue;
            }

            if (cfgObj is Dictionary<string, object?> opCfg &&
                opCfg.TryGetValue("gtc_encode", out var gtcEncode) &&
                !string.IsNullOrWhiteSpace(gtcEncode?.ToString()))
            {
                return gtcEncode!.ToString();
            }
        }

        return null;
    }

    private static IRecLabelEncoder? CreateGtcEncoder(string? gtcEncodeType, TrainingConfigView cfg)
    {
        if (string.IsNullOrWhiteSpace(gtcEncodeType))
        {
            return null;
        }

        return gtcEncodeType switch
        {
            "NRTRLabelEncode" => new NRTRLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar),
            "SARLabelEncode" => new SARLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar),
            "AttnLabelEncode" => new AttnLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar),
            _ => null
        };
    }

    private static string? ResolveGtcHeadName(TrainingConfigView cfg)
    {
        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is not IList<object?> list || list.Count < 2)
        {
            return null;
        }

        var second = list[1];
        if (second is not Dictionary<string, object?> headDict || headDict.Count == 0)
        {
            return null;
        }

        return headDict.Keys.FirstOrDefault();
    }

    private static string ResolveNeckEncoderType(TrainingConfigView cfg)
    {
        var direct = cfg.GetArchitectureString("Neck.encoder_type", string.Empty);
        if (!string.IsNullOrWhiteSpace(direct))
        {
            return direct;
        }

        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is not IList<object?> list || list.Count == 0)
        {
            return "reshape";
        }

        if (list[0] is not Dictionary<string, object?> firstHead || !firstHead.TryGetValue("CTCHead", out var ctcCfgRaw))
        {
            return "reshape";
        }

        if (ctcCfgRaw is not Dictionary<string, object?> ctcCfg || !ctcCfg.TryGetValue("Neck", out var neckCfgRaw))
        {
            return "reshape";
        }

        if (neckCfgRaw is not Dictionary<string, object?> neckCfg || !neckCfg.TryGetValue("name", out var neckNameRaw))
        {
            return "reshape";
        }

        var fromHead = neckNameRaw?.ToString();
        return string.IsNullOrWhiteSpace(fromHead) ? "reshape" : fromHead;
    }

    private static int ResolveGtcHiddenSize(TrainingConfigView cfg, int fallback)
    {
        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is not IList<object?> list || list.Count < 2)
        {
            return fallback;
        }

        var second = list[1];
        if (second is not Dictionary<string, object?> headDict || headDict.Count == 0)
        {
            return fallback;
        }

        var headKv = headDict.First();
        if (headKv.Value is not Dictionary<string, object?> headCfg)
        {
            return fallback;
        }

        var key = headKv.Key;
        if (key.Contains("NRTR", StringComparison.OrdinalIgnoreCase) &&
            headCfg.TryGetValue("nrtr_dim", out var nrtrDimRaw) &&
            int.TryParse(nrtrDimRaw?.ToString(), out var nrtrDim) &&
            nrtrDim > 0)
        {
            return nrtrDim;
        }

        return fallback;
    }

    private static string ResolveBackboneAlias(string backboneName)
    {
        return backboneName.ToLowerInvariant() switch
        {
            "pphgnetv2" => "PPHGNetV2_B4",
            _ => backboneName
        };
    }

    private static int ResolveGtcOutChannels(int ctcOutChannels, string? gtcHeadName, string? gtcEncodeType)
    {
        var normalized = (gtcHeadName ?? string.Empty).ToLowerInvariant();
        var encode = gtcEncodeType ?? string.Empty;

        if (normalized.Contains("nrtr", StringComparison.OrdinalIgnoreCase) || encode.Equals("NRTRLabelEncode", StringComparison.OrdinalIgnoreCase))
        {
            return ctcOutChannels + 3;
        }

        if (normalized.Contains("sar", StringComparison.OrdinalIgnoreCase) || encode.Equals("SARLabelEncode", StringComparison.OrdinalIgnoreCase))
        {
            return ctcOutChannels + 2;
        }

        return ctcOutChannels;
    }

    private static void DisposeTensorDictionary(Dictionary<string, Tensor> tensors)
    {
        foreach (var tensor in tensors.Values)
        {
            tensor.Dispose();
        }
    }
}
