using System.Text.Json;
using System.Text;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using PaddleOcr.Data;
using PaddleOcr.Data.LabelEncoders;
using PaddleOcr.Training.Runtime;
using PaddleOcr.Training.Rec.Losses;
using PaddleOcr.Training.Rec.Schedulers;
using PaddleOcr.Training.Rec.Heads;
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
        EnsureCharsetCoverage(cfg, trainLabelFiles, evalLabelFiles, charToId, _logger);

        var gtcEncodeType = ResolveGtcEncodeType(cfg);
        var ctcEncoder = new CTCLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar);
        var gtcEncoder = CreateGtcEncoder(gtcEncodeType, cfg);
        var resizeStrategy = RecTrainingResizeFactory.Create(cfg.GetArchitectureString("algorithm", "SVTR_LCNet"));
        var concatOptions = RecConcatAugmentOptions.FromConfig(cfg, shape.H, shape.W);

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
            multiScaleWidths: cfg.MultiScaleWidths,
            multiScaleHeights: cfg.MultiScaleHeights,
            delimiter: cfg.TrainDelimiter,
            ratioList: cfg.TrainRatioList,
            seed: cfg.Seed,
            concatOptions: concatOptions);

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
            useMultiScale: false,
            delimiter: cfg.EvalDelimiter,
            seed: cfg.Seed + 17,
            concatOptions: RecConcatAugmentOptions.Disabled);

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        _logger.LogInformation("Training(rec) device: {Device}", dev.type);
        _logger.LogInformation("runtime: requested={Requested}, cuda={Cuda}, amp={Amp}, reason={Reason}", runtime.RequestedDevice, runtime.UseCuda, runtime.UseAmp, runtime.Reason);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}, Vocab: {Vocab}", trainSet.Count, evalSet.Count, ctcEncoder.NumClasses);

        var model = BuildModel(cfg, ctcEncoder.NumClasses, gtcEncodeType);
        model.to(dev);

        var optimizer = BuildOptimizer(cfg, model);
        var lrScheduler = BuildLRScheduler(cfg, trainSet.Count, cfg.BatchSize);
        var lossFn = BuildLoss(cfg, gtcEncodeType);
        _logger.LogInformation("CTC input length mode: {Mode}", cfg.CtcInputLengthMode);

        using var teacher = TryCreateTeacherDistiller(cfg, model, shape.H, shape.W, dev);
        var distillEnabled = teacher is not null;
        if (distillEnabled)
        {
            _logger.LogInformation(
                "teacher-student distill enabled: teacher={TeacherDir}, alpha={Alpha:F3}, temp={Temp:F3}, strict={Strict}",
                cfg.TeacherModelDir,
                cfg.DistillWeight,
                cfg.DistillTemperature,
                cfg.StrictTeacherStudent);
        }

        // Log vocab info for diagnostics
        _logger.LogInformation("CTC vocab: size={VocabSize}, first_5=[{Vocab}]",
            ctcEncoder.NumClasses,
            string.Join(",", ctcEncoder.Characters.Take(5)));

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
        var distillMismatchWarned = false;
        var diagnosticsLogged = false;

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
                using var x = torch.tensor(batchData.Images, dtype: ScalarType.Float32).reshape(batchData.Batch, 3, batchData.Height, batchData.Width).to(dev);
                using var yCtc = torch.tensor(batchData.LabelCtc, dtype: ScalarType.Int64).reshape(batchData.Batch, cfg.MaxTextLength).to(dev);
                using var yGtc = torch.tensor(batchData.LabelGtc, dtype: ScalarType.Int64).reshape(batchData.Batch, cfg.MaxTextLength).to(dev);
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
                var ctcTime = ctcLogits.shape.Length >= 2
                    ? (int)Math.Clamp(ctcLogits.shape[1], 1, int.MaxValue)
                    : cfg.MaxTextLength;
                var ctcLens = CtcLengthSanitizer.Sanitize(
                    batchData.Lengths,
                    batchData.ValidRatios,
                    batchData.LabelCtc,
                    ctcTime,
                    cfg.MaxTextLength,
                    cfg.UseValidRatioForCtcInputLength);
                using var targetLengths = torch.tensor(ctcLens.TargetLengths, dtype: ScalarType.Int64, device: dev);
                using var inputLengths = torch.tensor(ctcLens.InputLengths, dtype: ScalarType.Int64, device: dev);

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
                var baseLossVal = loss.ToSingle();
                Tensor? kdLoss = null;
                Tensor? mixedLoss = null;
                float kdLossVal = 0f;
                Tensor effectiveLoss = loss;
                if (distillEnabled && teacher is not null)
                {
                    try
                    {
                        var teacherBatch = teacher.Run(
                            batchData.Images,
                            batchData.Batch,
                            batchData.Height,
                            batchData.Width,
                            batchData.ValidRatios);
                        using var teacherLogits = torch.tensor(teacherBatch.Data, dtype: ScalarType.Float32)
                            .reshape(batchData.Batch, teacherBatch.TimeSteps, teacherBatch.NumClasses)
                            .to(dev);
                        kdLoss = ComputeDistillLossForBatch(
                            ctcLogits,
                            teacherLogits,
                            cfg.DistillTemperature,
                            cfg.StrictTeacherStudent,
                            out var skipReason);
                        if (kdLoss is not null)
                        {
                            mixedLoss = loss * (1f - cfg.DistillWeight) + kdLoss * cfg.DistillWeight;
                            effectiveLoss = mixedLoss;
                            kdLossVal = kdLoss.ToSingle();
                        }
                        else if (!string.IsNullOrWhiteSpace(skipReason) && !distillMismatchWarned)
                        {
                            _logger.LogWarning("{Message}", skipReason);
                            distillMismatchWarned = true;
                        }
                    }
                    catch (Exception ex) when (!cfg.StrictTeacherStudent)
                    {
                        if (!distillMismatchWarned)
                        {
                            _logger.LogWarning(ex, "teacher distillation disabled for subsequent steps: {Message}", ex.Message);
                            distillMismatchWarned = true;
                        }

                        distillEnabled = false;
                    }
                }

                var lossVal = effectiveLoss.ToSingle();

                // Log diagnostics for first batch
                if (!diagnosticsLogged && globalStep == 0)
                {
                    diagnosticsLogged = true;
                    LogTrainingDiagnostics(ctcLogits, yCtc, batchData, cfg, ctcEncoder);
                }

                if (float.IsNaN(lossVal) || float.IsInfinity(lossVal))
                {
                    var nanReason =
                        $"ctc_time={ctcTime}, truncated_by_time={ctcLens.TruncatedByTime}, truncated_by_input={ctcLens.TruncatedByInput}, " +
                        $"truncated_by_repeat={ctcLens.TruncatedByRepeatConstraint}, empty_targets={ctcLens.EmptyTargets}";
                    _logger.LogWarning("NaN/Inf loss detected at step {Step}, skipping this batch. {Reason}", globalStep, nanReason);
                    AppendJsonLine(
                        tracePath,
                        new
                        {
                            event_type = "nan_skip",
                            epoch,
                            global_step = globalStep,
                            reason = nanReason,
                            batch = batchData.Batch,
                            height = batchData.Height,
                            width = batchData.Width
                        });
                    globalStep++;
                    mixedLoss?.Dispose();
                    kdLoss?.Dispose();
                    DisposeTensorDictionary(predictions);
                    DisposeTensorDictionary(lossDict);
                    continue;
                }

                var scaledLoss = ampHelper is not null ? ampHelper.ScaleLoss(effectiveLoss) : effectiveLoss;
                scaledLoss.backward();
                if (!ReferenceEquals(scaledLoss, effectiveLoss))
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
                    var meanTargetLen = batchData.Lengths.Length == 0 ? 0f : (float)batchData.Lengths.Average();
                    float? blankRatio = null;
                    if (ctcLogits.shape.Length >= 3)
                    {
                        using var stepPred = ctcLogits.argmax(2);
                        using var blankMask = stepPred.eq(0);
                        var blankCount = blankMask.to_type(ScalarType.Float32).mean().ToSingle();
                        blankRatio = blankCount;
                    }

                    AppendJsonLine(
                        tracePath,
                        new
                        {
                            epoch,
                            global_step = globalStep,
                            loss = lossVal,
                            base_loss = baseLossVal,
                            kd_loss = kdLossVal,
                            smooth_loss = smoothedLoss,
                            lr = lrScheduler.CurrentLR,
                            batch = batchData.Batch,
                            height = batchData.Height,
                            width = batchData.Width,
                            ctc_time_steps = ctcTime,
                            target_len_mean = meanTargetLen,
                            effective_batch = batchData.Batch - ctcLens.EmptyTargets,
                            ctc_truncated_by_time = ctcLens.TruncatedByTime,
                            ctc_truncated_by_input = ctcLens.TruncatedByInput,
                            ctc_truncated_by_repeat = ctcLens.TruncatedByRepeatConstraint,
                            ctc_empty_targets = ctcLens.EmptyTargets,
                            step_ms = stepMs,
                            ctc_blank_ratio = blankRatio,
                            optimizer_step_count = optimizerStepCount,
                            non_zero_grad_steps = nonZeroGradSteps
                        });
                    _logger.LogInformation(
                        "train step epoch={Epoch} step={Step} loss={Loss:F4} base_loss={BaseLoss:F4} kd_loss={KdLoss} smooth_loss={Smooth:F4} lr={Lr:F6} ctc_time={CtcTime} tgt_len_mean={TargetLen:F2} step_ms={StepMs:F1} ctc_blank_ratio={BlankRatio}",
                        epoch,
                        globalStep,
                        lossVal,
                        baseLossVal,
                        kdLoss is null ? "n/a" : $"{kdLossVal:F4}",
                        smoothedLoss,
                        lrScheduler.CurrentLR,
                        ctcTime,
                        meanTargetLen,
                        stepMs,
                        blankRatio is null ? "n/a" : $"{blankRatio.Value:F4}");
                }

                if (shouldUpdate || globalStep == 1)
                {
                    lrScheduler.Step(globalStep, epoch);
                    ApplyLearningRate(optimizer, lrScheduler.CurrentLR);
                }

                if (cfg.CalMetricDuringTrain && ShouldEvalByStep(cfg, globalStep))
                {
                    var stepEval = Evaluate(model, evalSet, cfg.EvalBatchSize, cfg.MaxTextLength, dev, ctcEncoder.Characters);
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

                mixedLoss?.Dispose();
                kdLoss?.Dispose();
                DisposeTensorDictionary(predictions);
                DisposeTensorDictionary(lossDict);
            }

            if (ShouldUseModelAveraging(cfg))
            {
                modelAverager.Update(model);
            }

            var trainLoss = sampleCount == 0 ? 0f : lossSum / sampleCount;
            var evalMetrics = Evaluate(model, evalSet, cfg.EvalBatchSize, cfg.MaxTextLength, dev, ctcEncoder.Characters);

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
            enableAugmentation: false,
            delimiter: cfg.EvalDelimiter,
            seed: cfg.Seed + 17);

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        var model = BuildModel(cfg, ctcEncoder.NumClasses, gtcEncodeType);
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt))
        {
            TryLoadCheckpoint(model, ckpt);
        }

        var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, cfg.MaxTextLength, dev, ctcEncoder.Characters);
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
        var algorithm = cfg.GetArchitectureString("algorithm", "SVTR_LCNet");
        var transformNameRaw = cfg.GetArchitectureString("Transform.name", string.Empty);
        var transformName = string.IsNullOrWhiteSpace(transformNameRaw) ? null : transformNameRaw;
        var backboneName = cfg.GetArchitectureString("Backbone.name", InferBackboneName(algorithm));
        var resolvedBackboneName = ResolveBackboneAlias(backboneName);
        var neckName = cfg.GetArchitectureString("Neck.name", InferNeckName(algorithm));
        var neckEncoderType = ResolveNeckEncoderType(cfg);
        if (cfg.GetByPathPublic("Architecture.Neck.encoder_type") is null &&
            string.Equals(neckEncoderType, "reshape", StringComparison.OrdinalIgnoreCase))
        {
            neckEncoderType = InferNeckEncoderType(algorithm);
        }

        var headName = cfg.GetArchitectureString("Head.name", InferHeadName(algorithm));
        var gtcHeadName = ResolveGtcHeadName(cfg, headName, algorithm, gtcEncodeType);
        var neckHiddenSize = ResolveNeckHiddenSize(cfg, algorithm, neckEncoderType);
        var headHiddenSize = ResolveHeadHiddenSize(cfg, algorithm, headName, gtcHeadName);
        var maxLen = cfg.MaxTextLength;
        var inChannels = cfg.GetArchitectureInt("in_channels", 3);
        var gtcOutChannels = ResolveGtcOutChannels(numClasses, gtcHeadName, gtcEncodeType);

        // For MultiHead: extract CTC neck config and NRTR dim
        MultiHeadCtcNeckConfig? ctcNeckConfig = null;
        var nrtrDim = 0;
        if (string.Equals(headName, "MultiHead", StringComparison.OrdinalIgnoreCase))
        {
            var extracted = ExtractMultiHeadConfig(cfg);
            ctcNeckConfig = extracted.CtcNeckConfig;
            nrtrDim = extracted.NrtrDim;

            // When MultiHead has internal CTC encoder, model-level neck should be "reshape"
            if (ctcNeckConfig is not null)
            {
                neckEncoderType = "reshape";
            }
        }

        _logger.LogInformation(
                "Building model: transform={Transform}, backbone={Backbone}, neck={Neck}({NeckEncoder}, hidden={NeckHidden}), head={Head}(hidden={HeadHidden}), num_classes={Classes}, gtc_head={GtcHead}, gtc_classes={GtcClasses}" +
                (ctcNeckConfig is not null ? ", ctc_encoder={CtcEnc}(dims={Dims},depth={Depth})" : "") +
                (nrtrDim > 0 ? ", nrtr_dim={NrtrDim}" : ""),
                transformName ?? "none",
                resolvedBackboneName,
                neckName,
                neckEncoderType,
                neckHiddenSize,
                headName,
                headHiddenSize,
                numClasses,
                gtcHeadName ?? "none",
                gtcOutChannels,
                ctcNeckConfig?.EncoderType ?? "none",
                ctcNeckConfig?.Dims ?? 0,
                ctcNeckConfig?.Depth ?? 0,
                nrtrDim);

        if (cfg.GetByPathPublic("Architecture.Backbone.name") is null)
        {
            _logger.LogInformation("Backbone.name missing in config; inferred '{Backbone}' from algorithm '{Algorithm}'", backboneName, algorithm);
        }

        if (cfg.GetByPathPublic("Architecture.Head.name") is null)
        {
            _logger.LogInformation("Head.name missing in config; inferred '{Head}' from algorithm '{Algorithm}'", headName, algorithm);
        }

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
            neckHiddenSize,
            maxLen,
            neckEncoderType,
            gtcHeadName,
            gtcOutChannels,
            transformName,
            headHiddenSize,
            ctcNeckConfig,
            nrtrDim);
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

    private ILRScheduler BuildLRScheduler(TrainingConfigView cfg, int trainSampleCount, int batchSize)
    {
        var lrConfig = cfg.GetOptimizerLrConfig();
        var lrName = lrConfig.TryGetValue("name", out var rawName) ? rawName?.ToString() ?? "Cosine" : "Cosine";
        var stepsPerEpoch = Math.Max(1, (int)Math.Ceiling(trainSampleCount / (double)Math.Max(1, batchSize)));
        var maxSteps = Math.Max(1, stepsPerEpoch * Math.Max(1, cfg.EpochNum));
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

        if (!lrConfig.ContainsKey("max_steps"))
        {
            lrConfig["max_steps"] = maxSteps;
        }

        if (lrConfig.TryGetValue("warmup_epoch", out var warmupEpoch) && !lrConfig.ContainsKey("warmup_epochs"))
        {
            lrConfig["warmup_epochs"] = warmupEpoch ?? 0;
        }

        if (!lrConfig.ContainsKey("warmup_steps"))
        {
            var warmupEpochs = 0;
            if (lrConfig.TryGetValue("warmup_epochs", out var warmupEpochsObj))
            {
                warmupEpochs = ParseInt(warmupEpochsObj, warmupEpochs);
            }
            else if (lrConfig.TryGetValue("warmup_epoch", out var warmupEpochObj))
            {
                warmupEpochs = ParseInt(warmupEpochObj, warmupEpochs);
            }

            lrConfig["warmup_steps"] = Math.Max(0, warmupEpochs) * stepsPerEpoch;
        }

        _logger.LogInformation(
            "lr scheduler: name={Name}, initial_lr={InitialLr:F6}, steps_per_epoch={StepsPerEpoch}, max_steps={MaxSteps}, warmup_steps={WarmupSteps}",
            lrName,
            cfg.LearningRate,
            stepsPerEpoch,
            Convert.ToInt32(lrConfig["max_steps"]),
            Convert.ToInt32(lrConfig["warmup_steps"]));

        return LRSchedulerBuilder.Build(lrName, lrConfig);
    }

    private static int ParseInt(object? raw, int fallback)
    {
        if (raw is null)
        {
            return fallback;
        }

        return raw switch
        {
            int i => i,
            long l => (int)l,
            float f => (int)f,
            double d => (int)d,
            decimal m => (int)m,
            _ => int.TryParse(raw.ToString(), out var parsed) ? parsed : fallback
        };
    }

    private IRecLoss BuildLoss(TrainingConfigView cfg, string? gtcEncodeType)
    {
        var algorithm = cfg.GetArchitectureString("algorithm", "SVTR_LCNet");
        var headName = cfg.GetArchitectureString("Head.name", InferHeadName(algorithm));
        var gtcHeadName = ResolveGtcHeadName(cfg, headName, algorithm, gtcEncodeType);
        var configuredLossName = cfg.GetLossString("name", string.Empty);
        var lossName = string.IsNullOrWhiteSpace(configuredLossName)
            ? InferLossName(headName)
            : configuredLossName;
        var lossConfig = cfg.GetLossConfig();
        if (string.IsNullOrWhiteSpace(configuredLossName))
        {
            _logger.LogInformation("Loss.name missing in config; inferred '{Loss}' from head '{Head}'", lossName, headName);
        }

        if (lossName.Equals("MultiLoss", StringComparison.OrdinalIgnoreCase))
        {
            EnsureMultiLossConfig(lossConfig, gtcHeadName, gtcEncodeType);
        }

        return RecLossBuilder.Build(lossName, lossConfig);
    }

    private static RecEvalMetrics Evaluate(
        RecModel model,
        ConfigRecDataset evalSet,
        int batchSize,
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
            using var x = torch.tensor(batchData.Images, dtype: ScalarType.Float32).reshape(batchData.Batch, 3, batchData.Height, batchData.Width).to(dev);
            using var y = torch.tensor(batchData.LabelCtc, dtype: ScalarType.Int64).reshape(batchData.Batch, maxTextLength).to(dev);
            var predictions = model.ForwardDict(x);
            var logits = predictions.TryGetValue("ctc", out var ctcValue)
                ? ctcValue
                : (predictions.TryGetValue("predict", out var predValue) ? predValue : predictions.Values.First());

            using var pred = logits.argmax(2).to_type(ScalarType.Int64).cpu();
            using var gt = y.cpu();

            var predFlat = pred.data<long>().ToArray();
            var gtFlat = gt.data<long>().ToArray();
            var predTimeSteps = (int)pred.shape[1];
            for (var i = 0; i < batchData.Batch; i++)
            {
                var predSeq = predFlat.Skip(i * predTimeSteps).Take(predTimeSteps).ToArray();
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
        var charAcc = charTotal == 0 ? 0f : Math.Clamp(1f - (float)charErrors / charTotal, 0f, 1f);
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
            // Skip blank (id == 0) or invalid IDs
            if (id <= 0 || id >= vocab.Count)
            {
                prev = 0;
                continue;
            }

            // Skip consecutive duplicates (CTC deduplication rule)
            if (id == prev)
            {
                continue;
            }

            sb.Append(vocab[(int)id]);
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
            // Skip blank (id == 0) or invalid IDs (consistent with DecodeCtcPrediction)
            if (id <= 0 || id >= vocab.Count)
            {
                continue;
            }

            sb.Append(vocab[(int)id]);
        }

        return sb.ToString();
    }

    private PaddleOcr.Training.RecTeacherDistiller? TryCreateTeacherDistiller(
        TrainingConfigView cfg,
        RecModel model,
        int imageH,
        int imageW,
        Device dev)
    {
        if (string.IsNullOrWhiteSpace(cfg.TeacherModelDir))
        {
            return null;
        }

        if (cfg.DistillWeight <= 0f)
        {
            throw new InvalidOperationException("Global.teacher_model_dir is set but Global.distill_weight <= 0. Set distill_weight in (0,1].");
        }

        var widths = cfg.MultiScaleWidths.Distinct().ToArray();
        var heights = cfg.MultiScaleHeights.Distinct().ToArray();
        if (cfg.UseMultiScale && (widths.Length > 1 || heights.Length > 1))
        {
            var msg = "Teacher distillation with multi-scale training (multiple widths/heights) is not supported in strict shape mode. Use a single train scale or disable distillation.";
            if (cfg.StrictTeacherStudent)
            {
                throw new InvalidOperationException(msg);
            }

            _logger.LogWarning("{Message}", msg);
            return null;
        }

        using var noGrad = torch.no_grad();
        using var dryInput = torch.zeros([1, 3, imageH, imageW], dtype: ScalarType.Float32, device: dev);
        var dryPredictions = model.ForwardDict(dryInput);
        try
        {
            var dryCtc = dryPredictions.TryGetValue("ctc", out var ctcValue)
                ? ctcValue
                : (dryPredictions.TryGetValue("predict", out var predValue) ? predValue : dryPredictions.Values.First());
            if (dryCtc.shape.Length != 3)
            {
                throw new InvalidOperationException($"Student CTC logits must be rank-3 [B,T,C], but got [{string.Join(",", dryCtc.shape)}]");
            }

            var studentTimeSteps = (int)dryCtc.shape[1];
            var studentNumClasses = (int)dryCtc.shape[2];
            return PaddleOcr.Training.RecTeacherDistiller.TryCreate(
                cfg,
                _logger,
                imageH,
                imageW,
                studentTimeSteps,
                studentNumClasses);
        }
        finally
        {
            DisposeTensorDictionary(dryPredictions);
        }
    }

    private static Tensor? ComputeDistillLossForBatch(
        Tensor studentLogits,
        Tensor teacherLogits,
        float temperature,
        bool strictTeacherStudent,
        out string? skipReason)
    {
        skipReason = null;
        if (studentLogits.shape.Length != 3 || teacherLogits.shape.Length != 3)
        {
            var msg = $"Teacher/student logits must be rank-3 [B,T,C], got student=[{string.Join(",", studentLogits.shape)}], teacher=[{string.Join(",", teacherLogits.shape)}].";
            if (strictTeacherStudent)
            {
                throw new InvalidOperationException(msg);
            }

            skipReason = msg + " distillation skipped because strict_teacher_student=false.";
            return null;
        }

        var sb = studentLogits.shape[0];
        var st = studentLogits.shape[1];
        var sc = studentLogits.shape[2];
        var tb = teacherLogits.shape[0];
        var tt = teacherLogits.shape[1];
        var tc = teacherLogits.shape[2];
        if (sb != tb || st != tt || sc != tc)
        {
            var msg =
                $"Teacher/student logits shape mismatch: student=[B:{sb}, T:{st}, C:{sc}], teacher=[B:{tb}, T:{tt}, C:{tc}]. " +
                "Use same rec_image_shape, dict and max_text_length as teacher model.";
            if (strictTeacherStudent)
            {
                throw new InvalidOperationException(msg);
            }

            skipReason = msg + " distillation skipped because strict_teacher_student=false.";
            return null;
        }

        return ComputeDistillLoss(studentLogits, teacherLogits, temperature);
    }

    private static Tensor ComputeDistillLoss(Tensor studentLogits, Tensor teacherLogits, float temperature)
    {
        using var studentScaled = studentLogits / temperature;
        using var teacherScaled = teacherLogits / temperature;
        using var studentLogProb = torch.nn.functional.log_softmax(studentScaled, dim: -1);
        using var teacherProb = torch.nn.functional.softmax(teacherScaled, dim: -1);
        using var teacherLogProb = torch.nn.functional.log_softmax(teacherScaled, dim: -1);
        using var tokenKl = (teacherProb * (teacherLogProb - studentLogProb)).sum(dim: -1);
        return tokenKl.mean() * (temperature * temperature);
    }

    private void TryLoadCheckpoint(RecModel model, string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
        {
            _logger.LogWarning("Checkpoint not found: {Path}", checkpointPath);
            return;
        }

        var rollbackPath = Path.Combine(Path.GetTempPath(), $"pocr_config_rec_rollback_{Guid.NewGuid():N}.pt");
        var canRollback = false;
        try
        {
            model.save(rollbackPath);
            canRollback = true;
            _logger.LogInformation("Loading checkpoint: {Path}", checkpointPath);
            model.load(checkpointPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load checkpoint {Path}", checkpointPath);
            if (canRollback && File.Exists(rollbackPath))
            {
                try
                {
                    model.load(rollbackPath);
                    _logger.LogInformation("Model state restored from rollback snapshot after checkpoint load failure.");
                }
                catch (Exception restoreEx)
                {
                    _logger.LogWarning(restoreEx, "Failed to restore rollback snapshot {Path}", rollbackPath);
                }
            }
        }
        finally
        {
            if (canRollback && File.Exists(rollbackPath))
            {
                try
                {
                    File.Delete(rollbackPath);
                }
                catch
                {
                    // best effort cleanup
                }
            }
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

    private static void EnsureCharsetCoverage(
        TrainingConfigView cfg,
        IReadOnlyList<string> trainLabelFiles,
        IReadOnlyList<string> evalLabelFiles,
        IReadOnlyDictionary<char, int> charToId,
        ILogger logger)
    {
        var missing = new HashSet<char>();
        var totalChars = 0L;
        var unknownChars = 0L;
        var totalSamples = 0L;
        var parsedSamples = 0L;
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

                totalSamples++;
                if (!RecLabelLineParser.TryParse(line, out _, out var text))
                {
                    continue;
                }

                parsedSamples++;
                foreach (var ch in text)
                {
                    totalChars++;
                    if (!charToId.ContainsKey(ch))
                    {
                        unknownChars++;
                        missing.Add(ch);
                    }
                }
            }
        }

        var unknownRatio = totalChars == 0 ? 0f : (float)unknownChars / totalChars;
        logger.LogInformation(
            "rec charset audit: parsed_samples={Parsed}/{Total}, total_chars={Chars}, unknown_chars={Unknown}, unknown_ratio={Ratio:F4}",
            parsedSamples,
            totalSamples,
            totalChars,
            unknownChars,
            unknownRatio);

        if (missing.Count > 0)
        {
            var preview = new string(missing.Take(32).ToArray());
            logger.LogWarning("rec charset missing {Count} chars from labels. example: {Chars}", missing.Count, preview);
        }

        if (cfg.CharsetCoverageFailFast && unknownRatio > cfg.CharsetMaxUnknownRatio)
        {
            throw new InvalidOperationException(
                $"rec charset coverage check failed: unknown_ratio={unknownRatio:F4} exceeds threshold={cfg.CharsetMaxUnknownRatio:F4}. " +
                "Use matching character_dict_path or disable Global.charset_coverage_fail_fast.");
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
            return InferGtcEncodeTypeFromConfig(cfg);
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

        return InferGtcEncodeTypeFromConfig(cfg);
    }

    private static IRecLabelEncoder? CreateGtcEncoder(string? gtcEncodeType, TrainingConfigView cfg)
    {
        if (string.IsNullOrWhiteSpace(gtcEncodeType))
        {
            return null;
        }

        return gtcEncodeType.Trim().ToLowerInvariant() switch
        {
            "nrtrlabelencode" => new NRTRLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar),
            "sarlabelencode" => new SARLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar),
            "attnlabelencode" => new AttnLabelEncode(cfg.MaxTextLength, cfg.RecCharDictPath, cfg.UseSpaceChar),
            _ => null
        };
    }

    private static string? ResolveGtcHeadName(TrainingConfigView cfg, string headName, string algorithm, string? gtcEncodeType)
    {
        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is IList<object?> list && list.Count >= 2)
        {
            var second = list[1];
            if (second is Dictionary<string, object?> headDict && headDict.Count > 0)
            {
                return headDict.Keys.FirstOrDefault();
            }
        }

        if (!IsMultiHead(headName))
        {
            return null;
        }

        var fromEncode = InferGtcHeadNameFromEncode(gtcEncodeType);
        if (!string.IsNullOrWhiteSpace(fromEncode))
        {
            return fromEncode;
        }

        return InferDefaultGtcHeadName(algorithm);
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

    private static (MultiHeadCtcNeckConfig? CtcNeckConfig, int NrtrDim) ExtractMultiHeadConfig(TrainingConfigView cfg)
    {
        MultiHeadCtcNeckConfig? ctcNeckConfig = null;
        var nrtrDim = 0;

        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is not IList<object?> list)
        {
            return (null, 0);
        }

        // Extract CTC neck config from head_list[0].CTCHead.Neck
        if (list.Count > 0 && list[0] is Dictionary<string, object?> firstHead &&
            firstHead.TryGetValue("CTCHead", out var ctcCfgRaw) &&
            ctcCfgRaw is Dictionary<string, object?> ctcCfg &&
            ctcCfg.TryGetValue("Neck", out var neckCfgRaw) &&
            neckCfgRaw is Dictionary<string, object?> neckCfg)
        {
            var encoderType = neckCfg.TryGetValue("name", out var nameObj) ? nameObj?.ToString() ?? "reshape" : "reshape";
            var dims = neckCfg.TryGetValue("dims", out var dimsObj) ? ToInt(dimsObj) : 0;
            var depth = neckCfg.TryGetValue("depth", out var depthObj) ? ToInt(depthObj) : 1;
            var hiddenDims = neckCfg.TryGetValue("hidden_dims", out var hiddenObj) ? ToInt(hiddenObj) : 0;

            if (dims > 0 && !string.Equals(encoderType, "reshape", StringComparison.OrdinalIgnoreCase))
            {
                ctcNeckConfig = new MultiHeadCtcNeckConfig(encoderType, dims, depth, hiddenDims);
            }
        }

        // Extract nrtr_dim from head_list[1].NRTRHead.nrtr_dim
        if (list.Count > 1 && list[1] is Dictionary<string, object?> secondHead &&
            secondHead.TryGetValue("NRTRHead", out var nrtrCfgRaw) &&
            nrtrCfgRaw is Dictionary<string, object?> nrtrCfg &&
            nrtrCfg.TryGetValue("nrtr_dim", out var dimObj))
        {
            nrtrDim = ToInt(dimObj);
        }

        return (ctcNeckConfig, nrtrDim);
    }

    private static int ToInt(object? obj)
    {
        if (obj is null) return 0;
        return obj switch
        {
            int i => i,
            long l => (int)l,
            float f => (int)f,
            double d => (int)d,
            decimal m => (int)m,
            _ => int.TryParse(obj.ToString(), out var parsed) ? parsed : 0
        };
    }

    private static string InferBackboneName(string algorithm)
    {
        return algorithm.Trim().ToLowerInvariant() switch
        {
            "svtr_lcnet" or "svtrlcnet" => "PPLCNetV3",
            "svtr_hgnet" or "svtrhgnet" => "PPHGNetV2_B4",
            "crnn" => "MobileNetV3",
            "svtr" or "svtrnet" => "SVTRNet",
            "nrtr" => "MTB",
            "sar" => "ResNet31",
            "robustscanner" => "ResNet31",
            _ => "MobileNetV1Enhance"
        };
    }

    private static string InferNeckName(string algorithm)
    {
        return algorithm.Trim().ToLowerInvariant() switch
        {
            _ => "SequenceEncoder"
        };
    }

    private static string InferNeckEncoderType(string algorithm)
    {
        return algorithm.Trim().ToLowerInvariant() switch
        {
            "svtr_lcnet" or "svtrlcnet" or "svtr_hgnet" or "svtrhgnet" or "svtr" or "svtrnet" => "svtr",
            "crnn" => "rnn",
            _ => "reshape"
        };
    }

    private static string InferHeadName(string algorithm)
    {
        return algorithm.Trim().ToLowerInvariant() switch
        {
            "svtr_lcnet" or "svtrlcnet" or "svtr_hgnet" or "svtrhgnet" => "MultiHead",
            "sar" => "SARHead",
            "nrtr" => "NRTRHead",
            "srn" => "SRNHead",
            "robustscanner" => "RobustScannerHead",
            _ => "CTCHead"
        };
    }

    private static int InferHeadHiddenSize(string algorithm, string headName)
    {
        var alg = algorithm.Trim().ToLowerInvariant();
        var head = headName.Trim().ToLowerInvariant();
        if (head.Contains("sar", StringComparison.OrdinalIgnoreCase))
        {
            return 512;
        }

        if (head.Contains("nrtr", StringComparison.OrdinalIgnoreCase))
        {
            return 384;
        }

        return alg switch
        {
            "svtr_lcnet" or "svtrlcnet" or "svtr_hgnet" or "svtrhgnet" => 120,
            "crnn" => 96,
            _ => 48
        };
    }

    private static int ResolveNeckHiddenSize(TrainingConfigView cfg, string algorithm, string neckEncoderType)
    {
        var direct = cfg.GetArchitectureInt("Neck.hidden_size", 0);
        if (direct > 0)
        {
            return direct;
        }

        var fromHeadList = TryResolveCtcNeckHiddenSize(cfg);
        if (fromHeadList > 0)
        {
            return fromHeadList;
        }

        return InferNeckHiddenSize(algorithm, neckEncoderType);
    }

    private static int ResolveHeadHiddenSize(TrainingConfigView cfg, string algorithm, string headName, string? gtcHeadName)
    {
        var direct = cfg.GetArchitectureInt("Head.hidden_size", 0);
        if (direct > 0)
        {
            return direct;
        }

        var fromHeadList = TryResolveGtcHeadHiddenSize(cfg);
        if (fromHeadList > 0)
        {
            return fromHeadList;
        }

        if (IsMultiHead(headName))
        {
            var gtcFallback = InferHeadHiddenSize(algorithm, gtcHeadName ?? string.Empty);
            if (gtcFallback > 0)
            {
                return gtcFallback;
            }
        }

        return InferHeadHiddenSize(algorithm, headName);
    }

    private static int TryResolveCtcNeckHiddenSize(TrainingConfigView cfg)
    {
        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is not IList<object?> list || list.Count == 0)
        {
            return 0;
        }

        if (list[0] is not Dictionary<string, object?> firstHead ||
            !firstHead.TryGetValue("CTCHead", out var ctcCfgRaw) ||
            ctcCfgRaw is not Dictionary<string, object?> ctcCfg ||
            !ctcCfg.TryGetValue("Neck", out var neckCfgRaw) ||
            neckCfgRaw is not Dictionary<string, object?> neckCfg)
        {
            return 0;
        }

        if (TryParsePositiveInt(neckCfg.TryGetValue("dims", out var dimsRaw) ? dimsRaw : null, out var dims))
        {
            return dims;
        }

        if (TryParsePositiveInt(neckCfg.TryGetValue("hidden_dims", out var hiddenDimsRaw) ? hiddenDimsRaw : null, out var hiddenDims))
        {
            return hiddenDims;
        }

        if (TryParsePositiveInt(neckCfg.TryGetValue("hidden_size", out var hiddenSizeRaw) ? hiddenSizeRaw : null, out var hiddenSize))
        {
            return hiddenSize;
        }

        return 0;
    }

    private static int TryResolveGtcHeadHiddenSize(TrainingConfigView cfg)
    {
        var raw = cfg.GetByPathPublic("Architecture.Head.head_list");
        if (raw is not IList<object?> list || list.Count < 2)
        {
            return 0;
        }

        var second = list[1];
        if (second is not Dictionary<string, object?> headDict || headDict.Count == 0)
        {
            return 0;
        }

        var headKv = headDict.First();
        if (headKv.Value is not Dictionary<string, object?> headCfg)
        {
            return 0;
        }

        var key = headKv.Key;
        if (key.Contains("NRTR", StringComparison.OrdinalIgnoreCase) &&
            TryParsePositiveInt(headCfg.TryGetValue("nrtr_dim", out var nrtrDimRaw) ? nrtrDimRaw : null, out var nrtrDim))
        {
            return nrtrDim;
        }

        if (key.Contains("SAR", StringComparison.OrdinalIgnoreCase) &&
            TryParsePositiveInt(headCfg.TryGetValue("enc_dim", out var encDimRaw) ? encDimRaw : null, out var encDim))
        {
            return encDim;
        }

        if ((key.Contains("Attn", StringComparison.OrdinalIgnoreCase) || key.Contains("Attention", StringComparison.OrdinalIgnoreCase)) &&
            TryParsePositiveInt(headCfg.TryGetValue("hidden_size", out var hiddenRaw) ? hiddenRaw : null, out var hiddenSize))
        {
            return hiddenSize;
        }

        return 0;
    }

    private static int InferNeckHiddenSize(string algorithm, string neckEncoderType)
    {
        var alg = algorithm.Trim().ToLowerInvariant();
        var neck = neckEncoderType.Trim().ToLowerInvariant();
        if (alg is "svtr_lcnet" or "svtrlcnet" or "svtr_hgnet" or "svtrhgnet" || neck == "svtr")
        {
            return 120;
        }

        if (neck is "rnn" or "cascadernn")
        {
            return 96;
        }

        return 48;
    }

    private static string? InferGtcEncodeTypeFromConfig(TrainingConfigView cfg)
    {
        var algorithm = cfg.GetArchitectureString("algorithm", "SVTR_LCNet");
        var headName = cfg.GetArchitectureString("Head.name", InferHeadName(algorithm));
        var gtcHeadName = ResolveGtcHeadName(cfg, headName, algorithm, gtcEncodeType: null);
        return InferGtcEncodeTypeFromHeadName(gtcHeadName);
    }

    private static bool IsMultiHead(string headName)
    {
        return headName.Contains("multi", StringComparison.OrdinalIgnoreCase);
    }

    private static string? InferDefaultGtcHeadName(string algorithm)
    {
        return algorithm.Trim().ToLowerInvariant() switch
        {
            "svtr_lcnet" or "svtrlcnet" or "svtr_hgnet" or "svtrhgnet" => "NRTRHead",
            _ => null
        };
    }

    private static string? InferGtcHeadNameFromEncode(string? gtcEncodeType)
    {
        if (string.IsNullOrWhiteSpace(gtcEncodeType))
        {
            return null;
        }

        return gtcEncodeType.Trim().ToLowerInvariant() switch
        {
            "nrtrlabelencode" => "NRTRHead",
            "sarlabelencode" => "SARHead",
            "attnlabelencode" => "AttentionHead",
            _ => null
        };
    }

    private static string? InferGtcEncodeTypeFromHeadName(string? gtcHeadName)
    {
        if (string.IsNullOrWhiteSpace(gtcHeadName))
        {
            return null;
        }

        var normalized = gtcHeadName.Trim().ToLowerInvariant();
        if (normalized.Contains("nrtr", StringComparison.OrdinalIgnoreCase))
        {
            return "NRTRLabelEncode";
        }

        if (normalized.Contains("sar", StringComparison.OrdinalIgnoreCase))
        {
            return "SARLabelEncode";
        }

        if (normalized.Contains("attn", StringComparison.OrdinalIgnoreCase) || normalized.Contains("attention", StringComparison.OrdinalIgnoreCase))
        {
            return "AttnLabelEncode";
        }

        return null;
    }

    private static string InferLossName(string headName)
    {
        var normalized = headName.Trim().ToLowerInvariant();
        if (normalized.Contains("multi", StringComparison.OrdinalIgnoreCase))
        {
            return "MultiLoss";
        }

        if (normalized.Contains("sar", StringComparison.OrdinalIgnoreCase))
        {
            return "SARLoss";
        }

        if (normalized.Contains("nrtr", StringComparison.OrdinalIgnoreCase))
        {
            return "NRTRLoss";
        }

        if (normalized.Contains("attn", StringComparison.OrdinalIgnoreCase) || normalized.Contains("attention", StringComparison.OrdinalIgnoreCase))
        {
            return "AttentionLoss";
        }

        return "CTCLoss";
    }

    private static void EnsureMultiLossConfig(Dictionary<string, object> lossConfig, string? gtcHeadName, string? gtcEncodeType)
    {
        if (HasLossConfigList(lossConfig))
        {
            return;
        }

        var gtcLossName = InferGtcLossName(gtcHeadName, gtcEncodeType);
        lossConfig["loss_config_list"] = new List<object?>
        {
            new Dictionary<string, object?> { ["CTCLoss"] = new Dictionary<string, object?>() },
            new Dictionary<string, object?> { [gtcLossName] = new Dictionary<string, object?>() }
        };
    }

    private static string InferGtcLossName(string? gtcHeadName, string? gtcEncodeType)
    {
        var normalized = (gtcHeadName ?? string.Empty).ToLowerInvariant();
        var encode = (gtcEncodeType ?? string.Empty).ToLowerInvariant();
        if (normalized.Contains("nrtr", StringComparison.OrdinalIgnoreCase) || encode == "nrtrlabelencode")
        {
            return "NRTRLoss";
        }

        if (normalized.Contains("sar", StringComparison.OrdinalIgnoreCase) || encode == "sarlabelencode")
        {
            return "SARLoss";
        }

        return "AttentionLoss";
    }

    private static bool HasLossConfigList(Dictionary<string, object> lossConfig)
    {
        if (!lossConfig.TryGetValue("loss_config_list", out var raw) || raw is null)
        {
            return false;
        }

        return raw switch
        {
            IList<object?> list => list.Count > 0,
            System.Collections.IList list => list.Count > 0,
            _ => false
        };
    }

    private static bool TryParsePositiveInt(object? raw, out int value)
    {
        value = 0;
        if (raw is null)
        {
            return false;
        }

        return int.TryParse(raw.ToString(), out value) && value > 0;
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

    private void LogTrainingDiagnostics(
        Tensor ctcLogits,
        Tensor yCtc,
        ConfigRecBatch batchData,
        TrainingConfigView cfg,
        CTCLabelEncode ctcEncoder)
    {
        try
        {
            // Log model output shape
            var logitsShape = ctcLogits.shape;
            _logger.LogInformation(
                "Diagnostics: model_output_shape=[B={Batch},T={Time},C={Classes}]",
                logitsShape[0], logitsShape.Length > 1 ? logitsShape[1] : 0, logitsShape.Length > 2 ? logitsShape[2] : 0);

            // Log blank ratio
            using var pred = ctcLogits.argmax(2);
            using var blankMask = pred.eq(0);
            var blankRatio = blankMask.to_type(ScalarType.Float32).mean().ToSingle();
            _logger.LogInformation("Diagnostics: blank_ratio={BlankRatio:F4}", blankRatio);

            // Decode and log first few predictions vs labels
            using var predCpu = pred.cpu();
            using var gtCpu = yCtc.cpu();
            var predFlat = predCpu.data<long>().ToArray();
            var gtFlat = gtCpu.data<long>().ToArray();
            var timeSteps = (int)pred.shape[1];
            var maxTextLength = cfg.MaxTextLength;

            var vocab = ctcEncoder.Characters;
            var sampleCount = Math.Min(3, batchData.Batch);
            for (var i = 0; i < sampleCount; i++)
            {
                var predSeq = predFlat.Skip(i * timeSteps).Take(timeSteps).ToArray();
                var gtSeq = gtFlat.Skip(i * maxTextLength).Take(maxTextLength).ToArray();
                var predText = DecodeCtcPrediction(predSeq, vocab);
                var gtText = DecodeLabel(gtSeq, vocab);
                _logger.LogInformation(
                    "Diagnostics: sample[{Idx}] pred=\"{Pred}\" label=\"{Label}\"",
                    i, predText.Length == 0 ? "(empty)" : predText, gtText.Length == 0 ? "(empty)" : gtText);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error logging training diagnostics");
        }
    }
}
