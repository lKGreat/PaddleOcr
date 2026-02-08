using System.Text.Json;
using Microsoft.Extensions.Logging;
using PaddleOcr.Inference.Paddle;
using PaddleOcr.Training.Runtime;
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
        EnsureCharsetCoverage(cfg.TrainLabelFile, cfg.EvalLabelFile, charToId, _logger);
        var trainSet = new SimpleRecDataset(cfg.TrainLabelFile, cfg.DataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);
        var evalSet = new SimpleRecDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        _logger.LogInformation("Training(rec) device: {Device}", dev.type);
        _logger.LogInformation("runtime: requested={Requested}, cuda={Cuda}, amp={Amp}, reason={Reason}", runtime.RequestedDevice, runtime.UseCuda, runtime.UseAmp, runtime.Reason);
        _logger.LogInformation("Train samples: {TrainCount}, Eval samples: {EvalCount}, Vocab: {Vocab}", trainSet.Count, evalSet.Count, vocab.Count);

        using var model = new SimpleRecNet(vocab.Count + 1, cfg.MaxTextLength);
        model.to(dev);
        var lr = cfg.LearningRate;
        var resumeCkpt = cfg.ResumeTraining ? ResolveEvalCheckpoint(cfg) : null;
        if (!string.IsNullOrWhiteSpace(resumeCkpt))
        {
            TryLoadCheckpoint(model, resumeCkpt);
        }

        using var teacher = RecTeacherDistiller.TryCreate(cfg, _logger, shape.H, shape.W, cfg.MaxTextLength, vocab.Count + 1);
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
            var ceLossSum = 0f;
            var kdLossSum = 0f;
            var sampleCount = 0;
            foreach (var (images, labels, batch) in trainSet.GetBatches(cfg.BatchSize, shuffle: true, rng))
            {
                using var x = torch.tensor(images, dtype: ScalarType.Float32).reshape(batch, 3, shape.H, shape.W).to(dev);
                using var y = torch.tensor(labels, dtype: ScalarType.Int64).reshape(batch, cfg.MaxTextLength).to(dev);
                optimizer.zero_grad();
                using var logits = model.call(x); // [B,T,V]

                var ceLoss = functional.cross_entropy(logits.reshape(batch * cfg.MaxTextLength, vocab.Count + 1), y.reshape(batch * cfg.MaxTextLength));
                Tensor? kdLoss = null;
                Tensor? totalLoss = ceLoss;
                try
                {
                    if (distillEnabled && teacher is not null)
                    {
                        var teacherBatch = teacher.Run(images, batch, shape.H, shape.W);
                        using var teacherLogits = torch.tensor(teacherBatch.Data, dtype: ScalarType.Float32)
                            .reshape(batch, teacherBatch.TimeSteps, teacherBatch.NumClasses)
                            .to(dev);
                        kdLoss = ComputeDistillLoss(logits, teacherLogits, cfg.DistillTemperature);
                        totalLoss = ceLoss * (1f - cfg.DistillWeight) + kdLoss * cfg.DistillWeight;
                    }

                    totalLoss.backward();
                    if (cfg.GradClipNorm > 0f)
                    {
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GradClipNorm);
                    }

                    optimizer.step();

                    var totalLossVal = totalLoss.ToSingle();
                    var ceLossVal = ceLoss.ToSingle();
                    var kdLossVal = kdLoss?.ToSingle() ?? 0f;
                    lossSum += totalLossVal * batch;
                    ceLossSum += ceLossVal * batch;
                    kdLossSum += kdLossVal * batch;
                    sampleCount += batch;
                }
                finally
                {
                    if (!ReferenceEquals(totalLoss, ceLoss))
                    {
                        totalLoss?.Dispose();
                    }

                    kdLoss?.Dispose();
                    ceLoss.Dispose();
                }
            }

            var trainLoss = sampleCount == 0 ? 0f : lossSum / sampleCount;
            var trainCeLoss = sampleCount == 0 ? 0f : ceLossSum / sampleCount;
            var trainKdLoss = sampleCount == 0 ? 0f : kdLossSum / sampleCount;
            var evalMetrics = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, cfg.MaxTextLength, dev, vocab);

            if (evalMetrics.Accuracy > bestAcc)
            {
                bestAcc = evalMetrics.Accuracy;
                staleEpochs = 0;
                SaveCheckpoint(cfg.SaveModelDir, model, "best.pt");
            }
            else
            {
                staleEpochs++;
            }

            SaveCheckpoint(cfg.SaveModelDir, model, "latest.pt");
            if (distillEnabled)
            {
                _logger.LogInformation(
                    "epoch={Epoch}/{Total} train_loss={Loss:F4} train_ce={Ce:F4} train_kd={Kd:F4} eval_acc={EvalAcc:F4} eval_char_acc={CharAcc:F4} eval_edit={Edit:F4}",
                    epoch,
                    cfg.EpochNum,
                    trainLoss,
                    trainCeLoss,
                    trainKdLoss,
                    evalMetrics.Accuracy,
                    evalMetrics.CharacterAccuracy,
                    evalMetrics.AvgEditDistance);
            }
            else
            {
                _logger.LogInformation(
                    "epoch={Epoch}/{Total} train_loss={Loss:F4} eval_acc={EvalAcc:F4} eval_char_acc={CharAcc:F4} eval_edit={Edit:F4}",
                    epoch,
                    cfg.EpochNum,
                    trainLoss,
                    evalMetrics.Accuracy,
                    evalMetrics.CharacterAccuracy,
                    evalMetrics.AvgEditDistance);
            }

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
        SaveSummary(cfg, summary, earlyStopped, resumeCkpt);
        return summary;
    }

    public EvaluationSummary Eval(TrainingConfigView cfg)
    {
        var shape = cfg.RecImageShape;
        var (charToId, vocab) = SimpleRecDataset.LoadDictionary(cfg.RecCharDictPath, cfg.UseSpaceChar);
        var evalSet = new SimpleRecDataset(cfg.EvalLabelFile, cfg.EvalDataDir, shape.H, shape.W, cfg.MaxTextLength, charToId);

        var runtime = TrainingDeviceResolver.Resolve(cfg);
        var dev = runtime.Device;
        using var model = new SimpleRecNet(vocab.Count + 1, cfg.MaxTextLength);
        model.to(dev);

        var ckpt = ResolveEvalCheckpoint(cfg);
        if (!string.IsNullOrWhiteSpace(ckpt))
        {
            TryLoadCheckpoint(model, ckpt);
        }

        var metrics = Evaluate(model, evalSet, cfg.EvalBatchSize, shape.H, shape.W, cfg.MaxTextLength, dev, vocab);
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

    private static Tensor ComputeDistillLoss(Tensor studentLogits, Tensor teacherLogits, float temperature)
    {
        using var studentScaled = studentLogits / temperature;
        using var teacherScaled = teacherLogits / temperature;
        using var studentLogProb = functional.log_softmax(studentScaled, dim: -1);
        using var teacherProb = functional.softmax(teacherScaled, dim: -1);
        using var teacherLogProb = functional.log_softmax(teacherScaled, dim: -1);
        using var tokenKl = (teacherProb * (teacherLogProb - studentLogProb)).sum(dim: -1);
        return tokenKl.mean() * (temperature * temperature);
    }

    private static RecEvalMetrics Evaluate(SimpleRecNet model, SimpleRecDataset evalSet, int batchSize, int h, int w, int maxTextLength, Device dev, IReadOnlyList<char> vocab)
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

                if (!RecLabelLineParser.TryParse(line, out _, out var text))
                {
                    continue;
                }

                foreach (var ch in text)
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
}

internal sealed record RecEvalMetrics(float Accuracy, float CharacterAccuracy, float AvgEditDistance);

internal sealed class RecTeacherDistiller : IDisposable
{
    private readonly PaddleNative? _native;
    private readonly PaddleNative.PaddlePredictor? _predictor;
    private readonly SimpleRecNet? _torchTeacher;
    private readonly Device _teacherDevice;
    private readonly int _imageH;
    private readonly int _imageW;

    private RecTeacherDistiller(
        PaddleNative native,
        PaddleNative.PaddlePredictor predictor,
        int timeSteps,
        int numClasses,
        int imageH,
        int imageW)
    {
        _native = native;
        _predictor = predictor;
        _imageH = imageH;
        _imageW = imageW;
        _teacherDevice = torch.CPU;
        TimeSteps = timeSteps;
        NumClasses = numClasses;
    }

    private RecTeacherDistiller(SimpleRecNet teacher, Device teacherDevice, int timeSteps, int numClasses, int imageH, int imageW)
    {
        _torchTeacher = teacher;
        _teacherDevice = teacherDevice;
        _imageH = imageH;
        _imageW = imageW;
        TimeSteps = timeSteps;
        NumClasses = numClasses;
    }

    public int TimeSteps { get; }

    public int NumClasses { get; }

    public static RecTeacherDistiller? TryCreate(
        TrainingConfigView cfg,
        ILogger logger,
        int imageH,
        int imageW,
        int studentTimeSteps,
        int studentNumClasses)
    {
        if (string.IsNullOrWhiteSpace(cfg.TeacherModelDir))
        {
            return null;
        }

        if (cfg.DistillWeight <= 0f)
        {
            throw new InvalidOperationException("Global.teacher_model_dir is set but Global.distill_weight <= 0. Set distill_weight in (0,1].");
        }

        var teacherPath = cfg.TeacherModelDir!;
        if (!Directory.Exists(teacherPath) && !File.Exists(teacherPath))
        {
            throw new FileNotFoundException($"Teacher model directory not found: {teacherPath}");
        }

        var torchTeacherPath = ResolveTorchTeacherCheckpoint(teacherPath);
        if (!string.IsNullOrWhiteSpace(torchTeacherPath))
        {
            return CreateTorchTeacher(
                torchTeacherPath!,
                logger,
                imageH,
                imageW,
                studentTimeSteps,
                studentNumClasses,
                cfg.StrictTeacherStudent);
        }

        if (!Directory.Exists(teacherPath))
        {
            throw new FileNotFoundException($"Teacher model directory not found: {teacherPath}");
        }

        var graphExists = File.Exists(Path.Combine(teacherPath, "inference.json")) ||
                          File.Exists(Path.Combine(teacherPath, "inference.pdmodel"));
        var paramsExists = File.Exists(Path.Combine(teacherPath, "inference.pdiparams"));
        if (!graphExists || !paramsExists)
        {
            throw new FileNotFoundException(
                $"Teacher model dir must contain inference graph + params: {teacherPath} (need inference.json|inference.pdmodel and inference.pdiparams), " +
                "or a torch teacher checkpoint (*.pt).");
        }

        PaddleNative? native = null;
        PaddleNative.PaddlePredictor? predictor = null;
        try
        {
            native = PaddleNative.Create(cfg.TeacherPaddleLibDir);
            predictor = native.CreatePredictor(teacherPath);

            var dryInput = new float[3 * imageH * imageW];
            var dry = predictor.Run(dryInput, [1, 3, imageH, imageW], 1f);
            if (dry.Dims.Length != 3)
            {
                throw new InvalidOperationException(
                    $"Teacher output must be rank-3 [B,T,C], but got [{string.Join(",", dry.Dims)}]");
            }

            var teacherTime = dry.Dims[1];
            var teacherClasses = dry.Dims[2];
            var mismatch = teacherTime != studentTimeSteps || teacherClasses != studentNumClasses;
            if (mismatch)
            {
                var msg =
                    $"Teacher/student logits shape mismatch: teacher=[T:{teacherTime}, C:{teacherClasses}], student=[T:{studentTimeSteps}, C:{studentNumClasses}]. " +
                    "Use same dict/max_text_length as teacher model.";
                if (cfg.StrictTeacherStudent)
                {
                    throw new InvalidOperationException(msg);
                }

                logger.LogWarning("{Message} distillation disabled because strict_teacher_student=false", msg);
                predictor.Dispose();
                native.Dispose();
                return null;
            }

            logger.LogInformation(
                "teacher loaded: dir={Dir}, logits=[T:{T}, C:{C}]",
                cfg.TeacherModelDir,
                teacherTime,
                teacherClasses);

            return new RecTeacherDistiller(native, predictor, teacherTime, teacherClasses, imageH, imageW);
        }
        catch
        {
            predictor?.Dispose();
            native?.Dispose();
            throw;
        }
    }

    public (float[] Data, int TimeSteps, int NumClasses) Run(float[] batchImages, int batchSize, int imageH, int imageW)
    {
        if (imageH != _imageH || imageW != _imageW)
        {
            throw new InvalidOperationException($"Teacher image shape mismatch: expected [{_imageH},{_imageW}], got [{imageH},{imageW}]");
        }

        if (_predictor is not null)
        {
            var result = _predictor.Run(batchImages, [batchSize, 3, imageH, imageW], 1f);
            if (result.Dims.Length != 3)
            {
                throw new InvalidOperationException($"Teacher output must be rank-3 [B,T,C], but got [{string.Join(",", result.Dims)}]");
            }

            if (result.Dims[0] != batchSize || result.Dims[1] != TimeSteps || result.Dims[2] != NumClasses)
            {
                throw new InvalidOperationException(
                    $"Teacher output shape changed unexpectedly: got [{string.Join(",", result.Dims)}], expected [{batchSize},{TimeSteps},{NumClasses}]");
            }

            return (result.Data, TimeSteps, NumClasses);
        }

        if (_torchTeacher is null)
        {
            throw new InvalidOperationException("Teacher is not initialized.");
        }

        using var noGrad = torch.no_grad();
        using var x = torch.tensor(batchImages, dtype: ScalarType.Float32)
            .reshape(batchSize, 3, imageH, imageW)
            .to(_teacherDevice);
        using var logits = _torchTeacher.call(x).to(torch.CPU);
        var output = logits.data<float>().ToArray();
        return (output, TimeSteps, NumClasses);
    }

    public void Dispose()
    {
        _predictor?.Dispose();
        _native?.Dispose();
        _torchTeacher?.Dispose();
    }

    private static string? ResolveTorchTeacherCheckpoint(string teacherPath)
    {
        if (File.Exists(teacherPath) && Path.GetExtension(teacherPath).Equals(".pt", StringComparison.OrdinalIgnoreCase))
        {
            return teacherPath;
        }

        if (!Directory.Exists(teacherPath))
        {
            return null;
        }

        var best = Path.Combine(teacherPath, "best.pt");
        if (File.Exists(best))
        {
            return best;
        }

        var latest = Path.Combine(teacherPath, "latest.pt");
        if (File.Exists(latest))
        {
            return latest;
        }

        return null;
    }

    private static RecTeacherDistiller? CreateTorchTeacher(
        string checkpointPath,
        ILogger logger,
        int imageH,
        int imageW,
        int studentTimeSteps,
        int studentNumClasses,
        bool strictTeacherStudent)
    {
        var device = torch.CPU;
        var teacher = new SimpleRecNet(studentNumClasses, studentTimeSteps);
        teacher.to(device);
        try
        {
            teacher.load(checkpointPath);
            teacher.eval();
        }
        catch
        {
            teacher.Dispose();
            throw;
        }

        logger.LogInformation(
            "teacher loaded (torch checkpoint): file={Path}, logits=[T:{T}, C:{C}]",
            checkpointPath,
            studentTimeSteps,
            studentNumClasses);

        if (!strictTeacherStudent)
        {
            // Current torch teacher mode is shape-locked to student architecture by construction.
        }

        return new RecTeacherDistiller(teacher, device, studentTimeSteps, studentNumClasses, imageH, imageW);
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
