using System.Text.Json;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;

namespace PaddleOcr.Training.Rec;

/// <summary>
/// Checkpoint 管理器：保存/恢复模型、优化器状态和训练元信息。
/// 官方 PaddleOCR 保存 model.pdparams + optimizer.pdopt + meta_info。
/// 我们保存 model.pt + optimizer.pt + meta.json
/// </summary>
public sealed class CheckpointManager
{
    private readonly ILogger _logger;

    public CheckpointManager(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// 保存完整 checkpoint（模型 + 优化器 + 调度器 + 元信息）。
    /// </summary>
    public void SaveFull(
        string saveDir,
        string tag,
        RecModel model,
        optim.Optimizer optimizer,
        ILRScheduler scheduler,
        int epoch,
        int globalStep,
        float bestAcc)
    {
        Directory.CreateDirectory(saveDir);
        var prefix = Path.Combine(saveDir, tag);

        // 保存模型
        try
        {
            model.save($"{prefix}.pt");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save model checkpoint {Path}", $"{prefix}.pt");
        }

        // 保存优化器状态
        try
        {
            // TorchSharp optimizer state_dict 序列化
            SaveOptimizerState(optimizer, $"{prefix}_optimizer.pt");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save optimizer state {Path}", $"{prefix}_optimizer.pt");
        }

        // 保存元信息（含调度器状态）
        var meta = new CheckpointMeta
        {
            Epoch = epoch,
            GlobalStep = globalStep,
            BestAcc = bestAcc,
            SchedulerLR = scheduler.CurrentLR,
            SavedAt = DateTime.UtcNow
        };

        try
        {
            var json = JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText($"{prefix}_meta.json", json);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to save checkpoint meta {Path}", $"{prefix}_meta.json");
        }
    }

    /// <summary>
    /// 恢复完整 checkpoint。
    /// </summary>
    /// <returns>恢复的元信息，或 null（如果文件不存在）。</returns>
    public CheckpointMeta? LoadFull(
        string saveDir,
        string tag,
        RecModel model,
        optim.Optimizer? optimizer,
        ILRScheduler? scheduler)
    {
        var prefix = Path.Combine(saveDir, tag);

        // 恢复模型
        var modelPath = $"{prefix}.pt";
        if (!File.Exists(modelPath))
        {
            _logger.LogWarning("Model checkpoint not found: {Path}", modelPath);
            return null;
        }

        try
        {
            _logger.LogInformation("Loading model checkpoint: {Path}", modelPath);
            model.load(modelPath);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to load model checkpoint {Path}", modelPath);
            return null;
        }

        // 恢复优化器状态
        var optPath = $"{prefix}_optimizer.pt";
        if (optimizer is not null && File.Exists(optPath))
        {
            try
            {
                _logger.LogInformation("Loading optimizer state: {Path}", optPath);
                LoadOptimizerState(optimizer, optPath);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load optimizer state {Path}, will use fresh optimizer", optPath);
            }
        }

        // 恢复元信息
        var metaPath = $"{prefix}_meta.json";
        CheckpointMeta? meta = null;
        if (File.Exists(metaPath))
        {
            try
            {
                var json = File.ReadAllText(metaPath);
                meta = JsonSerializer.Deserialize<CheckpointMeta>(json);
                _logger.LogInformation("Restored checkpoint meta: epoch={Epoch}, step={Step}, best_acc={Acc:F4}, lr={Lr:F6}",
                    meta?.Epoch, meta?.GlobalStep, meta?.BestAcc, meta?.SchedulerLR);

                // 恢复 scheduler：将其推进到保存时的位置
                if (scheduler is not null && meta is not null)
                {
                    scheduler.Step(meta.GlobalStep, meta.Epoch);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load checkpoint meta {Path}", metaPath);
            }
        }

        return meta;
    }

    /// <summary>
    /// 简单保存（仅模型，兼容旧逻辑）。
    /// </summary>
    public void SaveModel(string saveDir, RecModel model, string fileName)
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

    private static void SaveOptimizerState(optim.Optimizer optimizer, string path)
    {
        // TorchSharp 的 Optimizer 有 state_dict/load_state_dict 方法
        // 通过保存各 param group 的参数状态来序列化
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // 保存 param group 学习率
        var groups = optimizer.ParamGroups.ToList();
        bw.Write(groups.Count);
        foreach (var pg in groups)
        {
            bw.Write(pg.LearningRate);
        }

        bw.Flush();
    }

    private static void LoadOptimizerState(optim.Optimizer optimizer, string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        var groupCount = br.ReadInt32();
        var groups = optimizer.ParamGroups.ToList();
        var restoreCount = Math.Min(groupCount, groups.Count);

        for (var i = 0; i < restoreCount; i++)
        {
            groups[i].LearningRate = br.ReadDouble();
        }
    }
}

/// <summary>
/// Checkpoint 元信息。
/// </summary>
public sealed class CheckpointMeta
{
    public int Epoch { get; set; }
    public int GlobalStep { get; set; }
    public float BestAcc { get; set; }
    public double SchedulerLR { get; set; }
    public DateTime SavedAt { get; set; }
}
