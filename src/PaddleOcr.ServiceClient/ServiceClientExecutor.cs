using PaddleOcr.Core.Cli;
using Microsoft.Extensions.Logging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;

namespace PaddleOcr.ServiceClient;

public sealed class ServiceClientExecutor : ICommandExecutor
{
    private static readonly HttpClient Http = new();

    public async Task<CommandResult> ExecuteAsync(string subCommand, PaddleOcr.Core.Cli.ExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!subCommand.Equals("test", StringComparison.OrdinalIgnoreCase))
        {
            return CommandResult.Fail($"Unsupported service subcommand: {subCommand}");
        }

        if (!context.Options.TryGetValue("--server_url", out var serverUrl))
        {
            return CommandResult.Fail("service test requires --server_url");
        }

        if (!context.Options.TryGetValue("--image_dir", out var imageDir))
        {
            return CommandResult.Fail("service test requires --image_dir");
        }

        if (!Uri.TryCreate(serverUrl, UriKind.Absolute, out var serverUri))
        {
            return CommandResult.Fail($"invalid --server_url: {serverUrl}");
        }

        var imageFiles = EnumerateImages(imageDir).ToList();
        if (imageFiles.Count == 0)
        {
            return CommandResult.Fail($"No image found in: {imageDir}");
        }

        var visualize = context.Options.TryGetValue("--visualize", out var v) &&
                        v.Equals("true", StringComparison.OrdinalIgnoreCase);
        var outputDir = context.Options.TryGetValue("--output", out var outDir) ? outDir : "./hubserving_result";
        var parallel = ParseInt(context, "--parallel", 1, 1, 64);
        var timeoutMs = ParseInt(context, "--timeout_ms", 15000, 100, 300000);
        var retries = ParseInt(context, "--retries", 0, 0, 10);
        var stressRounds = ParseInt(context, "--stress_rounds", 1, 1, 1000);
        var dumpFailures = context.Options.TryGetValue("--dump_failures", out var df) &&
                           df.Equals("true", StringComparison.OrdinalIgnoreCase);
        if (visualize)
        {
            Directory.CreateDirectory(outputDir);
        }
        if (dumpFailures)
        {
            Directory.CreateDirectory(Path.Combine(outputDir, "failures"));
        }

        var totalMs = 0d;
        var okCount = 0;
        var failCount = 0;
        var totalRequests = imageFiles.Count * stressRounds;
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = parallel,
            CancellationToken = cancellationToken
        };
        var sync = new object();
        var workItems = Enumerable.Range(0, stressRounds)
            .SelectMany(round => imageFiles.Select(file => (Round: round + 1, Image: file)))
            .ToList();
        await Parallel.ForEachAsync(workItems, options, async (item, ct) =>
        {
            var imageFile = item.Image;
            byte[] bytes;
            try
            {
                bytes = await File.ReadAllBytesAsync(imageFile, ct);
            }
            catch (Exception ex)
            {
                context.Logger.LogWarning(ex, "error in loading image: {Image}", imageFile);
                lock (sync)
                {
                    failCount++;
                }
                if (dumpFailures)
                {
                    DumpFailure(outputDir, imageFile, item.Round, "read_error", ex.Message);
                }
                return;
            }

            HttpResponseMessage? resp = null;
            string? failureReason = null;
            var payload = new { images = new[] { Convert.ToBase64String(bytes) } };
            var sw = Stopwatch.StartNew();
            for (var attempt = 0; attempt <= retries; attempt++)
            {
                using var reqCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                reqCts.CancelAfter(timeoutMs);
                try
                {
                    resp = await Http.PostAsJsonAsync(serverUri, payload, reqCts.Token);
                    if (resp.IsSuccessStatusCode)
                    {
                        failureReason = null;
                        break;
                    }

                    failureReason = $"http_{(int)resp.StatusCode}";
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    failureReason = $"timeout_{timeoutMs}ms";
                }
                catch (Exception ex)
                {
                    failureReason = ex.GetType().Name;
                }
            }
            sw.Stop();

            lock (sync)
            {
                totalMs += sw.Elapsed.TotalMilliseconds;
            }
            if (resp is null || !resp.IsSuccessStatusCode)
            {
                context.Logger.LogWarning(
                    "service request failed: image={Image}, round={Round}, reason={Reason}",
                    imageFile, item.Round, failureReason ?? "unknown");
                lock (sync)
                {
                    failCount++;
                }
                if (dumpFailures)
                {
                    DumpFailure(outputDir, imageFile, item.Round, "request_failed", failureReason ?? "unknown");
                }
                return;
            }

            var json = await resp.Content.ReadAsStringAsync(ct);
            context.Logger.LogInformation("Predict time of {Image} (round={Round}): {Ms:F1}ms", imageFile, item.Round, sw.Elapsed.TotalMilliseconds);
            lock (sync)
            {
                okCount++;
            }
            if (!visualize)
            {
                return;
            }

            TryVisualizeResult(imageFile, json, outputDir, context);
        });

        if (okCount == 0)
        {
            return CommandResult.Fail("service test finished with 0 successful responses.");
        }

        var avgMs = totalMs / okCount;
        WriteReport(outputDir, new
        {
            total_requests = totalRequests,
            success = okCount,
            failed = failCount,
            avg_time_ms = avgMs,
            parallel,
            timeout_ms = timeoutMs,
            retries,
            stress_rounds = stressRounds,
            generated_at_utc = DateTime.UtcNow
        });
        return CommandResult.Ok(
            $"service test completed: success={okCount}/{totalRequests}, failed={failCount}, avg_time_ms={avgMs:F2}, parallel={parallel}, timeout_ms={timeoutMs}, retries={retries}, stress_rounds={stressRounds}");
    }

    private static IEnumerable<string> EnumerateImages(string path)
    {
        if (File.Exists(path))
        {
            yield return path;
            yield break;
        }

        if (!Directory.Exists(path))
        {
            yield break;
        }

        var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        };

        foreach (var file in Directory.EnumerateFiles(path, "*.*", SearchOption.TopDirectoryOnly))
        {
            if (exts.Contains(Path.GetExtension(file)))
            {
                yield return file;
            }
        }
    }

    private static void TryVisualizeResult(string imageFile, string responseJson, string outputDir, PaddleOcr.Core.Cli.ExecutionContext context)
    {
        try
        {
            using var doc = JsonDocument.Parse(responseJson);
            if (!doc.RootElement.TryGetProperty("results", out var results) ||
                results.ValueKind != JsonValueKind.Array ||
                results.GetArrayLength() == 0)
            {
                return;
            }

            var first = results[0];
            if (first.ValueKind != JsonValueKind.Array)
            {
                return;
            }

            var polygons = new List<PointF[]>();
            foreach (var item in first.EnumerateArray())
            {
                if (!item.TryGetProperty("text_region", out var region) || region.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                var poly = new List<PointF>();
                foreach (var p in region.EnumerateArray())
                {
                    if (p.ValueKind != JsonValueKind.Array || p.GetArrayLength() < 2)
                    {
                        continue;
                    }

                    var x = p[0].GetSingle();
                    var y = p[1].GetSingle();
                    poly.Add(new PointF(x, y));
                }

                if (poly.Count >= 2)
                {
                    polygons.Add(poly.ToArray());
                }
            }

            if (polygons.Count == 0)
            {
                return;
            }

            using var image = Image.Load<Rgb24>(imageFile);
            image.Mutate(ctx =>
            {
                foreach (var poly in polygons)
                {
                    ctx.DrawPolygon(Color.Lime, 2f, poly);
                }
            });
            var saveFile = Path.Combine(outputDir, Path.GetFileName(imageFile));
            image.Save(saveFile);
            context.Logger.LogInformation("The visualized image saved in {Path}", saveFile);
        }
        catch (Exception ex)
        {
            context.Logger.LogWarning(ex, "failed to visualize server result for {Image}", imageFile);
        }
    }

    private static int ParseInt(PaddleOcr.Core.Cli.ExecutionContext context, string key, int fallback, int min, int max)
    {
        if (!context.Options.TryGetValue(key, out var text) || !int.TryParse(text, out var value))
        {
            return fallback;
        }

        return Math.Clamp(value, min, max);
    }

    private static void DumpFailure(string outputDir, string imageFile, int round, string kind, string detail)
    {
        var dir = Path.Combine(outputDir, "failures");
        Directory.CreateDirectory(dir);
        var file = Path.Combine(
            dir,
            $"{Path.GetFileNameWithoutExtension(imageFile)}_r{round}_{DateTime.UtcNow:yyyyMMddHHmmssfff}.json");
        var payload = JsonSerializer.Serialize(new
        {
            image = imageFile,
            round,
            kind,
            detail,
            at_utc = DateTime.UtcNow
        }, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(file, payload);
    }

    private static void WriteReport(string outputDir, object report)
    {
        Directory.CreateDirectory(outputDir);
        var file = Path.Combine(outputDir, "service_test_report.json");
        var json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(file, json);
    }
}
