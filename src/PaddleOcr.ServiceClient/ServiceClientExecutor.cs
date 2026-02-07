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
        if (visualize)
        {
            Directory.CreateDirectory(outputDir);
        }

        var totalMs = 0d;
        var okCount = 0;
        foreach (var imageFile in imageFiles)
        {
            byte[] bytes;
            try
            {
                bytes = await File.ReadAllBytesAsync(imageFile, cancellationToken);
            }
            catch (Exception ex)
            {
                context.Logger.LogWarning(ex, "error in loading image: {Image}", imageFile);
                continue;
            }

            var payload = new { images = new[] { Convert.ToBase64String(bytes) } };
            var sw = Stopwatch.StartNew();
            HttpResponseMessage resp;
            try
            {
                resp = await Http.PostAsJsonAsync(serverUri, payload, cancellationToken);
            }
            catch (Exception ex)
            {
                context.Logger.LogWarning(ex, "request failed: {Image}", imageFile);
                continue;
            }
            finally
            {
                sw.Stop();
            }

            totalMs += sw.Elapsed.TotalMilliseconds;
            if (!resp.IsSuccessStatusCode)
            {
                context.Logger.LogWarning("server returned {Code} for {Image}", (int)resp.StatusCode, imageFile);
                continue;
            }

            var json = await resp.Content.ReadAsStringAsync(cancellationToken);
            context.Logger.LogInformation("Predict time of {Image}: {Ms:F1}ms", imageFile, sw.Elapsed.TotalMilliseconds);
            okCount++;
            if (!visualize)
            {
                continue;
            }

            TryVisualizeResult(imageFile, json, outputDir, context);
        }

        if (okCount == 0)
        {
            return CommandResult.Fail("service test finished with 0 successful responses.");
        }

        var avgMs = totalMs / okCount;
        return CommandResult.Ok($"service test completed: success={okCount}/{imageFiles.Count}, avg_time_ms={avgMs:F2}");
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
}
