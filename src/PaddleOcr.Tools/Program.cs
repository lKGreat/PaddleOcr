using Microsoft.Extensions.Logging;
using PaddleOcr.Config;
using PaddleOcr.Core.Cli;
using PaddleOcr.Core.Errors;
using PaddleOcr.Data;
using PaddleOcr.Export;
using PaddleOcr.Inference;
using PaddleOcr.Benchmark;
using PaddleOcr.Plugins;
using PaddleOcr.ServiceClient;
using PaddleOcr.Tools;
using PaddleOcr.Training;

var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.ClearProviders();
    builder.AddSimpleConsole(opt =>
    {
        opt.SingleLine = true;
        opt.TimestampFormat = "HH:mm:ss ";
    });
});

var logger = loggerFactory.CreateLogger("pocr");

try
{
    var app = new PocrApp(
        logger,
        new ConfigLoader(),
        new TrainingExecutor(),
        new InferenceExecutor(),
        new ExportExecutor(),
        new ServiceClientExecutor(),
        new E2eToolsExecutor(),
        new BenchmarkExecutor(
            new TrainingExecutor(),
            new InferenceExecutor(),
            new ExportExecutor(),
            new ServiceClientExecutor(),
            new E2eToolsExecutor()),
        new PluginExecutor());

    var code = await app.RunAsync(args);
    Environment.ExitCode = code;
}
catch (PocrException ex)
{
    logger.LogError(ex.Message);
    Environment.ExitCode = 2;
}
catch (Exception ex)
{
    logger.LogError(ex, "Unhandled error");
    Environment.ExitCode = 1;
}
