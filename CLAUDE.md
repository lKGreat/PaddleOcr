# CLAUDE.md - PaddleOCR (.NET 10 Native)

## Project Overview

This is a **C#/.NET 10 native reimplementation** of the PaddleOCR toolchain. It is **not** the original Python PaddleOCR — it is a ground-up C# port targeting .NET 10 with ONNX Runtime inference, TorchSharp-based training, and a modular multi-project architecture.

The CLI entry point is `pocr`, which dispatches to specialized executors for training, inference, export, benchmarking, plugins, and diagnostics.

## Quick Reference

```bash
# Build the entire solution
dotnet build PaddleOcr.slnx

# Build in Release mode
dotnet build PaddleOcr.slnx -c Release

# Run tests
dotnet test tests/PaddleOcr.Tests/PaddleOcr.Tests.csproj

# Run the CLI tool
dotnet run --project src/PaddleOcr.Tools/PaddleOcr.Tools.csproj -- <command> [args]

# Run CLI in Release mode (used by CI scripts)
dotnet run --project src/PaddleOcr.Tools/PaddleOcr.Tools.csproj -c Release -- <command> [args]
```

## Solution Structure

```
PaddleOcr.slnx                          # Visual Studio solution (12 projects)
├── src/
│   ├── PaddleOcr.Core/                  # Base interfaces, CLI abstractions, error types
│   ├── PaddleOcr.Config/                # YAML config loading, merging, override parsing
│   ├── PaddleOcr.Models/                # Shared model definitions (RecAlgorithm, RecTypes)
│   ├── PaddleOcr.Data/                  # Dataset handling, augmentation, label encoding
│   ├── PaddleOcr.Training/              # Training pipelines (DET, REC, CLS)
│   ├── PaddleOcr.Inference/             # ONNX inference, pre/post-processing
│   ├── PaddleOcr.Export/                # Model export (ONNX, Paddle, Center)
│   ├── PaddleOcr.ServiceClient/         # Remote service API client
│   ├── PaddleOcr.Benchmark/             # Performance benchmarking
│   ├── PaddleOcr.Plugins/              # Plugin system with trust validation
│   └── PaddleOcr.Tools/                # CLI entry point (pocr command router)
├── tests/
│   └── PaddleOcr.Tests/                # xUnit test suite (16 test files)
├── assets/
│   ├── configs/                         # YAML model configs (det/, rec/, cls/, kie/, table/, etc.)
│   │   └── local/                       # Tiny CI-fast configs for testing
│   ├── dicts/                           # 79+ language character dictionaries
│   └── samples/                         # Test fixtures (tiny_det/, tiny_rec/, tiny_cls/)
├── docs/                                # Strategic and operational documentation
└── scripts/                             # PowerShell build/CI scripts
```

## Dependency Graph

```
PaddleOcr.Core          (no project deps — base interfaces)
  ↑
PaddleOcr.Config        → Core
PaddleOcr.Models        → Core
  ↑
PaddleOcr.Data          → Core, Config
PaddleOcr.Inference     → Core, Config, Models
PaddleOcr.Training      → Core, Config, Data, Models
PaddleOcr.Export        → Core, Config
PaddleOcr.ServiceClient → Core
PaddleOcr.Benchmark     → Core
PaddleOcr.Plugins       → Core
  ↑
PaddleOcr.Tools         → All of the above (CLI entry point)
```

## Key Packages

| Package | Version | Used By |
|---------|---------|---------|
| Microsoft.ML.OnnxRuntime | 1.24.1 | Inference, Export |
| TorchSharp + TorchSharp-cpu | 0.105.2 | Training |
| SixLabors.ImageSharp | 3.1.12 | Inference, Training, Data, ServiceClient |
| YamlDotNet | 16.3.0 | Config |
| Microsoft.Extensions.Logging | 10.0.2 | Core, Tools |
| xunit | 2.9.3 | Tests |
| FluentAssertions | 8.8.0 | Tests |

## Architecture & Patterns

### Command Pattern (CLI Routing)

All subcommands implement `ICommandExecutor`:

```csharp
public interface ICommandExecutor
{
    Task<CommandResult> ExecuteAsync(
        string subCommand,
        ExecutionContext context,
        CancellationToken cancellationToken = default);
}
```

`PocrApp.RunAsync()` parses CLI args via `CommandLine.Parse()` and dispatches to the appropriate executor based on the root command (train, eval, infer, export, etc.).

### Configuration System

- YAML-based configs loaded by `ConfigLoader` (uses YamlDotNet)
- CLI overrides via `-o K=V` syntax, merged by `ConfigMerger`/`OverrideParser`
- Config files organized under `assets/configs/` by task type (det/, rec/, cls/, kie/, table/, etc.)
- Local/CI test configs in `assets/configs/local/` (tiny datasets, minimal epochs)

### Key Executors

| Executor | Project | Handles |
|----------|---------|---------|
| `TrainingExecutor` | Training | `train`, `eval` |
| `InferenceExecutor` | Inference | `infer <det\|rec\|cls\|e2e\|kie\|table\|sr\|system>` |
| `ExportExecutor` | Export | `export`, `export-pdmodel`, `export-onnx`, `export-center` |
| `ServiceClientExecutor` | ServiceClient | `service test` |
| `E2eToolsExecutor` | Data | `e2e <convert-label\|eval\|prepare-rec-det>` |
| `BenchmarkExecutor` | Benchmark | `benchmark run` |
| `PluginExecutor` | Plugins | `plugin <validate-package\|verify-trust\|load-runtime\|...>` |

### Training Module (largest subsystem)

Located in `src/PaddleOcr.Training/`, organized by task:
- `Rec/` — Recognition training (73+ files): backbones, necks, heads, losses, schedulers
- `Det/` — Detection training
- `Cls/` — Classification training

Key classes: `ConfigDrivenRecTrainer`, `RecModelBuilder`, `RecLossBuilder`, `CheckpointManager`

### Inference Module

Located in `src/PaddleOcr.Inference/`:
- `Onnx/OnnxRunners.cs` — ONNX session management (largest file)
- `InferenceComponentRegistry` — Runtime component registration
- `DetInferenceExtensions.cs` — Detection algorithms (DB, DB++, EAST, SAST, PSE, FCE, CT)
- Preprocessors and postprocessors organized by task type

## Testing

### Framework

- **xUnit** 2.9.3 with **FluentAssertions** 8.8.0
- **coverlet** 6.0.4 for code coverage
- Test project: `tests/PaddleOcr.Tests/`

### Running Tests

```bash
# Run all tests
dotnet test tests/PaddleOcr.Tests/PaddleOcr.Tests.csproj

# Run with verbosity
dotnet test tests/PaddleOcr.Tests/PaddleOcr.Tests.csproj -v normal

# Run specific test class
dotnet test tests/PaddleOcr.Tests/PaddleOcr.Tests.csproj --filter "FullyQualifiedName~ConfigTests"

# Run with coverage
dotnet test tests/PaddleOcr.Tests/PaddleOcr.Tests.csproj --collect:"XPlat Code Coverage"
```

### Test Files

| File | Coverage Area |
|------|---------------|
| `PocrAppTests.cs` | End-to-end CLI command execution |
| `ConfigTests.cs` | YAML config loading and merging |
| `CommandLineTests.cs` | CLI argument parsing |
| `InferenceExecutorTests.cs` | Inference pipeline |
| `ExportTests.cs` | Model export workflows |
| `DetInferenceExtensionsTests.cs` | Detection post-processing |
| `DetMetricEvaluatorTests.cs` | Detection metrics (IOU, precision, recall) |
| `RecTrainingParityTests.cs` | Training reproducibility |
| `TrainingDetStabilityTests.cs` | Detection training stability |
| `BenchmarkExecutorTests.cs` | Benchmarking system |
| `PluginValidatorTests.cs` | Plugin validation and trust |
| `E2eToolsExecutorTests.cs` | End-to-end tool utilities |
| `PostprocessUtilsTests.cs` | Post-processing utilities |
| `PathResolutionTests.cs` | Path resolution logic |
| `TableResultSerializerTests.cs` | Table output serialization |

### CI/Smoke Scripts

```bash
# Fast smoke test (config + doctor checks)
pwsh scripts/smoke-ci-fast.ps1

# Full acceptance replay (generates markdown report)
pwsh scripts/replay-acceptance.ps1
```

These scripts use configs from `assets/configs/local/` (e.g., `table_ci_fast.yml`, `kie_ci_fast.yml`, `det_tiny.yml`).

## CLI Command Surface

```
pocr train -c <config> [-o K=V ...]
pocr eval -c <config> [-o K=V ...]
pocr export -c <config> [-o K=V ...]
pocr export-pdmodel -c <config> [--static_equivalence strict|compatible] [-o K=V ...]
pocr export-onnx -c <config> [-o K=V ...]
pocr export-center -c <config> [-o K=V ...]
pocr infer <det|rec|cls|e2e|kie|kie-ser|kie-re|table|sr|system> [options]
pocr convert json2pdmodel --json_model_dir <dir> --output_dir <dir> --config <yml>
pocr convert check-json-model --json_model_dir <dir>
pocr config check -c <config>
pocr config diff --base <path> --target <path>
pocr doctor check-models [-c <config>] [--det_model_dir <path>] [...]
pocr doctor parity-table-kie -c <config> [--mode all|table|kie]
pocr doctor det-parity -c <config>
pocr doctor train-det-ready -c <config>
pocr service test --server_url <url> --image_dir <dir> [options]
pocr e2e <convert-label|eval|prepare-rec-det> [args]
pocr benchmark run --scenario <...> [--profile smoke|balanced|stress] [options]
pocr plugin <validate-package|verify-trust|load-runtime|load-runtime-dir> [options]
```

## Code Conventions

- **Target framework**: .NET 10.0 (`net10.0`)
- **Nullable reference types**: Enabled globally (`<Nullable>enable</Nullable>`)
- **Implicit usings**: Enabled (`<ImplicitUsings>enable</ImplicitUsings>`)
- **File-scoped namespaces**: Used throughout (e.g., `namespace PaddleOcr.Tools;`)
- **Async/await**: All executor methods are async (`Task<CommandResult>`)
- **Dependency injection**: Constructor injection via concrete types wired in `Program.cs`
- **Logging**: `Microsoft.Extensions.Logging` with simple console provider
- **Error handling**: Custom `PocrException` for expected errors (exit code 2), generic exceptions (exit code 1)
- **Exit codes**: 0 = success, 1 = unexpected error, 2 = known/expected error

## Files Not to Commit

The `.gitignore` excludes:
- Build artifacts: `bin/`, `obj/`, `out/`, `TestResults/`
- Model files: `*.pdmodel`, `*.pdiparams`, `*.onnx`, `*.ckpt`, `*.safetensors`, `*.weights`
- Large data: `train_data/`, `pretrain_models/`, `datasets/`, `data/`
- Runtime output: `output/`, `inference_results/`, `benchmark_results/`, `plugin_packages/`
- IDE files: `.vs/`, `.vscode/`, `*.user`
- Agent files: `AGENTS.md`

Exception: `assets/samples/tiny_det/**` is tracked (CI test fixtures).

## Adding New Features

### Adding a new CLI command
1. Implement `ICommandExecutor` in the appropriate project
2. Add the command routing in `PocrApp.RunAsync()` switch expression
3. Wire the executor in `Program.cs`
4. Add tests in `tests/PaddleOcr.Tests/`

### Adding a new recognition backbone/head/loss
1. Add the implementation under `src/PaddleOcr.Training/Rec/` in the appropriate subdirectory (Backbones/, Heads/, Losses/)
2. Register it in the relevant builder (`RecModelBuilder`, `RecLossBuilder`)
3. Add corresponding config entries under `assets/configs/rec/`

### Adding a new detection algorithm
1. Add post-processing logic in `DetInferenceExtensions.cs`
2. Add corresponding config under `assets/configs/det/`
3. Update the algorithm dispatch in inference executor

### Adding a new test
1. Create a test class in `tests/PaddleOcr.Tests/` following the `*Tests.cs` naming pattern
2. Use xUnit `[Fact]` and `[Theory]` attributes
3. Use FluentAssertions for assertions (e.g., `result.Should().Be(expected)`)
4. Use tiny configs from `assets/configs/local/` for integration tests
