# Benchmark and Plugin Spec (1.x)

Last updated: 2026-02-07

## Benchmark command

Command:

```bash
pocr benchmark run --scenario <infer:system|service:test|e2e:eval|export:export-onnx|train:train> \
  [--profile smoke|balanced|stress] [--warmup N] [--iterations N] [--continue_on_error true] [--report_json <path>] \
  [scenario options...]
```

Output:
- Writes JSON report to `--report_json` or default `benchmark_results/<scenario>_<timestamp>.json`.
- Includes summary metrics: `avg`, `p50`, `p95`, `p99`, `throughput_per_sec`, and per-iteration samples.
- For `service:test`, profile presets inject defaults when flags are not explicitly set:
  - `smoke`: `parallel=1`, `timeout_ms=8000`, `retries=0`, `stress_rounds=1`
  - `balanced`: `parallel=4`, `timeout_ms=15000`, `retries=1`, `stress_rounds=2`
  - `stress`: `parallel=8`, `timeout_ms=20000`, `retries=2`, `stress_rounds=5`

Training benchmark fixture:
- `assets/configs/local/train_bench_rec_ci_fast.yml`
- Example:
  - `pocr benchmark run --scenario train:train -c assets/configs/local/train_bench_rec_ci_fast.yml --warmup 0 --iterations 1`

## Plugin package command

Command:

```bash
pocr plugin validate-package --package_dir <dir>
pocr plugin load-runtime --package_dir <dir>
pocr plugin load-runtime-dir --plugins_root <dir>
```

Required file layout:

```text
<package_dir>/
  plugin.json
  <entry_assembly>.dll
```

`plugin.json` required fields (`1.x`):
- `schema_version`: must start with `1.`
- `name`
- `version`
- `type`: `preprocess` | `postprocess` | `metric`
- `entry_assembly` and `entry_type` (required for assembly mode)

Runtime extension fields:
- `runtime_name`: optional registry key to register; defaults to `name`.
- `runtime_target`: required when `type=postprocess`, one of `det|rec|cls`.
- `alias_of`: optional alias mode source key; if present, assembly fields are optional.

Runtime contract interfaces (assembly mode):
- `PaddleOcr.Inference.Onnx.IInferencePreprocessPlugin`
- `PaddleOcr.Inference.Onnx.IDetPostprocessPlugin`
- `PaddleOcr.Inference.Onnx.IRecPostprocessPlugin`
- `PaddleOcr.Inference.Onnx.IClsPostprocessPlugin`

Optional:
- `files`: list of additional files that must exist under package dir.
