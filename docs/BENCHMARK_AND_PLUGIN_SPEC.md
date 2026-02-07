# Benchmark and Plugin Spec (1.x)

Last updated: 2026-02-07

## Benchmark command

Command:

```bash
pocr benchmark run --scenario <infer:system|service:test|e2e:eval|export:export-onnx|train:train> \
  [--warmup N] [--iterations N] [--continue_on_error true] [--report_json <path>] \
  [scenario options...]
```

Output:
- Writes JSON report to `--report_json` or default `benchmark_results/<scenario>_<timestamp>.json`.
- Includes summary metrics: `avg`, `p50`, `p95`, `p99`, `throughput_per_sec`, and per-iteration samples.

## Plugin package command

Command:

```bash
pocr plugin validate-package --package_dir <dir>
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
- `entry_assembly`
- `entry_type`

Optional:
- `files`: list of additional files that must exist under package dir.
