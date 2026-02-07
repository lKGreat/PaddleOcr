# Native Implementation Status Report

Last updated: 2026-02-07

## Summary

- Completed items: `1-13`, `15-17`, `19`
- Remaining items: `14`, `18`, `20`
- Current command coverage is tracked in `docs/CAPABILITY_MATRIX.md`

## Delivered in this phase

- Added native ONNX infer pipelines for:
  - `infer table`
  - `infer kie`
  - `infer kie-ser`
  - `infer kie-re`
- Upgraded `infer det` contract:
  - DB/DB++/EAST/SAST/PSE/FCE/CT algorithm selection
  - DET threshold/unclip/box-type/dilation/resize parameters
  - custom result file output (`--save_res_path`)
  - quality metrics output (`det_metrics.json`, optional GT label evaluation)
- Added DET parity doctor gate:
  - `doctor det-parity -c <config>`
- Enhanced `rec` training/eval metrics:
  - full-sequence accuracy
  - character accuracy
  - average edit distance
- Upgraded DET training core:
  - dual-branch output (`shrink`, `threshold`)
  - DB-style composite loss (BCE + L1 weighted)
  - configurable shrink/threshold map generation via config (`Loss.det_*`)
  - evaluation upgraded to detection-level matching (`precision/recall/fscore`) with configurable IoU threshold
- Unified export manifest schema with `schema_version=1.0`.
- Added `service test` runtime controls:
  - `--parallel`
  - `--timeout_ms`
- Standardized default infer output paths to `inference_results/<command>`.
- Extended convert family:
  - `convert json2pdmodel`
  - `convert check-json-model`
- Added tiny config templates for `table/kie/sr`.

## Remaining work

1. Item 14: introduce explicit pre/post-processing registry and plugin contracts.
2. Item 18: add deterministic-output regression tests for infer/export flows.
3. Item 20: consolidate final acceptance report with benchmark/smoke evidence.
