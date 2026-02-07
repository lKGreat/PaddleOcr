# DET Capability Spec (Native .NET 10)

Last updated: 2026-02-07

## Scope

This spec defines the DET-first native capability baseline for `E:\codeding\PaddleOcr`.
Reference behavior is mapped from `E:\codeding\AI\PaddleOCR-3.3.2\tools` DET toolchain.

## Current Native Contract

- Command: `pocr infer det`
- Runtime: ONNX only (`--use_onnx=true`)
- Supported algorithm: `DB`, `DB++`, `EAST`, `SAST`, `PSE`, `FCE`, `CT`
- Output:
  - visualization images: `<draw_img_save_dir or default>/`
  - result file: `<draw_img_save_dir>/det_results.txt` or `--save_res_path`
  - metrics file: `<draw_img_save_dir>/det_metrics.json` or `--det_metrics_path` (`schema_version=1.1`)

## DET Infer Arguments

- `--image_dir` required
- `--det_model_dir` required
- `--det_algorithm` default `DB` (`DB|DB++`)
- `--det_db_thresh` default `0.3`
- `--det_db_box_thresh` default `0.6`
- `--det_db_unclip_ratio` default `1.5`
- `--det_db_score_mode` default `fast` (`fast|slow`)
- `--det_max_candidates` default `1000`
- `--det_box_type` default `quad` (`quad|poly`)
- `--det_limit_side_len` default `640`
- `--det_limit_type` default `max` (`max|min`)
- `--use_dilation` default `false`
- `--use_slice` default `false`
- `--det_slice_merge_iou` default `0.3`
- `--det_slice_min_bound_distance` default `50`
- `--save_res_path` optional
- `--det_gt_label` optional (label file in PaddleOCR format)
- `--det_eval_iou_thresh` default `0.5`
- `--det_metrics_path` optional

## Doctor Gate

- Command: `pocr doctor det-parity -c <config>`
- Checks:
  - `Architecture.model_type` preferred `det`
  - algorithm and postprocess pairing consistency
  - DET numeric parameter ranges
  - optional `Global.det_model_dir` path existence
  - optional `train-det-ready` gate if Train/Eval dataset exists

## Planned Extensions

1. Add configurable NMS strategy for polygon-heavy scenes.
2. Add offline DET benchmark baseline snapshots by algorithm/backbone.
3. Add richer error buckets in DET quality report (missed/over-segmentation).
4. Add parity checks against Paddle reference outputs on a pinned fixture set.

## Implemented In This Iteration

1. Strategy-based DET postprocess routing for `DB|DB++|EAST|SAST|PSE|FCE|CT`.
2. `det_metrics.json` upgraded to `schema_version=1.1` with `per_image` section.
3. Added `algorithm_runtime_profile` summary (avg preprocess/inference/postprocess/total ms).
4. Added dynamic DET resize resolution (`det_limit_type=max|min` + stride alignment) for dynamic ONNX inputs.
5. Added deterministic golden regression unit tests for each DET algorithm decode path.
6. Added per-image runtime export (`runtime_per_image`) and runtime fields in `per_image`.
7. Added DB score mode and max-candidate controls (`det_db_score_mode`, `det_max_candidates`).
8. Added slice inference mode (`use_slice`) with overlap merge and runtime tracking.
