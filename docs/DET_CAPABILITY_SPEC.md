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
  - metrics file: `<draw_img_save_dir>/det_metrics.json` or `--det_metrics_path`

## DET Infer Arguments

- `--image_dir` required
- `--det_model_dir` required
- `--det_algorithm` default `DB` (`DB|DB++`)
- `--det_db_thresh` default `0.3`
- `--det_db_box_thresh` default `0.6`
- `--det_db_unclip_ratio` default `1.5`
- `--det_box_type` default `quad` (`quad|poly`)
- `--det_limit_side_len` default `640`
- `--det_limit_type` default `max` (`max|min`)
- `--use_dilation` default `false`
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

1. Improve per-algorithm postprocess parity quality against Paddle reference implementation.
2. Add algorithm-aware preprocess parity including resize rules and dynamic shapes.
3. Add dataset-level detailed report (per-image precision/recall breakdown).
4. Add deterministic DET infer regression golden tests per algorithm.
