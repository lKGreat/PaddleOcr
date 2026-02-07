# PaddleOcr (.NET 10, Native-first)

This repository is a C#/.NET 10 native implementation workspace for PaddleOCR toolchain migration.

## Current Status

- Multi-project architecture is bootstrapped.
- YAML config loader and `-o` override merger are implemented.
- `pocr` command router is implemented for all planned command surfaces.
- Asset sync script is available:
  - `scripts/sync-assets.ps1`
- Initial ONNX execution path for `infer system` is implemented:
  - Requires `--use_onnx true`
  - Requires `--image_dir` and `--rec_model_dir`
  - Optional `--det_model_dir`, `--cls_model_dir`
  - Produces `system_results.txt` in `--draw_img_save_dir` (default `./inference_results`)

## Commands (implemented surface)

- `pocr train -c <config> -o K=V ...`
- `pocr eval -c <config> -o K=V ...`
- `pocr export -c <config> -o K=V ...`
- `pocr export-onnx -c <config> -o K=V ...`
- `pocr export-center -c <config> -o K=V ...`
- `pocr infer <det|rec|cls|e2e|kie|kie-ser|kie-re|table|sr|system> ...`
- `pocr convert json2pdmodel ...`
- `pocr service test --server_url ... --image_dir ...`
- `pocr e2e <convert-label|eval> ...`

## Notes

- Full OCR post-processing and training kernel implementation is still in progress.
- Current `infer system` ONNX path executes sessions and records output tensor shapes as integration baseline.

