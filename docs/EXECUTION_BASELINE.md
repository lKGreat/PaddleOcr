# Execution Baseline Snapshot

Last updated: 2026-02-07

## Command Coverage (High-level)
- `train/eval`: native support for `cls/det/rec`
- `export/export-onnx/export-center`: native implemented
- `convert`: `json2pdmodel`, `check-json-model`
- `infer`: `det/rec/cls/system/e2e/sr/table/kie/kie-ser/kie-re` (native ONNX path)
- `service test`: HTTP batch runner with concurrency/timeout/retry/report
- `e2e`: `convert-label`, `eval`

## Runtime Defaults
- Infer outputs: `inference_results/<command>`
- Service report: `<output>/service_test_report.json`

## Verification Baseline
- Build: `dotnet build E:\codeding\PaddleOcr\PaddleOcr.slnx -c Release`
- Tests: `dotnet test E:\codeding\PaddleOcr\PaddleOcr.slnx -c Release`

