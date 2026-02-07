# Capability Matrix

Last updated: 2026-02-07

## Scope

This matrix tracks command parity between `PaddleOCR/tools` and this native `.NET 10` implementation.

Status keys:
- `Done`: runnable native implementation exists.
- `Partial`: command exists but lacks full feature parity.
- `Planned`: not implemented yet.

## Matrix

| Area | Command | PaddleOCR Tools | .NET Native | Status | Notes |
|---|---|---|---|---|---|
| Train | `train -c` | `tools/train.py` | `PaddleOcr.Training` | Done | `cls/det/rec` implemented |
| Train | `eval -c` | `tools/eval.py` | `PaddleOcr.Training` | Done | `cls/det/rec` implemented |
| Export | `export -c` | `tools/export_model.py` | `PaddleOcr.Export` | Done | Native checkpoint export |
| Export | `export-onnx -c` | `tools/export_to_onnx.py` | `PaddleOcr.Export` | Done | ONNX artifact export |
| Export | `export-center -c` | `tools/export_center.py` | `PaddleOcr.Export` | Done | Model center format |
| Convert | `convert json2pdmodel` | `tools/convert_json_to_pdmodel.py` | `PaddleOcr.Export` | Done | Shim conversion implemented |
| Infer | `infer det` | `tools/infer_det.py` | `PaddleOcr.Inference` | Partial | ONNX path with DB/DB++/EAST/SAST/PSE/FCE/CT, metrics, score_mode/max_candidates, and slice mode |
| Infer | `infer rec` | `tools/infer_rec.py` | `PaddleOcr.Inference` | Done | ONNX path |
| Infer | `infer cls` | `tools/infer_cls.py` | `PaddleOcr.Inference` | Done | ONNX path |
| Infer | `infer system` | `tools/infer/predict_system.py` | `PaddleOcr.Inference` | Done | ONNX det+rec+cls pipeline |
| Infer | `infer e2e` | `tools/infer_e2e.py` | `PaddleOcr.Inference` | Done | Routed to ONNX system chain |
| Infer | `infer sr` | `tools/infer_sr.py` | `PaddleOcr.Inference` | Done | ONNX SR pipeline |
| Infer | `infer table` | `tools/infer_table.py` | `PaddleOcr.Inference` | Partial | ONNX runner + OCR fusion + tensor metadata output |
| Infer | `infer kie` | `tools/infer_kie.py` | `PaddleOcr.Inference` | Partial | ONNX runner + OCR fusion + tensor metadata output |
| Infer | `infer kie-ser` | `tools/infer_kie_token_ser.py` | `PaddleOcr.Inference` | Partial | ONNX runner + OCR fusion + tensor metadata output |
| Infer | `infer kie-re` | `tools/infer_kie_token_ser_re.py` | `PaddleOcr.Inference` | Partial | ONNX runner + OCR fusion + tensor metadata output |
| Service | `service test` | `tools/test_hubserving.py` | `PaddleOcr.ServiceClient` | Done | HTTP batch + optional visualize |
| E2E tools | `e2e convert-label` | `tools/end2end/convert_ppocr_label.py` | `PaddleOcr.Data` | Done | Label conversion |
| E2E tools | `e2e eval` | `tools/end2end/eval_end2end.py` | `PaddleOcr.Data` | Done | Polygon IoU + edit distance metrics |
