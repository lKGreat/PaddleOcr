---
name: PaddleOCR C# Reimplementation
overview: One-to-one reimplementation of PaddleOCR PP-OCRv5 mobile (det + rec + cls) inference pipeline in C# .NET 10, covering all model architectures, pre/post processing, and the full OCR pipeline.
todos:
  - id: layers
    content: "Implement basic layers: ConvBNLayer, LearnableAffineBlock, LearnableRepLayer (with rep fusion), LCNetV3Block, SELayer, SEModule, RSELayer, ResidualUnit, SVTRBlock, Attention, Mlp"
    status: in_progress
  - id: backbone-pplcnetv3
    content: Implement PPLCNetV3 backbone with both det (scale=0.75, NET_CONFIG_det, 4 output maps) and rec (scale=0.95, NET_CONFIG_rec, single output with pooling) modes
    status: pending
  - id: backbone-mobilenetv3
    content: Implement MobileNetV3 (small, scale=0.35) backbone for classification with 11 ResidualUnit blocks
    status: pending
  - id: neck-rsefpn
    content: "Implement RSEFPN neck: 4x RSELayer(1x1) input, top-down fusion, 4x RSELayer(3x3) output, upsample+concat to 96ch"
    status: pending
  - id: neck-svtr
    content: "Implement EncoderWithSVTR neck: conv reduction (480->60->120), 2x SVTR blocks (global attention, 8 heads), guide concat, conv expansion (960->60->120)"
    status: pending
  - id: head-db
    content: "Implement DBHead: binarize+thresh dual Head (Conv3x3->TransConv2x2->TransConv2x2->Sigmoid), step function with k=50"
    status: pending
  - id: head-ctc
    content: "Implement CTCHead: single Linear(120, vocab_size) with softmax in inference"
    status: pending
  - id: head-multihead
    content: "Implement MultiHead: wraps CTCHead (with SVTR neck) + NRTRHead (training only), inference returns CTC output only"
    status: pending
  - id: head-cls
    content: "Implement ClsHead: AdaptiveAvgPool(1) + Linear(200, 2) + softmax"
    status: pending
  - id: preprocess
    content: "Implement preprocessing: Det (resize limit 960, ImageNet normalize), Rec (resize h=48, w<=320, pad, [-1,1] normalize), Cls (resize h=48, w<=192, pad, [-1,1] normalize)"
    status: pending
  - id: postprocess
    content: "Implement postprocessing: DBPostProcess (threshold, contours, unclip with Clipper2, scale), CTCLabelDecode (argmax, dedup, blank removal, char mapping), ClsPostProcess (argmax, label mapping)"
    status: pending
  - id: pipeline
    content: "Implement TextSystem pipeline: det -> crop+sort -> cls (optional rotation correction) -> rec -> filter by confidence"
    status: pending
  - id: model-loading
    content: "Implement model weight loading: support ONNX Runtime inference or custom weight loading from exported PaddlePaddle models"
    status: pending
isProject: false
---

