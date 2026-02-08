#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import paddle.inference as pdi
from PIL import Image


def _load_input(image_path: str | None, shape: tuple[int, int, int]) -> np.ndarray:
    c, h, w = shape
    if c != 3:
        raise ValueError(f"only 3-channel input is supported in verifier, got c={c}")

    if image_path is None:
        return np.zeros((1, c, h, w), dtype="float32")

    img = Image.open(image_path).convert("RGB").resize((w, h))
    arr = np.asarray(img, dtype="float32") / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Paddle rec inference.json/pdiparams by real forward.")
    parser.add_argument("--model_dir", required=True, help="Directory containing inference.json and inference.pdiparams")
    parser.add_argument("--image_path", default=None, help="Optional input image path")
    parser.add_argument("--rec_image_shape", default="3,48,320", help="Input shape C,H,W for dry-run input")
    parser.add_argument("--save_json", default=None, help="Optional path to save verification result json")
    args = parser.parse_args()

    model_file = os.path.join(args.model_dir, "inference.json")
    params_file = os.path.join(args.model_dir, "inference.pdiparams")
    if not os.path.isfile(model_file) or not os.path.isfile(params_file):
        raise FileNotFoundError(f"missing model artifacts in {args.model_dir}")

    c, h, w = [int(x.strip()) for x in args.rec_image_shape.split(",")]
    x = _load_input(args.image_path, (c, h, w))

    config = pdi.Config(model_file, params_file)
    config.disable_gpu()
    config.disable_glog_info()
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)
    predictor = pdi.create_predictor(config)

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    if not input_names:
        raise RuntimeError("predictor has no inputs")
    if not output_names:
        raise RuntimeError("predictor has no outputs")

    inp = predictor.get_input_handle(input_names[0])
    inp.reshape(x.shape)
    inp.copy_from_cpu(x)

    if "valid_ratio" in input_names:
        vr = predictor.get_input_handle("valid_ratio")
        vr_data = np.array([1.0], dtype="float32")
        vr.reshape(vr_data.shape)
        vr.copy_from_cpu(vr_data)

    predictor.run()
    out = predictor.get_output_handle(output_names[0]).copy_to_cpu()

    result = {
        "ok": True,
        "model_dir": args.model_dir,
        "inputs": list(input_names),
        "outputs": list(output_names),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_min": float(out.min()),
        "output_max": float(out.max()),
    }

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as ex:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(ex)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
