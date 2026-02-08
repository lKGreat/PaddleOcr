#!/usr/bin/env python3
"""Convert ONNX model to Paddle static inference model.

Outputs:
  <output_dir>/inference.pdmodel
  <output_dir>/inference.pdiparams
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import onnx
from onnx import helper
from x2paddle.convert import onnx2paddle


def _first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _rewrite_hardswish(onnx_path: str, output_path: str) -> int:
    model = onnx.load(onnx_path)
    replaced = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type != "HardSwish":
            new_nodes.append(node)
            continue

        replaced += 1
        input_name = node.input[0]
        output_name = node.output[0]
        hs_name = f"{output_name}_hardsigmoid"
        prefix = node.name or f"HardSwish_{replaced}"
        hard_sigmoid = helper.make_node(
            "HardSigmoid",
            [input_name],
            [hs_name],
            name=f"{prefix}_hsig",
            alpha=1.0 / 6.0,
            beta=0.5,
        )
        mul = helper.make_node(
            "Mul",
            [input_name, hs_name],
            [output_name],
            name=f"{prefix}_mul",
        )
        new_nodes.extend([hard_sigmoid, mul])

    if replaced == 0:
        return 0

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX to Paddle static converter")
    parser.add_argument("--onnx_model", required=True, help="Path to ONNX model")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--enable_optim", action="store_true", help="Enable x2paddle graph optimization")
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx_model)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"onnx model not found: {onnx_path}")

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pocr_x2paddle_") as temp_dir:
        rewritten_path = os.path.join(temp_dir, "rewritten.onnx")
        replaced = _rewrite_hardswish(onnx_path, rewritten_path)
        onnx_input = rewritten_path if replaced > 0 else onnx_path
        if replaced > 0:
            print(f"Rewrote HardSwish nodes: {replaced}")

        onnx2paddle(
            model_path=onnx_input,
            save_dir=temp_dir,
            enable_optim=bool(args.enable_optim),
            disable_feedback=True,
            enable_onnx_checker=True,
        )

        graph_src = _first_existing(
            [
                os.path.join(temp_dir, "inference_model", "model.pdmodel"),
                os.path.join(temp_dir, "inference_model", "model.json"),
                os.path.join(temp_dir, "model.pdmodel"),
                os.path.join(temp_dir, "model.json"),
            ]
        )
        params_src = _first_existing(
            [
                os.path.join(temp_dir, "inference_model", "model.pdiparams"),
                os.path.join(temp_dir, "inference_model", "model.pdparams"),
                os.path.join(temp_dir, "model.pdiparams"),
                os.path.join(temp_dir, "model.pdparams"),
            ]
        )

        if graph_src is None or params_src is None:
            raise RuntimeError(
                "x2paddle conversion succeeded but pdmodel/pdiparams artifacts were not found"
            )

        pdmodel_dst = os.path.join(output_dir, "inference.pdmodel")
        pdiparams_dst = os.path.join(output_dir, "inference.pdiparams")
        shutil.copyfile(graph_src, pdmodel_dst)
        shutil.copyfile(params_src, pdiparams_dst)
        if graph_src.endswith(".json"):
            shutil.copyfile(graph_src, os.path.join(output_dir, "inference.json"))

        print(pdmodel_dst)
        print(pdiparams_dst)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
