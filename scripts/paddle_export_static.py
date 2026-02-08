#!/usr/bin/env python3
"""Prepare strict Paddle static inference bundle from an existing Paddle source.

Strict mode requirements:
  - source must be Paddle static graph artifacts (json/pdmodel + pdiparams)
  - output must contain inference.pdiparams and inference.json (preferred PP-OCRv5 format)
  - predictor load check must pass
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Optional, Tuple

import paddle


def _pick_graph_and_params(source: str) -> Tuple[str, str]:
    source = os.path.abspath(source)
    if os.path.isdir(source):
        graph_candidates = [
            os.path.join(source, "inference.json"),
            os.path.join(source, "inference.pdmodel"),
            os.path.join(source, "model.json"),
            os.path.join(source, "model.pdmodel"),
        ]
        params_candidates = [
            os.path.join(source, "inference.pdiparams"),
            os.path.join(source, "model.pdiparams"),
            os.path.join(source, "model.pdparams"),
        ]
        graph = _first_existing(graph_candidates)
        params = _first_existing(params_candidates)
        if graph and params:
            return graph, params
        raise FileNotFoundError(f"missing graph/params in directory: {source}")

    if not os.path.isfile(source):
        raise FileNotFoundError(f"source does not exist: {source}")

    ext = os.path.splitext(source)[1].lower()
    if ext not in (".json", ".pdmodel"):
        raise ValueError(f"unsupported strict source file: {source}")

    params_candidates = [
        os.path.splitext(source)[0] + ".pdiparams",
        os.path.join(os.path.dirname(source), "inference.pdiparams"),
        os.path.join(os.path.dirname(source), "model.pdiparams"),
    ]
    params = _first_existing(params_candidates)
    if not params:
        raise FileNotFoundError(f"cannot find pdiparams for source: {source}")
    return source, params


def _first_existing(paths: list[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _verify_load(graph_path: str, params_path: str) -> None:
    cfg = paddle.inference.Config(graph_path, params_path)
    predictor = paddle.inference.create_predictor(cfg)
    # Force materialization of model io to validate full load.
    _ = predictor.get_input_names()
    _ = predictor.get_output_names()


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict Paddle static export bridge")
    parser.add_argument("--source", required=True, help="Paddle static source dir/file")
    parser.add_argument("--output_dir", required=True, help="output directory")
    args = parser.parse_args()

    graph_src, params_src = _pick_graph_and_params(args.source)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    graph_ext = os.path.splitext(graph_src)[1].lower()
    if graph_ext == ".json":
        graph_dst = os.path.join(output_dir, "inference.json")
    else:
        # Keep pdmodel for compatibility, but strict consumer should prefer json format.
        graph_dst = os.path.join(output_dir, "inference.pdmodel")
    params_dst = os.path.join(output_dir, "inference.pdiparams")

    shutil.copyfile(graph_src, graph_dst)
    shutil.copyfile(params_src, params_dst)

    # Keep source inference.yml when available to preserve Paddle bundle metadata.
    source_dir = args.source if os.path.isdir(args.source) else os.path.dirname(args.source)
    source_yml = os.path.join(source_dir, "inference.yml")
    if os.path.exists(source_yml):
        shutil.copyfile(source_yml, os.path.join(output_dir, "inference.yml"))

    # Validate destination can be loaded by Paddle predictor.
    _verify_load(graph_dst, params_dst)

    print(graph_dst)
    print(params_dst)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
