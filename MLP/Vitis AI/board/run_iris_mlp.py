#!/usr/bin/env python3
"""Executa uma inferência da MLP Iris na DPU da ZCU104 via VART."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import vart
import xir


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xmodel", type=Path, default=HERE / "iris_mlp.xmodel"
    )
    parser.add_argument(
        "--preprocessing", type=Path, default=HERE / "preprocessing.json"
    )
    parser.add_argument(
        "--features",
        nargs=4,
        type=float,
        metavar=("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"),
        required=True,
        help="Quatro medidas Iris brutas em centímetros.",
    )
    return parser.parse_args()


def only_dpu_subgraph(graph: xir.Graph):
    children = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [
        subgraph
        for subgraph in children
        if subgraph.has_attr("device") and subgraph.get_attr("device") == "DPU"
    ]
    if len(dpu) != 1:
        raise RuntimeError(f"Esperado exatamente um subgrafo DPU; encontrados {len(dpu)}")
    return dpu[0]


def fixed_point(tensor) -> int:
    if not tensor.has_attr("fix_point"):
        raise RuntimeError(f"Tensor {tensor.name} não possui atributo fix_point")
    return int(tensor.get_attr("fix_point"))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def main() -> None:
    args = parse_args()
    preprocessing = json.loads(args.preprocessing.read_text(encoding="utf-8"))
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["scale"], dtype=np.float32)
    raw = np.asarray(args.features, dtype=np.float32)
    normalized = (raw - mean) / scale

    graph = xir.Graph.deserialize(str(args.xmodel))
    runner = vart.Runner.create_runner(only_dpu_subgraph(graph), "run")
    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]
    input_fix = fixed_point(input_tensor)
    output_fix = fixed_point(output_tensor)

    input_buffer = np.clip(
        np.rint(normalized.reshape(input_tensor.dims) * (2**input_fix)),
        -128,
        127,
    ).astype(np.int8)
    output_buffer = np.empty(output_tensor.dims, dtype=np.int8)
    job_id = runner.execute_async([input_buffer], [output_buffer])
    runner.wait(job_id)

    logits = output_buffer.astype(np.float32).reshape(-1) / (2**output_fix)
    probabilities = softmax(logits)
    predicted_index = int(np.argmax(logits))
    result = {
        "raw_features_cm": raw.tolist(),
        "normalized_features": normalized.tolist(),
        "input_int8": input_buffer.reshape(-1).tolist(),
        "input_fix_point": input_fix,
        "output_int8": output_buffer.reshape(-1).tolist(),
        "output_fix_point": output_fix,
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "predicted_index": predicted_index,
        "predicted_class": preprocessing["classes"][predicted_index],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
