#!/usr/bin/env python3
"""Executa uma inferencia LeNet batch 1 no DPU via VART."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import vart
import xir


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def find_dpu_subgraph(graph):
    dpu = [item for item in graph.get_root_subgraph().toposort_child_subgraph() if item.has_attr("device") and item.get_attr("device") == "DPU"]
    if len(dpu) != 1:
        raise RuntimeError(f"esperado 1 subgrafo DPU, encontrado {len(dpu)}")
    return dpu[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmodel", type=Path, default=Path("lenet_mnist_no_softmax.xmodel"))
    parser.add_argument("--dataset", type=Path, default=Path("mnist_test_uint8.npz"))
    parser.add_argument("--reference", type=Path, default=Path("quantized_test_outputs.npz"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/smoke_inference.json"))
    args = parser.parse_args()
    for required in (args.xmodel, args.dataset, args.reference):
        if not required.is_file():
            raise FileNotFoundError(required)
    dataset = np.load(args.dataset)
    images = dataset["images"]
    labels = dataset["labels"].reshape(-1).astype(np.int64)
    reference_predictions = np.load(args.reference)["predictions"].reshape(-1).astype(np.int64)
    graph = xir.Graph.deserialize(str(args.xmodel))
    runner = vart.Runner.create_runner(find_dpu_subgraph(graph), "run")
    input_tensor, output_tensor = runner.get_input_tensors()[0], runner.get_output_tensors()[0]
    input_shape, output_shape = tuple(input_tensor.dims), tuple(output_tensor.dims)
    input_fix, output_fix = int(input_tensor.get_attr("fix_point")), int(output_tensor.get_attr("fix_point"))
    image = images[args.sample_index].astype(np.float32) / 255.0
    input_buffer = np.ascontiguousarray(np.clip(np.rint(image * (2**input_fix)), -128, 127).astype(np.int8).reshape(input_shape))
    output_buffer = np.empty(output_shape, dtype=np.int8, order="C")

    def execute() -> None:
        job_id = runner.execute_async([input_buffer], [output_buffer])
        status = runner.wait(job_id)
        if status not in (0, None):
            raise RuntimeError(f"runner.wait retornou {status}")

    for _ in range(args.warmup):
        execute()
    t0 = perf_counter_ns(); execute(); t1 = perf_counter_ns()
    raw_output = output_buffer.reshape(-1).copy()
    logits = raw_output.astype(np.float32) * (2.0 ** (-output_fix))
    prediction = int(np.argmax(logits))
    expected = int(reference_predictions[args.sample_index])
    report = {
        "status": "passed" if prediction == expected else "failed", "xmodel": str(args.xmodel),
        "xmodel_sha256": sha256(args.xmodel), "sample_index": args.sample_index,
        "label": int(labels[args.sample_index]), "prediction_dpu": prediction,
        "prediction_quantized_host": expected, "prediction_matches_reference": prediction == expected,
        "raw_output_int8": raw_output.tolist(), "dequantized_logits": logits.tolist(),
        "input_shape": list(input_shape), "input_fix_point": input_fix,
        "output_shape": list(output_shape), "output_fix_point": output_fix,
        "warmup": args.warmup, "inference_only_ms": (t1 - t0) / 1e6,
        "latency_boundary": "execute_async + wait; input pre-quantized in runner buffer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("decisao DPU divergiu da referencia INT8 host")


if __name__ == "__main__":
    main()
