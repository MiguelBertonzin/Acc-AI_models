#!/usr/bin/env python3
"""Mede acuracia e latencia do XModel nas 10.000 imagens MNIST na ZCU104."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import vart
import xir


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
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/xmodel_accuracy_latency.json"))
    args = parser.parse_args()
    data = np.load(args.dataset)
    images = data["images"][:args.samples]
    labels = data["labels"].reshape(-1).astype(np.int64)[:args.samples]
    host_predictions = np.load(args.reference)["predictions"].reshape(-1).astype(np.int64)[:args.samples]
    graph = xir.Graph.deserialize(str(args.xmodel))
    runner = vart.Runner.create_runner(find_dpu_subgraph(graph), "run")
    input_tensor, output_tensor = runner.get_input_tensors()[0], runner.get_output_tensors()[0]
    input_shape, output_shape = tuple(input_tensor.dims), tuple(output_tensor.dims)
    if input_shape[0] != 1:
        raise RuntimeError(f"este validador requer batch 1, XModel usa {input_shape}")
    input_fix = int(input_tensor.get_attr("fix_point"))
    output_fix = int(output_tensor.get_attr("fix_point"))
    input_buffer = np.empty(input_shape, dtype=np.int8, order="C")
    output_buffer = np.empty(output_shape, dtype=np.int8, order="C")

    def execute() -> None:
        job = runner.execute_async([input_buffer], [output_buffer])
        status = runner.wait(job)
        if status not in (0, None):
            raise RuntimeError(f"runner.wait retornou {status}")

    input_buffer.fill(0)
    for _ in range(args.warmup):
        execute()
    predictions = np.empty(len(images), dtype=np.int64)
    latencies_ns = np.empty(len(images), dtype=np.int64)
    start_all = perf_counter_ns()
    for index, image in enumerate(images):
        quantized = np.clip(np.rint((image.astype(np.float32) / 255.0) * (2**input_fix)), -128, 127).astype(np.int8)
        input_buffer[...] = quantized.reshape(input_shape)
        t0 = perf_counter_ns(); execute(); t1 = perf_counter_ns()
        latencies_ns[index] = t1 - t0
        predictions[index] = int(np.argmax(output_buffer.reshape(-1).astype(np.float32) * (2.0 ** (-output_fix))))
    elapsed_s = (perf_counter_ns() - start_all) / 1e9
    correct = int(np.sum(predictions == labels))
    agreement = int(np.sum(predictions == host_predictions))
    latency_ms = latencies_ns.astype(np.float64) / 1e6
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    report = {
        "samples": int(len(labels)), "correct": correct, "accuracy_percent": 100.0 * correct / len(labels),
        "quantized_host_argmax_agreement": agreement,
        "quantized_host_argmax_agreement_percent": 100.0 * agreement / len(labels),
        "warmup": args.warmup, "input_shape": list(input_shape), "input_fix_point": input_fix,
        "output_shape": list(output_shape), "output_fix_point": output_fix,
        "latency_inference_only_ms": {
            "mean": float(latency_ms.mean()), "std_population": float(latency_ms.std()),
            "min": float(latency_ms.min()), "p50": float(np.percentile(latency_ms, 50)),
            "p90": float(np.percentile(latency_ms, 90)), "p95": float(np.percentile(latency_ms, 95)),
            "p99": float(np.percentile(latency_ms, 99)), "max": float(latency_ms.max()),
        },
        "end_to_end_loop_seconds": elapsed_s,
        "end_to_end_throughput_images_per_second": len(labels) / elapsed_s,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "latency_boundary": "execute_async + wait only; excludes image normalization/input quantization and argmax",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
