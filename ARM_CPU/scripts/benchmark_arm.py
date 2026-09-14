#!/usr/bin/env python3
"""Benchmark CPU-only para TFLite/LiteRT na ARM Cortex-A53 da ZCU104."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    RUNTIME = "tflite_runtime"
except ImportError:
    import tensorflow.lite as tflite
    RUNTIME = "tensorflow.lite"

from model_specs import ROOT, SPECS, prepare_sample


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def describe(values):
    data = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    return {
        "count": int(len(data)), "mean": mean, "median": float(np.median(data)),
        "sample_sd": sd, "cv_percent": 100.0 * sd / mean if mean else None,
        "p90": percentile(data, 90), "p95": percentile(data, 95),
        "p99": percentile(data, 99), "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def quantize(value, detail):
    scale, zero = detail["quantization"]
    return np.clip(np.rint(value / scale + zero), -128, 127).astype(np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, choices=SPECS)
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--threads", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--inferences", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = SPECS[args.network]
    model_path = ROOT / f"models/tflite/{args.network}_{args.precision}.tflite"
    data = np.load(spec["test"])
    raw = data[spec["input_key"]]
    labels = data[spec["labels_key"]].astype(np.int64)

    interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=args.threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    prepared = []
    for sample in raw:
        value = prepare_sample(spec, sample)
        if input_detail["dtype"] == np.int8:
            value = quantize(value, input_detail)
        prepared.append(value.astype(input_detail["dtype"], copy=False))

    for index in range(args.warmup):
        interpreter.set_tensor(input_detail["index"], prepared[index % len(prepared)])
        interpreter.invoke()

    inference_latencies = []
    e2e_latencies = []
    predictions = np.empty(args.inferences, dtype=np.int64)
    global_start = time.perf_counter_ns()
    for index in range(args.inferences):
        sample_index = index % len(raw)

        e2e_start = time.perf_counter_ns()
        value = prepare_sample(spec, raw[sample_index])
        if input_detail["dtype"] == np.int8:
            value = quantize(value, input_detail)
        value = value.astype(input_detail["dtype"], copy=False)
        interpreter.set_tensor(input_detail["index"], value)

        inference_start = time.perf_counter_ns()
        interpreter.invoke()
        inference_end = time.perf_counter_ns()

        output = interpreter.get_tensor(output_detail["index"])
        predictions[index] = int(np.argmax(output[0]))
        e2e_end = time.perf_counter_ns()

        inference_latencies.append((inference_end - inference_start) / 1e6)
        e2e_latencies.append((e2e_end - e2e_start) / 1e6)
    global_end = time.perf_counter_ns()

    duration_s = (global_end - global_start) / 1e9
    unique_predictions = predictions[: min(len(labels), args.inferences)]
    unique_labels = labels[: len(unique_predictions)]
    correct = int(np.sum(unique_predictions == unique_labels))

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "network": spec["name"], "precision": args.precision,
        "runtime": RUNTIME, "platform": platform.platform(),
        "machine": platform.machine(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "threads": args.threads, "batch": 1, "warmup": args.warmup,
        "inferences": args.inferences, "duration_s": duration_s,
        "throughput_inf_s": args.inferences / duration_s,
        "inference_only_ms": describe(inference_latencies),
        "application_e2e_ms": describe(e2e_latencies),
        "accuracy_unique_prefix": {
            "samples": int(len(unique_labels)), "correct": correct,
            "accuracy": correct / len(unique_labels),
        },
        "model": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256(model_path),
        "test_dataset": str(spec["test"].relative_to(ROOT)),
        "test_dataset_sha256": sha256(spec["test"]),
        "input": {
            "shape": input_detail["shape"].tolist(),
            "dtype": np.dtype(input_detail["dtype"]).name,
            "quantization": list(input_detail["quantization"]),
        },
        "output": {
            "shape": output_detail["shape"].tolist(),
            "dtype": np.dtype(output_detail["dtype"]).name,
            "quantization": list(output_detail["quantization"]),
        },
        "outliers_removed": False,
        "notes": [
            "CPU-only: no VART runner, DPU delegate or FPGA overlay is loaded.",
            "Inference-only covers interpreter.invoke().",
            "E2E covers preprocessing, quantization if needed, tensor copy, invoke, output copy and argmax.",
            "Energy telemetry is intentionally separate and must be paired with this load on the board.",
        ],
    }

    output = args.output or ROOT / "results" / f"{args.network}_{args.precision}_t{args.threads}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

