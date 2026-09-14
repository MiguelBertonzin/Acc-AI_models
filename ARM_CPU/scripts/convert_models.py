#!/usr/bin/env python3
"""Converte os três modelos canônicos para TFLite FP32 e INT8 integral."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from model_specs import ROOT, SPECS, prepare_sample


OUT = ROOT / "models/tflite"
MANIFEST = ROOT / "manifests/conversion_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representative_dataset(spec):
    data = np.load(spec["calibration"])
    samples = data[spec["input_key"]]
    for sample in samples:
        yield [prepare_sample(spec, sample)]


def tensor_details(model_content):
    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=1)
    interpreter.allocate_tensors()

    def simplify(item):
        scale, zero = item["quantization"]
        return {
            "name": item["name"],
            "shape": [int(v) for v in item["shape"]],
            "dtype": np.dtype(item["dtype"]).name,
            "quantization_scale": float(scale),
            "quantization_zero_point": int(zero),
        }

    return {
        "input": simplify(interpreter.get_input_details()[0]),
        "output": simplify(interpreter.get_output_details()[0]),
    }


def convert_one(key, spec):
    model = tf.keras.models.load_model(spec["model"], compile=False)

    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_content = fp32_converter.convert()
    fp32_path = OUT / f"{key}_fp32.tflite"
    fp32_path.write_bytes(fp32_content)

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_dataset(spec)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_content = int8_converter.convert()
    int8_path = OUT / f"{key}_int8.tflite"
    int8_path.write_bytes(int8_content)

    return {
        "network": spec["name"],
        "source_h5": str(spec["model"].relative_to(ROOT)),
        "source_sha256": sha256(spec["model"]),
        "calibration_dataset": str(spec["calibration"].relative_to(ROOT)),
        "calibration_sha256": sha256(spec["calibration"]),
        "keras_input_shape": [int(v) if v is not None else None for v in model.input_shape],
        "keras_output_shape": [int(v) if v is not None else None for v in model.output_shape],
        "parameters": int(model.count_params()),
        "fp32": {
            "file": str(fp32_path.relative_to(ROOT)),
            "bytes": fp32_path.stat().st_size,
            "sha256": sha256(fp32_path),
            **tensor_details(fp32_content),
        },
        "int8": {
            "file": str(int8_path.relative_to(ROOT)),
            "bytes": int8_path.stat().st_size,
            "sha256": sha256(int8_path),
            **tensor_details(int8_content),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["all", *SPECS], default="all")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    selected = SPECS if args.network == "all" else {args.network: SPECS[args.network]}
    results = {key: convert_one(key, spec) for key, spec in selected.items()}

    if "mlp" in selected:
        import joblib

        scaler = joblib.load(ROOT / "data/mlp/iris_scaler.joblib")
        np.savez_compressed(
            ROOT / "data/mlp/iris_scaler_params.npz",
            mean=np.asarray(scaler.mean_, dtype=np.float32),
            scale=np.asarray(scaler.scale_, dtype=np.float32),
        )

    previous = {}
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text(encoding="utf-8")).get("models", {})
    previous.update(results)

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "host_architecture": os.uname().machine,
        "target": "Linux AArch64 / ARM Cortex-A53 / ZCU104",
        "models": previous,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
