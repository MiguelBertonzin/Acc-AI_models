#!/usr/bin/env python3
"""Valida acurácia e paridade de classes dos modelos TFLite no host."""

import argparse
import json
import math
import os
from datetime import datetime, timezone

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from model_specs import ROOT, SPECS, prepare_sample


def wilson(correct, total, z=1.959963984540054):
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def quantize(value, detail):
    scale, zero = detail["quantization"]
    if scale == 0:
        raise ValueError("Tensor INT8 sem escala de quantização")
    return np.clip(np.rint(value / scale + zero), -128, 127).astype(np.int8)


def dequantize(value, detail):
    scale, zero = detail["quantization"]
    return (value.astype(np.float32) - zero) * scale


def run_tflite(path, samples):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    outputs = []

    for value in samples:
        if input_detail["dtype"] == np.int8:
            value = quantize(value, input_detail)
        else:
            value = value.astype(input_detail["dtype"], copy=False)
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if output_detail["dtype"] == np.int8:
            output = dequantize(output, output_detail)
        outputs.append(np.asarray(output[0], dtype=np.float32))
    return np.stack(outputs)


def validate_one(key, spec):
    data = np.load(spec["test"])
    raw = data[spec["input_key"]]
    labels = data[spec["labels_key"]].astype(np.int64)
    samples = [prepare_sample(spec, item) for item in raw]

    keras_model = tf.keras.models.load_model(spec["model"], compile=False)
    keras_outputs = keras_model.predict(np.concatenate(samples), batch_size=128, verbose=0)
    keras_predictions = np.argmax(keras_outputs, axis=1)

    result = {
        "network": spec["name"],
        "unique_samples": int(len(labels)),
        "keras": {
            "correct": int(np.sum(keras_predictions == labels)),
            "accuracy": float(np.mean(keras_predictions == labels)),
        },
    }

    for precision in ("fp32", "int8"):
        path = ROOT / f"models/tflite/{key}_{precision}.tflite"
        outputs = run_tflite(path, samples)
        predictions = np.argmax(outputs, axis=1)
        correct = int(np.sum(predictions == labels))
        agreement = int(np.sum(predictions == keras_predictions))
        difference = np.abs(outputs - keras_outputs)
        result[precision] = {
            "model": str(path.relative_to(ROOT)),
            "correct": correct,
            "accuracy": correct / len(labels),
            "accuracy_wilson95": wilson(correct, len(labels)),
            "keras_class_agreement": agreement,
            "keras_class_agreement_rate": agreement / len(labels),
            "output_mae_vs_keras": float(np.mean(difference)),
            "output_max_abs_vs_keras": float(np.max(difference)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=["all", *SPECS], default="all")
    args = parser.parse_args()
    selected = SPECS if args.network == "all" else {args.network: SPECS[args.network]}
    results = {key: validate_one(key, spec) for key, spec in selected.items()}
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "models": results,
    }
    output = ROOT / "results/host_validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

