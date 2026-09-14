#!/usr/bin/env python3
"""Valida o modelo quantizado e compara decisões e probabilidades com float."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from common import ROOT, relative, sha256, softmax, wilson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts/quantized/iris_mlp_logits_quantized.h5",
    )
    parser.add_argument("--test-data", type=Path, default=ROOT / "data/prepared/iris_test.npz")
    parser.add_argument(
        "--float-outputs", type=Path, default=ROOT / "results/float_test_outputs.npz"
    )
    parser.add_argument(
        "--outputs", type=Path, default=ROOT / "results/quantized_test_outputs.npz"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/quantized_validation.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.model, args.test_data, args.float_outputs):
        if not required.is_file():
            raise FileNotFoundError(required)

    data = np.load(args.test_data)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    with vitis_quantize.quantize_scope():
        model = tf.keras.models.load_model(args.model, compile=False)
    logits = model.predict(features, batch_size=len(features), verbose=0).astype(np.float32)
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=1).astype(np.int64)

    float_data = np.load(args.float_outputs)
    float_logits = float_data["logits"].astype(np.float32)
    float_probabilities = float_data["probabilities"].astype(np.float32)
    float_predictions = float_data["predictions"].astype(np.int64)
    if logits.shape != float_logits.shape:
        raise RuntimeError("Shapes float e quantizado incompatíveis")

    correct = int(np.sum(predictions == labels))
    agreement = int(np.sum(predictions == float_predictions))
    logits_error = np.abs(logits - float_logits)
    probability_error = np.abs(probabilities - float_probabilities)
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outputs,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
    )
    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "model": relative(args.model),
        "model_sha256": sha256(args.model),
        "test_samples": int(len(labels)),
        "correct": correct,
        "accuracy": correct / len(labels),
        "accuracy_percent": 100.0 * correct / len(labels),
        "wilson_95": wilson(correct, len(labels)),
        "float_argmax_agreement": agreement,
        "float_argmax_agreement_percent": 100.0 * agreement / len(labels),
        "max_abs_logit_error": float(logits_error.max()),
        "mean_abs_logit_error": float(logits_error.mean()),
        "max_abs_probability_error": float(probability_error.max()),
        "mean_abs_probability_error": float(probability_error.mean()),
        "outputs": relative(args.outputs),
        "outputs_sha256": sha256(args.outputs),
        "decision": "argmax of quantized logits",
        "softmax_policy": "computed in software only for probability comparison",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
