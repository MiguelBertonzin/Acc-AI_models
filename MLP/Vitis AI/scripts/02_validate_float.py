#!/usr/bin/env python3
"""Valida o modelo float compatível com Vitis AI no holdout Iris."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import ROOT, relative, sha256, softmax, wilson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/float/iris_mlp_keras2_vai3_5_logits.h5",
    )
    parser.add_argument("--test-data", type=Path, default=ROOT / "data/prepared/iris_test.npz")
    parser.add_argument(
        "--outputs", type=Path, default=ROOT / "results/float_test_outputs.npz"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/float_validation.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.model, args.test_data):
        if not required.is_file():
            raise FileNotFoundError(required)

    data = np.load(args.test_data)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    model = tf.keras.models.load_model(args.model, compile=False)
    logits = model.predict(features, batch_size=len(features), verbose=0).astype(np.float32)
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    correct = int(np.sum(predictions == labels))

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
        "test_data": relative(args.test_data),
        "test_data_sha256": sha256(args.test_data),
        "outputs": relative(args.outputs),
        "outputs_sha256": sha256(args.outputs),
        "samples": int(len(labels)),
        "correct": correct,
        "accuracy": correct / len(labels),
        "accuracy_percent": 100.0 * correct / len(labels),
        "wilson_95": wilson(correct, len(labels)),
        "decision": "argmax of float logits",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
