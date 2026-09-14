#!/usr/bin/env python3
"""Reconstrói a MLP em Keras 2.12, preserva pesos e expõe logits para a DPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import ROOT, relative, sha256, softmax


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(4,), dtype=tf.float32, name="features")
    x = tf.keras.layers.Dense(8, activation="relu", name="dense1")(inputs)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense2")(x)
    outputs = tf.keras.layers.Dense(3, activation="linear", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="iris_mlp_logits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source/iris_mlp_clean.h5")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data/prepared/iris_test.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/float/iris_mlp_keras2_vai3_5_logits.h5",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/keras2_conversion.json"
    )
    parser.add_argument("--atol", type=float, default=2e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.source, args.test_data):
        if not required.is_file():
            raise FileNotFoundError(required)

    source = tf.keras.models.load_model(args.source, compile=False)
    model = build_model()
    for name in ("dense1", "dense2", "output"):
        model.get_layer(name).set_weights(source.get_layer(name).get_weights())

    data = np.load(args.test_data)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    source_probabilities = source.predict(features, batch_size=len(features), verbose=0)
    logits = model.predict(features, batch_size=len(features), verbose=0).astype(np.float32)
    rebuilt_probabilities = softmax(logits)
    error = np.abs(source_probabilities - rebuilt_probabilities)
    source_predictions = np.argmax(source_probabilities, axis=1)
    rebuilt_predictions = np.argmax(logits, axis=1)
    max_abs_error = float(error.max())
    if max_abs_error > args.atol or not np.array_equal(source_predictions, rebuilt_predictions):
        raise RuntimeError(
            f"Conversão inválida: max_abs_error={max_abs_error}, "
            f"agreement={np.mean(source_predictions == rebuilt_predictions)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, include_optimizer=False)
    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "source": relative(args.source),
        "source_sha256": sha256(args.source),
        "output": relative(args.output),
        "output_sha256": sha256(args.output),
        "parameters": int(model.count_params()),
        "architecture": "4-8-8-3",
        "source_last_activation": source.layers[-1].activation.__name__,
        "dpu_model_last_activation": model.layers[-1].activation.__name__,
        "softmax_policy": "removed from DPU graph; apply on ARM only when probabilities are required",
        "samples": int(len(labels)),
        "max_abs_probability_error": max_abs_error,
        "mean_abs_probability_error": float(error.mean()),
        "argmax_agreement": int(np.sum(source_predictions == rebuilt_predictions)),
        "argmax_total": int(len(labels)),
        "source_correct": int(np.sum(source_predictions == labels)),
        "rebuilt_correct": int(np.sum(rebuilt_predictions == labels)),
        "atol": args.atol,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
