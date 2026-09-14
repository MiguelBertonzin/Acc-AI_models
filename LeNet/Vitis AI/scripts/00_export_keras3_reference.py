#!/usr/bin/env python3
"""Exporta uma prova deterministica do H5 Keras 3 original."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/float/lenet_mnist_keras3_no_softmax.h5")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/cache/keras/datasets/mnist.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared/keras3_conversion_probe.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/keras3_reference.json")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    for required in (args.model, args.dataset):
        if not required.is_file():
            raise FileNotFoundError(required)
    raw = np.load(args.dataset)
    test_images = raw["x_test"]
    test_labels = raw["y_test"].astype(np.int64)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(test_images), size=args.samples, replace=False)
    inputs = (test_images[indices].astype(np.float32) / 255.0)[..., np.newaxis]

    model = tf.keras.models.load_model(args.model, compile=False)
    if model.input_shape != (None, 28, 28, 1) or model.output_shape != (None, 10):
        raise RuntimeError(f"formas inesperadas: {model.input_shape} -> {model.output_shape}")
    if model.layers[-1].activation.__name__ != "linear" or model.count_params() != 44426:
        raise RuntimeError("modelo de referencia nao e a LeNet logits esperada")
    outputs = model.predict(inputs, batch_size=args.samples, verbose=0).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, inputs=inputs, outputs=outputs, labels=test_labels[indices], test_indices=indices)
    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", None),
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "probe": str(args.output.relative_to(ROOT)),
        "probe_sha256": sha256(args.output),
        "samples": args.samples,
        "seed": args.seed,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameters": int(model.count_params()),
        "last_activation": model.layers[-1].activation.__name__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
