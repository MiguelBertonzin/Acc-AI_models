#!/usr/bin/env python3
"""Reconstrói a LeNet em Keras 2.12, importa pesos e valida os logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def build_lenet() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32, name="input")
    x = tf.keras.layers.Conv2D(6, (5, 5), padding="valid", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)
    x = tf.keras.layers.Conv2D(16, (5, 5), padding="valid", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(120, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dense(84, activation="relu", name="dense2")(x)
    outputs = tf.keras.layers.Dense(10, activation="linear", name="output")(x)
    return tf.keras.Model(inputs, outputs, name="LeNet_MNIST")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/float/lenet_mnist_keras3_no_softmax.h5")
    parser.add_argument("--probe", type=Path, default=ROOT / "data/prepared/keras3_conversion_probe.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "models/float/lenet_mnist_keras2_vai3_5_no_softmax.h5")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/keras2_conversion.json")
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    for required in (args.source, args.probe):
        if not required.is_file():
            raise FileNotFoundError(required)
    model = build_lenet()
    model.load_weights(args.source, by_name=True, skip_mismatch=False)
    weight_checks = []
    with h5py.File(args.source, "r") as source_h5:
        for layer_name in ("conv1", "conv2", "dense1", "dense2", "output"):
            layer = model.get_layer(layer_name)
            loaded_weights = layer.get_weights()
            for tensor_name, loaded in zip(("kernel", "bias"), loaded_weights):
                source_tensor = np.asarray(
                    source_h5[f"model_weights/{layer_name}/{layer_name}/{tensor_name}"]
                )
                tensor_error = np.abs(source_tensor - loaded)
                weight_checks.append(
                    {
                        "tensor": f"{layer_name}/{tensor_name}",
                        "shape": list(loaded.shape),
                        "max_abs_error": float(tensor_error.max(initial=0.0)),
                        "exact_match": bool(np.array_equal(source_tensor, loaded)),
                    }
                )
    if not all(item["exact_match"] for item in weight_checks):
        raise RuntimeError("um ou mais tensores de pesos divergiram do H5 de origem")
    probe = np.load(args.probe)
    expected = probe["outputs"].astype(np.float32)
    actual = model.predict(probe["inputs"], batch_size=len(expected), verbose=0).astype(np.float32)
    error = np.abs(expected - actual)
    max_error = float(error.max())
    agreement = int(np.sum(np.argmax(expected, axis=1) == np.argmax(actual, axis=1)))
    if max_error > args.atol or agreement != len(expected):
        raise RuntimeError(f"conversao reprovada: erro={max_error}, argmax={agreement}/{len(expected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, include_optimizer=False)
    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": sha256(args.source),
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": sha256(args.output),
        "probe_sha256": sha256(args.probe),
        "samples": int(len(expected)),
        "max_abs_error": max_error,
        "mean_abs_error": float(error.mean()),
        "argmax_matches": agreement,
        "weights_exact_match": True,
        "weight_tensor_checks": weight_checks,
        "parameters": int(model.count_params()),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "last_activation": model.layers[-1].activation.__name__,
        "atol": args.atol,
        "numerical_note": "Logits compare TensorFlow 2.21/GPU with TensorFlow 2.12/CPU; weights are checked bit-exact separately.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
