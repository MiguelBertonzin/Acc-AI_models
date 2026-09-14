#!/usr/bin/env python3
"""Reconstrói a ResNet8 em Keras 2.12, carrega os pesos e valida os logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resnet8() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32, 32, 3), dtype=tf.float32, name="input_layer")

    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", use_bias=False, name="conv2d"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    initial = tf.keras.layers.Activation("relu", name="activation")(x)

    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", use_bias=False, name="conv2d_1"
    )(initial)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = tf.keras.layers.Activation("relu", name="activation_1")(x)
    x = tf.keras.layers.Conv2D(
        16, 3, padding="same", use_bias=False, name="conv2d_2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_2")(x)
    x = tf.keras.layers.Add(name="add")([x, initial])
    stage16 = tf.keras.layers.Activation("relu", name="activation_2")(x)

    x = tf.keras.layers.Conv2D(
        32, 3, strides=2, padding="same", use_bias=False, name="conv2d_3"
    )(stage16)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_3")(x)
    x = tf.keras.layers.Activation("relu", name="activation_3")(x)
    main32 = tf.keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="conv2d_4"
    )(x)
    shortcut32 = tf.keras.layers.Conv2D(
        32, 1, strides=2, padding="valid", use_bias=False, name="conv2d_5"
    )(stage16)
    main32 = tf.keras.layers.BatchNormalization(name="batch_normalization_4")(main32)
    shortcut32 = tf.keras.layers.BatchNormalization(
        name="batch_normalization_5"
    )(shortcut32)
    x = tf.keras.layers.Add(name="add_1")([main32, shortcut32])
    stage32 = tf.keras.layers.Activation("relu", name="activation_4")(x)

    x = tf.keras.layers.Conv2D(
        64, 3, strides=2, padding="same", use_bias=False, name="conv2d_6"
    )(stage32)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_6")(x)
    x = tf.keras.layers.Activation("relu", name="activation_5")(x)
    main64 = tf.keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False, name="conv2d_7"
    )(x)
    shortcut64 = tf.keras.layers.Conv2D(
        64, 1, strides=2, padding="valid", use_bias=False, name="conv2d_8"
    )(stage32)
    main64 = tf.keras.layers.BatchNormalization(name="batch_normalization_7")(main64)
    shortcut64 = tf.keras.layers.BatchNormalization(
        name="batch_normalization_8"
    )(shortcut64)
    x = tf.keras.layers.Add(name="add_2")([main64, shortcut64])
    x = tf.keras.layers.Activation("relu", name="activation_6")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)
    outputs = tf.keras.layers.Dense(10, activation="linear", name="dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="functional")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "models/float/resnet8_cifar10_keras3_no_softmax.h5",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=ROOT / "data/prepared/keras3_conversion_probe.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "models/float/resnet8_cifar10_keras2_vai3_5_no_softmax.h5",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/keras2_conversion.json",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-5,
        help="Tolerância entre TensorFlow 2.21/Keras 3 e TensorFlow 2.12/Keras 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.source, args.probe):
        if not required.is_file():
            raise FileNotFoundError(required)

    model = build_resnet8()
    model.load_weights(args.source, by_name=True, skip_mismatch=False)

    probe = np.load(args.probe)
    expected = probe["outputs"].astype(np.float32)
    actual = model.predict(probe["inputs"], batch_size=len(expected), verbose=0).astype(
        np.float32
    )
    abs_error = np.abs(expected - actual)
    max_abs_error = float(abs_error.max())
    mean_abs_error = float(abs_error.mean())
    argmax_matches = int(
        np.sum(np.argmax(expected, axis=1) == np.argmax(actual, axis=1))
    )
    if max_abs_error > args.atol:
        raise RuntimeError(
            f"Conversão reprovada: max_abs_error={max_abs_error} > atol={args.atol}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, include_optimizer=False)

    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": sha256(args.source),
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": sha256(args.output),
        "probe": str(args.probe.relative_to(ROOT)),
        "probe_sha256": sha256(args.probe),
        "parameters": int(model.count_params()),
        "layers": len(model.layers),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "last_activation": model.layers[-1].activation.__name__,
        "samples": int(len(expected)),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "argmax_matches": argmax_matches,
        "argmax_total": int(len(expected)),
        "atol": args.atol,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
