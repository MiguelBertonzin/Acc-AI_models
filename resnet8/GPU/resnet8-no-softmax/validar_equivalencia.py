#!/usr/bin/env python3
"""Validate the logits model against the original softmax model on CIFAR-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOFTMAX = PROJECT_ROOT / "resnet8_cifar10_keras3.h5"
DEFAULT_LOGITS = PROJECT_ROOT / "resnet8_cifar10_keras3_no_softmax.h5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "resultados" / "validacao_equivalencia.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax-model", type=Path, default=DEFAULT_SOFTMAX)
    parser.add_argument("--logits-model", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not args.softmax_model.is_file() or not args.logits_model.is_file():
        raise FileNotFoundError("Both model files are required.")

    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("No TensorFlow GPU is available; CPU fallback is forbidden.")
    tf.config.set_visible_devices(physical[0], "GPU")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_op_determinism()

    with tf.device("/GPU:0"):
        softmax_model = tf.keras.models.load_model(args.softmax_model, compile=False)
        logits_model = tf.keras.models.load_model(args.logits_model, compile=False)

    softmax_activation = tf.keras.activations.serialize(softmax_model.layers[-1].activation)
    logits_activation = tf.keras.activations.serialize(logits_model.layers[-1].activation)
    if softmax_activation != "softmax" or logits_activation != "linear":
        raise ValueError(
            f"Unexpected activations: softmax={softmax_activation!r}, logits={logits_activation!r}"
        )
    if softmax_model.input_shape != logits_model.input_shape:
        raise ValueError("Model input shapes differ.")
    if softmax_model.output_shape != logits_model.output_shape:
        raise ValueError("Model output shapes differ.")

    @tf.function(
        input_signature=[tf.TensorSpec((None, 32, 32, 3), tf.float32)],
        autograph=False,
        jit_compile=False,
    )
    def softmax_infer(batch: tf.Tensor) -> tf.Tensor:
        with tf.device("/GPU:0"):
            return softmax_model(batch, training=False)

    @tf.function(
        input_signature=[tf.TensorSpec((None, 32, 32, 3), tf.float32)],
        autograph=False,
        jit_compile=False,
    )
    def logits_infer(batch: tf.Tensor) -> tf.Tensor:
        with tf.device("/GPU:0"):
            return logits_model(batch, training=False)

    (_, _), (images, labels) = tf.keras.datasets.cifar10.load_data()
    images = images.astype(np.float32) / 255.0
    labels = labels.reshape(-1)
    probability_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    output_devices: set[str] = set()

    for start in range(0, len(images), args.batch_size):
        with tf.device("/GPU:0"):
            batch = tf.convert_to_tensor(images[start : start + args.batch_size])
            probabilities_tensor = softmax_infer(batch)
            logits_tensor = logits_infer(batch)
        output_devices.update((probabilities_tensor.device, logits_tensor.device))
        probability_parts.append(probabilities_tensor.numpy())
        logits_parts.append(logits_tensor.numpy())

    probabilities = np.concatenate(probability_parts)
    logits = np.concatenate(logits_parts)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    reconstructed = np.exp(shifted)
    reconstructed /= np.sum(reconstructed, axis=1, keepdims=True)
    softmax_classes = np.argmax(probabilities, axis=1)
    logits_classes = np.argmax(logits, axis=1)
    different = np.flatnonzero(softmax_classes != logits_classes)

    softmax_weights = softmax_model.get_weights()
    logits_weights = logits_model.get_weights()
    if len(softmax_weights) != len(logits_weights):
        raise RuntimeError("The models have different numbers of weight arrays.")
    maximum_weight_error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(softmax_weights, logits_weights, strict=True)
    )
    maximum_probability_error = float(np.max(np.abs(probabilities - reconstructed)))

    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    result = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CIFAR-10 official test split",
        "unique_images": int(len(images)),
        "input_normalization": "float32 / 255.0",
        "validation_batch_size": args.batch_size,
        "device": "GPU",
        "gpu_name": gpu_name,
        "output_devices": sorted(output_devices),
        "softmax_model": {
            "path": str(args.softmax_model.resolve()),
            "sha256": sha256(args.softmax_model),
            "size_bytes": args.softmax_model.stat().st_size,
            "activation": softmax_activation,
            "parameters": int(softmax_model.count_params()),
        },
        "logits_model": {
            "path": str(args.logits_model.resolve()),
            "sha256": sha256(args.logits_model),
            "size_bytes": args.logits_model.stat().st_size,
            "activation": logits_activation,
            "parameters": int(logits_model.count_params()),
        },
        "input_shape": list(softmax_model.input_shape),
        "output_shape": list(softmax_model.output_shape),
        "weight_array_count": len(softmax_weights),
        "maximum_weight_absolute_error": maximum_weight_error,
        "class_indices_equal": bool(different.size == 0),
        "equal_class_indices": int(len(images) - different.size),
        "different_class_indices": int(different.size),
        "different_indices": different.tolist(),
        "softmax_correct": int(np.count_nonzero(softmax_classes == labels)),
        "logits_correct": int(np.count_nonzero(logits_classes == labels)),
        "softmax_accuracy_percent": 100.0 * float(np.mean(softmax_classes == labels)),
        "logits_accuracy_percent": 100.0 * float(np.mean(logits_classes == labels)),
        "maximum_probability_reconstruction_error": maximum_probability_error,
        "mean_probability_reconstruction_error": float(
            np.mean(np.abs(probabilities - reconstructed))
        ),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "passed": bool(
            different.size == 0
            and maximum_weight_error == 0.0
            and maximum_probability_error <= 1e-6
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"Equivalence validation failed: {result}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
