#!/usr/bin/env python3
"""Valida o modelo Keras 2 float nas 10.000 imagens de teste."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    radius /= denominator
    return [center - radius, center + radius]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT
        / "models/float/resnet8_cifar10_keras2_vai3_5_no_softmax.h5",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=ROOT / "data/prepared/cifar10_test_uint8.npz",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "results/float_test_outputs.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/float_validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.model, args.test_data):
        if not required.is_file():
            raise FileNotFoundError(required)

    data = np.load(args.test_data)
    images = data["images"].astype(np.float32) / 255.0
    labels = data["labels"].reshape(-1).astype(np.int64)
    model = tf.keras.models.load_model(args.model, compile=False)
    logits = model.predict(images, batch_size=args.batch_size, verbose=1).astype(
        np.float32
    )
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    correct = int(np.sum(predictions == labels))

    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outputs, logits=logits, predictions=predictions, labels=labels
    )
    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "test_data": str(args.test_data.relative_to(ROOT)),
        "test_data_sha256": sha256(args.test_data),
        "outputs": str(args.outputs.relative_to(ROOT)),
        "outputs_sha256": sha256(args.outputs),
        "samples": int(len(labels)),
        "correct": correct,
        "accuracy": correct / len(labels),
        "accuracy_percent": 100.0 * correct / len(labels),
        "wilson_95": wilson(correct, len(labels)),
        "batch_size": args.batch_size,
        "preprocessing": "uint8 to float32, divide by 255.0",
        "softmax": False,
        "decision": "argmax of logits",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
