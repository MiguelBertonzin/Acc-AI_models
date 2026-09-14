#!/usr/bin/env python3
"""Valida a LeNet Keras 2 float nas 10.000 imagens oficiais de teste."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import sha256, wilson


ROOT = Path(__file__).resolve().parents[1]


def class_statistics(labels: np.ndarray, predictions: np.ndarray) -> list[dict]:
    rows = []
    for class_id in range(10):
        selected = labels == class_id
        total = int(selected.sum())
        correct = int(np.sum(predictions[selected] == class_id))
        rows.append(
            {
                "class": class_id,
                "samples": total,
                "correct": correct,
                "accuracy": correct / total,
                "accuracy_percent": 100.0 * correct / total,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/float/lenet_mnist_keras2_vai3_5_no_softmax.h5")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data/prepared/mnist_test_uint8.npz")
    parser.add_argument("--outputs", type=Path, default=ROOT / "results/float_test_outputs.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/float_validation.json")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    for required in (args.model, args.test_data):
        if not required.is_file():
            raise FileNotFoundError(required)
    data = np.load(args.test_data)
    images = (data["images"].astype(np.float32) / 255.0)[..., np.newaxis]
    labels = data["labels"].reshape(-1).astype(np.int64)
    model = tf.keras.models.load_model(args.model, compile=False)
    logits = model.predict(images, batch_size=args.batch_size, verbose=1).astype(np.float32)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    correct = int(np.sum(predictions == labels))
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.outputs, logits=logits, predictions=predictions, labels=labels)
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
        "per_class": class_statistics(labels, predictions),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "batch_size": args.batch_size,
        "preprocessing": "uint8 -> float32; divide by 255.0; add channels_last dimension",
        "softmax": False,
        "decision": "argmax of logits",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
