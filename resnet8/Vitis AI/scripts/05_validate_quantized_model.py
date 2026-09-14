#!/usr/bin/env python3
"""Valida o H5 quantizado e compara suas decisões com o modelo float."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize


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
        / "artifacts/quantized/resnet8_cifar10_no_softmax_quantized.h5",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=ROOT / "data/prepared/cifar10_test_uint8.npz",
    )
    parser.add_argument(
        "--float-outputs",
        type=Path,
        default=ROOT / "results/float_test_outputs.npz",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=ROOT / "results/quantized_test_outputs.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/quantized_validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.model, args.test_data, args.float_outputs):
        if not required.is_file():
            raise FileNotFoundError(required)

    data = np.load(args.test_data)
    images = data["images"].astype(np.float32) / 255.0
    labels = data["labels"].reshape(-1).astype(np.int64)
    with vitis_quantize.quantize_scope():
        model = tf.keras.models.load_model(args.model, compile=False)
    logits = model.predict(images, batch_size=args.batch_size, verbose=1).astype(
        np.float32
    )
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    float_data = np.load(args.float_outputs)
    float_predictions = float_data["predictions"].reshape(-1).astype(np.int64)
    if len(float_predictions) != len(predictions):
        raise RuntimeError("Saídas float e quantizadas têm comprimentos diferentes")

    correct = int(np.sum(predictions == labels))
    agreement = int(np.sum(predictions == float_predictions))
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outputs, logits=logits, predictions=predictions, labels=labels
    )
    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "samples": int(len(labels)),
        "correct": correct,
        "accuracy": correct / len(labels),
        "accuracy_percent": 100.0 * correct / len(labels),
        "wilson_95": wilson(correct, len(labels)),
        "float_argmax_agreement": agreement,
        "float_argmax_agreement_percent": 100.0 * agreement / len(labels),
        "outputs": str(args.outputs.relative_to(ROOT)),
        "outputs_sha256": sha256(args.outputs),
        "batch_size": args.batch_size,
        "preprocessing": "uint8 to float32, divide by 255.0",
        "softmax": False,
        "decision": "argmax of quantized logits",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
