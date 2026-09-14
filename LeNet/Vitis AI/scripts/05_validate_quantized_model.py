#!/usr/bin/env python3
"""Valida o H5 INT8 e compara suas decisões e logits com o modelo float."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from common import sha256, wilson


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/quantized/lenet_mnist_no_softmax_quantized.h5")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data/prepared/mnist_test_uint8.npz")
    parser.add_argument("--float-outputs", type=Path, default=ROOT / "results/float_test_outputs.npz")
    parser.add_argument("--outputs", type=Path, default=ROOT / "results/quantized_test_outputs.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/quantized_validation.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-accuracy-drop-pp", type=float, default=1.0)
    args = parser.parse_args()

    for required in (args.model, args.test_data, args.float_outputs):
        if not required.is_file():
            raise FileNotFoundError(required)
    data = np.load(args.test_data)
    images = (data["images"].astype(np.float32) / 255.0)[..., np.newaxis]
    labels = data["labels"].reshape(-1).astype(np.int64)
    with vitis_quantize.quantize_scope():
        model = tf.keras.models.load_model(args.model, compile=False)
    logits = model.predict(images, batch_size=args.batch_size, verbose=1).astype(np.float32)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    float_data = np.load(args.float_outputs)
    float_logits = float_data["logits"].astype(np.float32)
    float_predictions = float_data["predictions"].reshape(-1).astype(np.int64)
    if logits.shape != float_logits.shape:
        raise RuntimeError(f"formas divergentes: INT8 {logits.shape}, float {float_logits.shape}")

    correct = int(np.sum(predictions == labels))
    float_correct = int(np.sum(float_predictions == labels))
    agreement = int(np.sum(predictions == float_predictions))
    accuracy = correct / len(labels)
    float_accuracy = float_correct / len(labels)
    drop_pp = 100.0 * (float_accuracy - accuracy)
    absolute_error = np.abs(logits - float_logits)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class = []
    for class_id in range(10):
        selected = labels == class_id
        total = int(selected.sum())
        class_correct = int(np.sum(predictions[selected] == class_id))
        per_class.append({
            "class": class_id,
            "samples": total,
            "correct": class_correct,
            "accuracy_percent": 100.0 * class_correct / total,
        })
    status = "passed" if drop_pp <= args.max_accuracy_drop_pp else "failed"

    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.outputs, logits=logits, predictions=predictions, labels=labels)
    report = {
        "status": status,
        "tensorflow_version": tf.__version__,
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "samples": int(len(labels)),
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "wilson_95": wilson(correct, len(labels)),
        "float_correct": float_correct,
        "float_accuracy_percent": 100.0 * float_accuracy,
        "accuracy_delta_quantized_minus_float_pp": 100.0 * (accuracy - float_accuracy),
        "accuracy_drop_pp": drop_pp,
        "maximum_accepted_accuracy_drop_pp": args.max_accuracy_drop_pp,
        "float_argmax_agreement": agreement,
        "float_argmax_agreement_percent": 100.0 * agreement / len(labels),
        "logit_mean_absolute_error": float(absolute_error.mean()),
        "logit_root_mean_squared_error": float(np.sqrt(np.mean((logits - float_logits) ** 2))),
        "logit_max_absolute_error": float(absolute_error.max()),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "outputs": str(args.outputs.relative_to(ROOT)),
        "outputs_sha256": sha256(args.outputs),
        "batch_size": args.batch_size,
        "preprocessing": "uint8 -> float32; divide by 255.0; add channels_last dimension",
        "softmax": False,
        "decision": "argmax of quantized logits",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "passed":
        raise RuntimeError(f"queda de acuracia {drop_pp:.4f} pp excedeu {args.max_accuracy_drop_pp:.4f} pp")


if __name__ == "__main__":
    main()
