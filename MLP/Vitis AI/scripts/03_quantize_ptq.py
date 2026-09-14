#!/usr/bin/env python3
"""Executa PTQ INT8 pof2s com todas as 120 amostras de treino Iris."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from common import ROOT, relative, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/float/iris_mlp_keras2_vai3_5_logits.h5",
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=ROOT / "data/prepared/iris_calibration_train.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/quantized/iris_mlp_logits_quantized.h5",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/quantization.json"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strategy", default="pof2s")
    parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.model, args.calibration_data):
        if not required.is_file():
            raise FileNotFoundError(required)

    calibration = np.load(args.calibration_data)
    features = calibration["features"].astype(np.float32)
    labels = calibration["labels"].astype(np.int64)
    if len(features) != 120:
        raise RuntimeError(f"Esperadas 120 amostras de treino, recebidas {len(features)}")

    float_model = tf.keras.models.load_model(args.model, compile=False)
    quantizer = vitis_quantize.VitisQuantizer(
        float_model,
        quantize_strategy=args.strategy,
        target=args.target,
    )
    quantized_model = quantizer.quantize_model(
        calib_dataset=features,
        calib_batch_size=args.batch_size,
        verbose=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized_model.save(args.output, include_optimizer=False)

    report = {
        "status": "passed",
        "tensorflow_version": tf.__version__,
        "model": relative(args.model),
        "model_sha256": sha256(args.model),
        "calibration_data": relative(args.calibration_data),
        "calibration_data_sha256": sha256(args.calibration_data),
        "calibration_samples": int(len(features)),
        "calibration_unique_samples": int(len(np.unique(features, axis=0))),
        "calibration_class_counts": {
            str(index): int(np.sum(labels == index)) for index in np.unique(labels)
        },
        "calibration_batch_size": args.batch_size,
        "calibration_batches": int(np.ceil(len(features) / args.batch_size)),
        "test_samples_used": 0,
        "strategy": args.strategy,
        "numeric_format": "INT8 power-of-two scale",
        "target": args.target,
        "output": relative(args.output),
        "output_sha256": sha256(args.output),
        "input_shape": list(quantized_model.input_shape),
        "output_shape": list(quantized_model.output_shape),
        "softmax": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
