#!/usr/bin/env python3
"""Executa PTQ INT8 pof2s da LeNet com o Vitis AI Quantizer 3.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/float/lenet_mnist_keras2_vai3_5_no_softmax.h5")
    parser.add_argument("--calibration-data", type=Path, default=ROOT / "data/prepared/mnist_calibration_train_uint8.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/quantized/lenet_mnist_no_softmax_quantized.h5")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/quantization.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--strategy", default="pof2s")
    parser.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
    args = parser.parse_args()

    for required in (args.model, args.calibration_data):
        if not required.is_file():
            raise FileNotFoundError(required)
    calibration = np.load(args.calibration_data)
    images = (calibration["images"].astype(np.float32) / 255.0)[..., np.newaxis]
    labels = calibration["labels"].reshape(-1).astype(np.int64)
    float_model = tf.keras.models.load_model(args.model, compile=False)
    quantizer = vitis_quantize.VitisQuantizer(
        float_model,
        quantize_strategy=args.strategy,
        target=args.target,
    )
    quantized_model = quantizer.quantize_model(
        calib_dataset=images,
        calib_batch_size=args.batch_size,
        verbose=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized_model.save(args.output, include_optimizer=False)
    report = {
        "status": "passed",
        "tool": "Vitis AI Quantizer for TensorFlow 2",
        "tensorflow_version": tf.__version__,
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "calibration_data": str(args.calibration_data.relative_to(ROOT)),
        "calibration_data_sha256": sha256(args.calibration_data),
        "calibration_split": "MNIST official training split",
        "calibration_samples": int(len(images)),
        "calibration_class_counts": np.bincount(labels, minlength=10).tolist(),
        "calibration_batch_size": args.batch_size,
        "preprocessing": "uint8 -> float32; divide by 255.0; add channels_last dimension",
        "strategy": args.strategy,
        "numeric_format": "INT8",
        "target": args.target,
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "input_shape": list(quantized_model.input_shape),
        "output_shape": list(quantized_model.output_shape),
        "softmax": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
