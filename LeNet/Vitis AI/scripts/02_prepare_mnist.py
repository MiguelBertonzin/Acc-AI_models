#!/usr/bin/env python3
"""Prepara 10.000 imagens balanceadas de treino e todo o teste MNIST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/cache/keras/datasets/mnist.npz")
    parser.add_argument("--calibration-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--calibration-output", type=Path, default=ROOT / "data/prepared/mnist_calibration_train_uint8.npz")
    parser.add_argument("--test-output", type=Path, default=ROOT / "data/prepared/mnist_test_uint8.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/mnist_preparation.json")
    args = parser.parse_args()

    if args.calibration_samples < 10 or args.calibration_samples % 10:
        raise ValueError("calibration-samples deve ser multiplo de 10")
    raw = np.load(args.source)
    x_train, y_train = raw["x_train"], raw["y_train"].astype(np.int64)
    x_test, y_test = raw["x_test"], raw["y_test"].astype(np.int64)
    per_class = args.calibration_samples // 10
    rng = np.random.default_rng(args.seed)
    selected = []
    for class_id in range(10):
        candidates = np.flatnonzero(y_train == class_id)
        if per_class > len(candidates):
            raise ValueError(f"classe {class_id} possui somente {len(candidates)} imagens")
        selected.append(rng.choice(candidates, size=per_class, replace=False))
    calibration_indices = np.concatenate(selected).astype(np.int64)
    rng.shuffle(calibration_indices)

    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.calibration_output,
        images=x_train[calibration_indices],
        labels=y_train[calibration_indices],
        train_indices=calibration_indices,
    )
    np.savez_compressed(
        args.test_output,
        images=x_test,
        labels=y_test,
        test_indices=np.arange(len(x_test), dtype=np.int64),
    )
    report = {
        "status": "passed",
        "dataset": "MNIST",
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": sha256(args.source),
        "calibration_split": "official training split",
        "calibration_selection": "stratified without replacement",
        "calibration_samples": int(len(calibration_indices)),
        "calibration_class_counts": np.bincount(y_train[calibration_indices], minlength=10).tolist(),
        "calibration_output": str(args.calibration_output.relative_to(ROOT)),
        "calibration_sha256": sha256(args.calibration_output),
        "test_split": "official test split",
        "test_samples": int(len(x_test)),
        "test_class_counts": np.bincount(y_test, minlength=10).tolist(),
        "test_output": str(args.test_output.relative_to(ROOT)),
        "test_sha256": sha256(args.test_output),
        "raw_dtype": str(x_train.dtype),
        "model_input_shape": [28, 28, 1],
        "runtime_preprocessing": "astype(float32) / 255.0; add channels_last dimension",
        "seed": args.seed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
