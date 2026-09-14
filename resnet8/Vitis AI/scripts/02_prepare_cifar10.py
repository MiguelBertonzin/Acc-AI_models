#!/usr/bin/env python3
"""Prepara calibração estratificada e teste CIFAR-10 dentro do projeto."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/cache/keras/datasets/cifar-10-batches-py",
    )
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=ROOT / "data/prepared/cifar10_calibration_train_uint8.npz",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=ROOT / "data/prepared/cifar10_test_uint8.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/cifar10_preparation.json",
    )
    return parser.parse_args()


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def main() -> None:
    args = parse_args()
    if args.calibration_samples < 10 or args.calibration_samples % 10:
        raise ValueError("--calibration-samples deve ser múltiplo de 10")

    train_paths = [args.source_dir / f"data_batch_{index}" for index in range(1, 6)]
    test_path = args.source_dir / "test_batch"
    for required in [*train_paths, test_path]:
        if not required.is_file():
            raise FileNotFoundError(required)
    train_batches = [load_batch(path) for path in train_paths]
    x_train = np.concatenate([item[0] for item in train_batches], axis=0)
    y_train = np.concatenate([item[1] for item in train_batches], axis=0)
    x_test, y_test = load_batch(test_path)

    per_class = args.calibration_samples // 10
    rng = np.random.default_rng(args.seed)
    selected = []
    for class_id in range(10):
        candidates = np.flatnonzero(y_train == class_id)
        selected.append(rng.choice(candidates, size=per_class, replace=False))
    calibration_indices = np.concatenate(selected).astype(np.int64)
    rng.shuffle(calibration_indices)

    x_calibration = x_train[calibration_indices]
    y_calibration = y_train[calibration_indices]
    test_indices = np.arange(len(x_test), dtype=np.int64)

    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.calibration_output,
        images=x_calibration,
        labels=y_calibration,
        train_indices=calibration_indices,
    )
    np.savez_compressed(
        args.test_output,
        images=x_test,
        labels=y_test,
        test_indices=test_indices,
    )

    report = {
        "dataset": "CIFAR-10",
        "source_dir": str(args.source_dir.relative_to(ROOT)),
        "source_batch_sha256": {
            path.name: sha256(path) for path in [*train_paths, test_path]
        },
        "calibration_split": "train",
        "calibration_selection": "stratified without replacement",
        "calibration_samples": int(len(x_calibration)),
        "calibration_class_counts": np.bincount(y_calibration, minlength=10).tolist(),
        "calibration_output": str(args.calibration_output.relative_to(ROOT)),
        "calibration_sha256": sha256(args.calibration_output),
        "test_split": "test",
        "test_samples": int(len(x_test)),
        "test_class_counts": np.bincount(y_test, minlength=10).tolist(),
        "test_output": str(args.test_output.relative_to(ROOT)),
        "test_sha256": sha256(args.test_output),
        "raw_dtype": str(x_train.dtype),
        "shape": list(x_train.shape[1:]),
        "preprocessing_at_runtime": "astype(float32) / 255.0",
        "seed": args.seed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
