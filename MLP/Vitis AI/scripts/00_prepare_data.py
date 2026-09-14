#!/usr/bin/env python3
"""Prepara calibração e teste Iris sem vazamento entre os subconjuntos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from common import ROOT, relative, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaler", type=Path, default=ROOT / "source/iris_scaler.joblib")
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=ROOT / "data/prepared/iris_calibration_train.npz",
    )
    parser.add_argument(
        "--test-output", type=Path, default=ROOT / "data/prepared/iris_test.npz"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/data_preparation.json"
    )
    return parser.parse_args()


def counts(labels: np.ndarray) -> dict[str, int]:
    values, frequencies = np.unique(labels, return_counts=True)
    return {str(int(value)): int(frequency) for value, frequency in zip(values, frequencies)}


def main() -> None:
    args = parse_args()
    if not args.scaler.is_file():
        raise FileNotFoundError(args.scaler)

    features, labels = load_iris(return_X_y=True)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    scaler = joblib.load(args.scaler)
    train_scaled = scaler.transform(train_x).astype(np.float32)
    test_scaled = scaler.transform(test_x).astype(np.float32)

    if np.intersect1d(
        np.ascontiguousarray(train_x).view(
            np.dtype((np.void, train_x.dtype.itemsize * train_x.shape[1]))
        ),
        np.ascontiguousarray(test_x).view(
            np.dtype((np.void, test_x.dtype.itemsize * test_x.shape[1]))
        ),
    ).size:
        raise RuntimeError("Há amostras idênticas compartilhadas entre treino e teste")

    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.calibration_output,
        features=train_scaled,
        labels=train_y,
        raw_features=train_x,
    )
    np.savez_compressed(
        args.test_output,
        features=test_scaled,
        labels=test_y,
        raw_features=test_x,
    )

    report = {
        "status": "passed",
        "dataset": "scikit-learn Iris",
        "sklearn_version": sklearn.__version__,
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "calibration_source": "training split only",
            "test_used_for_calibration": False,
        },
        "preprocessing": "StandardScaler previously fitted on the training split",
        "scaler": relative(args.scaler),
        "scaler_sha256": sha256(args.scaler),
        "scaler_mean": np.asarray(scaler.mean_).tolist(),
        "scaler_scale": np.asarray(scaler.scale_).tolist(),
        "calibration": {
            "path": relative(args.calibration_output),
            "sha256": sha256(args.calibration_output),
            "samples": int(len(train_y)),
            "class_counts": counts(train_y),
            "feature_min": train_scaled.min(axis=0).tolist(),
            "feature_max": train_scaled.max(axis=0).tolist(),
        },
        "test": {
            "path": relative(args.test_output),
            "sha256": sha256(args.test_output),
            "samples": int(len(test_y)),
            "class_counts": counts(test_y),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
