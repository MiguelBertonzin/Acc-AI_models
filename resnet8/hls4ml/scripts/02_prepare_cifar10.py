#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hls_common import HLS_ROOT, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o test set CIFAR-10 para o fluxo HLS.")
    parser.add_argument("--output-dir", type=Path, default=HLS_ROOT / "data")
    args = parser.parse_args()

    from tensorflow.keras.datasets import cifar10

    out = args.output_dir.expanduser().resolve()
    allowed = (HLS_ROOT / "data").resolve()
    if out != allowed:
        raise ValueError(f"Os dados deste fluxo devem ficar em {allowed}.")
    out.mkdir(parents=True, exist_ok=True)

    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = np.ascontiguousarray(x_test.astype(np.float32) / 255.0)
    y_test = y_test.reshape(-1).astype(np.uint8)
    x_path = out / "cifar10_x_test.npy"
    y_path = out / "cifar10_y_test.npy"
    np.save(x_path, x_test)
    np.save(y_path, y_test)

    manifest = {
        "x": {
            "path": str(x_path),
            "shape": list(x_test.shape),
            "dtype": str(x_test.dtype),
            "min": float(x_test.min()),
            "max": float(x_test.max()),
            "sha256": sha256(x_path),
        },
        "y": {
            "path": str(y_path),
            "shape": list(y_test.shape),
            "dtype": str(y_test.dtype),
            "min": int(y_test.min()),
            "max": int(y_test.max()),
            "sha256": sha256(y_path),
        },
        "normalization": "x.astype(float32) / 255.0",
    }
    manifest_path = out / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

