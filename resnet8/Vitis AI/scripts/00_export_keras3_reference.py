#!/usr/bin/env python3
"""Exporta uma prova determinística do modelo Keras 3 original."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/float/resnet8_cifar10_keras3_no_softmax.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/prepared/keras3_conversion_probe.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/keras3_reference.json",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples deve ser positivo")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    inputs = rng.random((args.samples, 32, 32, 3), dtype=np.float32)

    model = tf.keras.models.load_model(args.model, compile=False)
    outputs = model.predict(inputs, batch_size=args.samples, verbose=0).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, inputs=inputs, outputs=outputs)

    report = {
        "purpose": "Keras 3 to Keras 2 conversion reference",
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__ if hasattr(tf.keras, "__version__") else None,
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "probe": str(args.output.relative_to(ROOT)),
        "probe_sha256": sha256(args.output),
        "seed": args.seed,
        "samples": args.samples,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameters": int(model.count_params()),
        "last_layer": model.layers[-1].name,
        "last_activation": model.layers[-1].activation.__name__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
