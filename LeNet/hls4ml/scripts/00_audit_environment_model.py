#!/usr/bin/env python3
"""Audita ambiente, arquitetura, faixas numericas e dados da LeNet/hls4ml."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hls4ml
import keras
import numpy as np
import tensorflow as tf
from hls4ml.backends.backend import get_backend


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MODEL = PROJECT / "lenet_mnist_final.h5"
MNIST = Path.home() / ".keras/datasets/mnist.npz"
REPORT = ROOT / "reports/model_profile_initial.json"
TEST_DATA = ROOT / "data/mnist_test_uint8.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return {"command": command, "returncode": result.returncode, "output": result.stdout}
    except Exception as error:
        return {"command": command, "error": repr(error)}


def main() -> None:
    if not MODEL.is_file() or not MNIST.is_file():
        raise FileNotFoundError(f"model={MODEL.is_file()}, mnist={MNIST.is_file()}")
    model = keras.models.load_model(MODEL, compile=False)
    expected = ["conv1", "pool1", "conv2", "pool2", "flatten", "dense1", "dense2", "output"]
    if [layer.name for layer in model.layers] != expected or model.count_params() != 44426:
        raise RuntimeError("arquitetura LeNet inesperada")
    if keras.activations.serialize(model.layers[-1].activation) != "linear":
        raise RuntimeError("a saida deve ser linear, sem Softmax")

    raw = np.load(MNIST)
    images_uint8 = raw["x_test"]
    labels = raw["y_test"].reshape(-1).astype(np.int64)
    images = (images_uint8.astype(np.float32) / 255.0)[..., np.newaxis]
    probe = keras.Model(model.inputs, [layer.output for layer in model.layers])
    activations = probe.predict(images, batch_size=256, verbose=0)
    backend = get_backend("Vitis")
    layers = []
    for layer, values in zip(model.layers, activations):
        entry = {
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": list(values.shape[1:]),
            "parameters": int(layer.count_params()),
            "activation_range_all_10000_tests": {
                "min": float(values.min()),
                "max": float(values.max()),
                "percentile_0_1": float(np.percentile(values, 0.1)),
                "percentile_99_9": float(np.percentile(values, 99.9)),
            },
            "weights": [],
        }
        for index, weights in enumerate(layer.get_weights()):
            entry["weights"].append({
                "index": index, "shape": list(weights.shape),
                "min": float(weights.min()), "max": float(weights.max()),
            })
        if layer.__class__.__name__ in ("Conv2D", "Dense"):
            kernel = layer.get_weights()[0]
            n_in = int(np.prod(kernel.shape[:-1]))
            n_out = int(kernel.shape[-1])
            entry["multiplier_dimensions"] = [n_in, n_out]
            entry["multiplications"] = n_in * n_out
            entry["valid_reuse_factors_up_to_512"] = [
                int(value) for value in backend.get_valid_reuse_factors(n_in, n_out) if int(value) <= 512
            ]
        layers.append(entry)

    TEST_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TEST_DATA, images=images_uint8, labels=labels, test_indices=np.arange(len(labels), dtype=np.int64))
    report = {
        "status": "passed",
        "model": str(MODEL),
        "model_sha256": sha256(MODEL),
        "name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameters": int(model.count_params()),
        "macs_per_inference": 281640,
        "final_activation": "linear",
        "softmax": False,
        "test_samples": int(len(labels)),
        "test_data": str(TEST_DATA.relative_to(ROOT)),
        "test_data_sha256": sha256(TEST_DATA),
        "preprocessing": "uint8 -> float32 / 255.0; add channels_last dimension",
        "layers": layers,
        "software": {
            "python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "hls4ml": hls4ml.__version__, "numpy": np.__version__,
            "vitis_hls": shutil.which("vitis_hls") or "/opt/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls",
            "vivado": shutil.which("vivado") or "/opt/Xilinx/Vivado/2024.2/bin/vivado",
        },
        "vitis_hls_version": command_output(["/opt/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls", "-version"]),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "model_sha256", "parameters", "final_activation", "test_data_sha256", "software")}, indent=2))


if __name__ == "__main__":
    main()
