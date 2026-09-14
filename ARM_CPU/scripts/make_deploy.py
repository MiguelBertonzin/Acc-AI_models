#!/usr/bin/env python3
"""Monta pacote autocontido para copiar à ZCU104, sempre dentro de ARM_CPU."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from model_specs import ROOT

DEPLOY = ROOT / "deploy_zcu104"
FILES = [
    "models/tflite/mlp_fp32.tflite", "models/tflite/mlp_int8.tflite",
    "models/tflite/lenet_fp32.tflite", "models/tflite/lenet_int8.tflite",
    "models/tflite/resnet8_fp32.tflite", "models/tflite/resnet8_int8.tflite",
    "data/mlp/iris_test.npz", "data/mlp/iris_calibration_train.npz", "data/mlp/iris_scaler_params.npz",
    "data/lenet/mnist_test_uint8.npz", "data/lenet/mnist_calibration_train_uint8.npz",
    "data/resnet8/cifar10_test_uint8.npz", "data/resnet8/cifar10_calibration_train_uint8.npz",
    "scripts/model_specs.py", "scripts/infer_arm.py", "scripts/benchmark_arm.py", "scripts/check_board.py",
    "requirements-board.txt", "README.md", "GUIA_CHATGPT_PYNQ_ZCU104.md", "run_all_arm.sh",
    "manifests/conversion_manifest.json", "results/host_validation.json",
]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    DEPLOY.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative in FILES:
        source = ROOT / relative
        target = DEPLOY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({"file": relative, "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "target": "AMD/Xilinx ZCU104, Linux AArch64, ARM Cortex-A53", "files": entries}
    (DEPLOY / "DEPLOY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"{entry['sha256']}  {entry['file']}" for entry in entries]
    (DEPLOY / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
