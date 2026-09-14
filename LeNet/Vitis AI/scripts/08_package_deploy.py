#!/usr/bin/env python3
"""Empacota XModel, teste, referencia e scripts para executar na ZCU104."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/deploy/lenet_mnist_zcu104_vai3_5")
    args = parser.parse_args()
    sources = {
        ROOT / "artifacts/compiled/lenet_mnist_zcu104_vai3_5/lenet_mnist_no_softmax.xmodel": "lenet_mnist_no_softmax.xmodel",
        ROOT / "data/prepared/mnist_test_uint8.npz": "mnist_test_uint8.npz",
        ROOT / "results/quantized_test_outputs.npz": "quantized_test_outputs.npz",
        ROOT / "board/00_inspect_dpu.py": "00_inspect_dpu.py",
        ROOT / "board/01_smoke_inference.py": "01_smoke_inference.py",
        ROOT / "board/02_validate_accuracy.py": "02_validate_accuracy.py",
        ROOT / "config/arch_zcu104_vai3.5.json": "arch_zcu104_vai3.5.json",
    }
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} nao esta vazio; preserve o pacote existente")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results").mkdir(exist_ok=True)
    copied = []
    for source, name in sources.items():
        destination = args.output_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)
    manifest = {
        "package": args.output_dir.name,
        "target": "DPUCZDX8G_ISA1_B4096",
        "board": "ZCU104",
        "vitis_ai": "3.5 host toolchain",
        "required_before_run": "xdputil query must report a matching DPU fingerprint",
        "test_samples": 10000,
        "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in copied],
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_path = args.output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in copied) + f"{sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
