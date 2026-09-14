#!/usr/bin/env python3
"""Empacota XModel, pré-processamento e executor VART para a ZCU104."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from common import ROOT, relative, sha256


OUTPUT = ROOT / "artifacts/deploy/iris_mlp_zcu104_vai3_5"
FILES = {
    ROOT / "artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel": "iris_mlp.xmodel",
    ROOT / "config/arch_zcu104_vai3.5.json": "arch_zcu104_vai3.5.json",
    ROOT / "config/preprocessing.json": "preprocessing.json",
    ROOT / "board/run_iris_mlp.py": "run_iris_mlp.py",
    ROOT / "board/vart_common.py": "vart_common.py",
    ROOT / "board/validate_iris_mlp.py": "validate_iris_mlp.py",
    ROOT / "board/benchmark_iris_mlp.py": "benchmark_iris_mlp.py",
    ROOT / "board/discover_zcu104_sensors.py": "discover_zcu104_sensors.py",
    ROOT / "data/prepared/iris_test.npz": "iris_test.npz",
    ROOT / "results/float_test_outputs.npz": "float_test_outputs.npz",
    ROOT / "results/quantized_test_outputs.npz": "quantized_test_outputs.npz",
}


def main() -> None:
    for source in FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
    # Atualização idempotente apenas dos arquivos gerenciados deste pacote.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for source, name in FILES.items():
        destination = OUTPUT / name
        shutil.copy2(source, destination)
        copied.append(destination)

    manifest = {
        "status": "ready_for_zcu104",
        "target": "DPUCZDX8G_ISA1_B4096",
        "vitis_ai_version": "3.5",
        "model": "iris_mlp.xmodel",
        "input": {"shape": [1, 4], "dtype": "int8", "fix_point": 5},
        "output": {"shape": [1, 3], "dtype": "int8", "fix_point": 3},
        "preprocessing": "preprocessing.json",
        "runner": "run_iris_mlp.py",
        "validator": "validate_iris_mlp.py",
        "benchmark": "benchmark_iris_mlp.py",
        "canonical_test_samples": 30,
        "example": "python3 run_iris_mlp.py --features 5.1 3.5 1.4 0.2",
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in copied
        ],
    }
    manifest_path = OUTPUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_paths = sorted([*copied, manifest_path], key=lambda path: path.name)
    (OUTPUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps({"package_dir": relative(OUTPUT), **manifest}, indent=2))


if __name__ == "__main__":
    main()
