#!/usr/bin/env python3
"""Compila o H5 quantizado para DPUCZDX8G B4096 da ZCU104."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/quantized/lenet_mnist_no_softmax_quantized.h5")
    parser.add_argument("--arch", type=Path, default=ROOT / "config/arch_zcu104_vai3.5.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/compiled/lenet_mnist_zcu104_vai3_5")
    parser.add_argument("--net-name", default="lenet_mnist_no_softmax")
    args = parser.parse_args()

    for required in (args.model, args.arch):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} nao esta vazio; preserve a execucao existente")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs/compiler/compile_lenet_mnist.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "vai_c_tensorflow2", "--model", str(args.model), "--arch", str(args.arch),
        "--output_dir", str(args.output_dir), "--net_name", args.net_name,
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout)
    xmodels = sorted(args.output_dir.rglob("*.xmodel"))
    report = {
        "status": "passed" if completed.returncode == 0 and xmodels else "failed",
        "tool": "Vitis AI Compiler for TensorFlow 2 (vai_c_tensorflow2)",
        "command": command,
        "returncode": completed.returncode,
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": sha256(args.model),
        "arch": str(args.arch.relative_to(ROOT)),
        "arch_sha256": sha256(args.arch),
        "target": json.loads(args.arch.read_text(encoding="utf-8"))["target"],
        "output_dir": str(args.output_dir.relative_to(ROOT)),
        "xmodels": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in xmodels],
        "log": str(log_path.relative_to(ROOT)),
        "log_sha256": sha256(log_path),
    }
    report_path = ROOT / "reports/compilation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(f"compilacao falhou; consulte {log_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
