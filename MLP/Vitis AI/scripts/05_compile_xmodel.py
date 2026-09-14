#!/usr/bin/env python3
"""Compila o modelo quantizado para a DPUCZDX8G B4096 da ZCU104."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import ROOT, relative, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts/quantized/iris_mlp_logits_quantized.h5",
    )
    parser.add_argument(
        "--arch", type=Path, default=ROOT / "config/arch_zcu104_vai3.5.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/compiled/iris_mlp_zcu104_vai3_5",
    )
    parser.add_argument("--net-name", default="iris_mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.model, args.arch):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} não está vazio; preserve o resultado ou use outro --output-dir"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs/compiler/compile_iris_mlp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "vai_c_tensorflow2",
        "--model",
        str(args.model),
        "--arch",
        str(args.arch),
        "--output_dir",
        str(args.output_dir),
        "--net_name",
        args.net_name,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout)
    xmodels = sorted(args.output_dir.rglob("*.xmodel"))
    target = json.loads(args.arch.read_text(encoding="utf-8"))["target"]
    report = {
        "status": "passed" if completed.returncode == 0 and xmodels else "failed",
        "command": command,
        "returncode": completed.returncode,
        "model": relative(args.model),
        "model_sha256": sha256(args.model),
        "arch": relative(args.arch),
        "arch_sha256": sha256(args.arch),
        "target": target,
        "output_dir": relative(args.output_dir),
        "xmodels": [
            {
                "path": relative(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in xmodels
        ],
        "log": relative(log_path),
        "log_sha256": sha256(log_path),
    }
    report_path = ROOT / "reports/compilation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(f"Compilação falhou; consulte {log_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
