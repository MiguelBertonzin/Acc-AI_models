#!/usr/bin/env python3
"""Coleta versões e configuração do DPU antes de executar o XModel."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/board_environment.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imports = {}
    for module_name in ("xir", "vart"):
        try:
            module = __import__(module_name)
            imports[module_name] = {
                "available": True,
                "version": getattr(module, "__version__", None),
                "path": getattr(module, "__file__", None),
            }
        except Exception as error:
            imports[module_name] = {
                "available": False,
                "error": repr(error),
            }

    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "imports": imports,
        "xdputil_query": run(["xdputil", "query"]),
        "xdputil_status": run(["xdputil", "status"]),
        "uname": run(["uname", "-a"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not all(imports[name]["available"] for name in ("xir", "vart")):
        raise RuntimeError("XIR/VART não estão disponíveis na placa")
    if report["xdputil_query"]["returncode"] != 0:
        raise RuntimeError("xdputil query falhou; o DPU/runtime ainda não está pronto")


if __name__ == "__main__":
    main()
