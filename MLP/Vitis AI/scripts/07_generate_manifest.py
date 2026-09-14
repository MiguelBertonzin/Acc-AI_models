#!/usr/bin/env python3
"""Gera inventário SHA-256 dos arquivos reproduzíveis do fluxo Vitis AI."""

from __future__ import annotations

import json
from pathlib import Path

from common import ROOT, relative, sha256


OUTPUT_JSON = ROOT / "manifests/flow_manifest.json"
OUTPUT_TXT = ROOT / "manifests/SHA256SUMS.txt"
INCLUDE_ROOTS = [
    ROOT / "README.md",
    ROOT / "README_CHATGPT_WEB_ZCU104_XMODEL.md",
    ROOT / "source",
    ROOT / "config",
    ROOT / "docker",
    ROOT / "board",
    ROOT / "scripts",
    ROOT / "data/prepared",
    ROOT / "models/float",
    ROOT / "artifacts/quantized",
    ROOT / "artifacts/compiled",
    ROOT / "artifacts/deploy",
    ROOT / "reports",
    ROOT / "results",
    ROOT / "logs",
]


def collect() -> list[Path]:
    paths: list[Path] = []
    for root in INCLUDE_ROOTS:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path in (OUTPUT_JSON, OUTPUT_TXT):
                continue
            paths.append(path)
    return sorted(set(paths), key=relative)


def main() -> None:
    files = [
        {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in collect()
    ]
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps({"root": str(ROOT), "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT_TXT.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files),
        encoding="utf-8",
    )
    print(f"{len(files)} arquivos registrados em {relative(OUTPUT_JSON)}")


if __name__ == "__main__":
    main()
