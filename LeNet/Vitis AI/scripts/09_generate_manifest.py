#!/usr/bin/env python3
"""Gera inventario e hashes de todos os artefatos reproduziveis do fluxo."""

from __future__ import annotations

import json
from pathlib import Path

from common import sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "manifests/host_flow_manifest.json"
OUTPUT_TXT = ROOT / "manifests/HOST_FLOW_SHA256SUMS.txt"
INCLUDE_ROOTS = [
    ROOT / "README.md", ROOT / "HANDOFF_CHATGPT_WEB_ZCU104_VITIS_AI.txt", ROOT / "run_full_flow.sh",
    ROOT / "config", ROOT / "docker", ROOT / "scripts", ROOT / "board",
    ROOT / "models/float", ROOT / "data/prepared", ROOT / "artifacts/quantized",
    ROOT / "artifacts/compiled", ROOT / "artifacts/deploy", ROOT / "reports", ROOT / "results", ROOT / "logs",
]


def collect() -> list[Path]:
    paths = []
    for root in INCLUDE_ROOTS:
        candidates = [root] if root.is_file() else (root.rglob("*") if root.exists() else [])
        for path in candidates:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path not in (OUTPUT_JSON, OUTPUT_TXT):
                paths.append(path)
    return sorted(set(paths), key=lambda item: str(item.relative_to(ROOT)))


def main() -> None:
    files = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in collect()]
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({"root": str(ROOT), "files": files}, indent=2) + "\n", encoding="utf-8")
    OUTPUT_TXT.write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in files), encoding="utf-8")
    print(f"{len(files)} arquivos registrados em {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
