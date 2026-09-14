#!/usr/bin/env python3
"""Gera inventário e hashes dos artefatos reproduzíveis do fluxo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "manifests/host_flow_manifest.json"
OUTPUT_TXT = ROOT / "manifests/HOST_FLOW_SHA256SUMS.txt"
INCLUDE_ROOTS = [
    ROOT / "README.md",
    ROOT / "README.md",
    ROOT / "config",
    ROOT / "docker",
    ROOT / "scripts",
    ROOT / "board",
    ROOT / "models/float",
    ROOT / "data/prepared",
    ROOT / "artifacts/quantized",
    ROOT / "artifacts/compiled",
    ROOT / "artifacts/deploy",
    ROOT / "reports",
    ROOT / "results",
    ROOT / "logs",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> list[Path]:
    paths = []
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
    return sorted(set(paths), key=lambda item: str(item.relative_to(ROOT)))


def main() -> None:
    files = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
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
    print(f"{len(files)} arquivos registrados em {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
