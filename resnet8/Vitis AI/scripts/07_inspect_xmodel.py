#!/usr/bin/env python3
"""Inspeciona subgrafos, tensores e fixed-points do XModel compilado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import xir


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_info(tensor) -> dict:
    return {
        "name": tensor.name,
        "dims": list(tensor.dims),
        "dtype": str(tensor.dtype),
        "fix_point": tensor.get_attr("fix_point")
        if tensor.has_attr("fix_point")
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xmodel",
        type=Path,
        default=ROOT
        / "artifacts/compiled/resnet8_no_softmax_zcu104_vai3_5/resnet8_no_softmax.xmodel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/xmodel_inspection.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xmodel.is_file():
        raise FileNotFoundError(args.xmodel)
    graph = xir.Graph.deserialize(str(args.xmodel))
    children = graph.get_root_subgraph().toposort_child_subgraph()
    subgraphs = []
    for subgraph in children:
        subgraphs.append(
            {
                "name": subgraph.get_name(),
                "device": subgraph.get_attr("device")
                if subgraph.has_attr("device")
                else None,
                "ops": len(subgraph.get_ops()),
                "inputs": [tensor_info(item) for item in subgraph.get_input_tensors()],
                "outputs": [
                    tensor_info(item) for item in subgraph.get_output_tensors()
                ],
            }
        )
    report = {
        "xmodel": str(args.xmodel.relative_to(ROOT)),
        "xmodel_sha256": sha256(args.xmodel),
        "graph": graph.get_name(),
        "subgraph_count": len(subgraphs),
        "dpu_subgraph_count": sum(item["device"] == "DPU" for item in subgraphs),
        "subgraphs": subgraphs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
