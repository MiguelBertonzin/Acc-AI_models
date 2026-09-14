#!/usr/bin/env python3
"""Inspeciona subgrafos, tensores e fixed-points do XModel compilado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xir

from common import sha256


ROOT = Path(__file__).resolve().parents[1]


def tensor_info(tensor) -> dict:
    return {
        "name": tensor.name,
        "dims": list(tensor.dims),
        "dtype": str(tensor.dtype),
        "fix_point": tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmodel", type=Path, default=ROOT / "artifacts/compiled/lenet_mnist_zcu104_vai3_5/lenet_mnist_no_softmax.xmodel")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/xmodel_inspection.json")
    args = parser.parse_args()
    if not args.xmodel.is_file():
        raise FileNotFoundError(args.xmodel)
    graph = xir.Graph.deserialize(str(args.xmodel))
    children = graph.get_root_subgraph().toposort_child_subgraph()
    subgraphs = []
    for subgraph in children:
        subgraphs.append({
            "name": subgraph.get_name(),
            "device": subgraph.get_attr("device") if subgraph.has_attr("device") else None,
            "ops": len(subgraph.get_ops()),
            "inputs": [tensor_info(item) for item in subgraph.get_input_tensors()],
            "outputs": [tensor_info(item) for item in subgraph.get_output_tensors()],
        })
    dpu_count = sum(item["device"] == "DPU" for item in subgraphs)
    report = {
        "status": "passed" if dpu_count > 0 else "failed",
        "xmodel": str(args.xmodel.relative_to(ROOT)),
        "xmodel_sha256": sha256(args.xmodel),
        "xmodel_bytes": args.xmodel.stat().st_size,
        "graph": graph.get_name(),
        "subgraph_count": len(subgraphs),
        "dpu_subgraph_count": dpu_count,
        "subgraphs": subgraphs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("XModel nao contem subgrafo DPU")


if __name__ == "__main__":
    main()
