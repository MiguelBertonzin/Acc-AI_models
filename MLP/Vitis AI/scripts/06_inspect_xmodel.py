#!/usr/bin/env python3
"""Inspeciona subgrafos, tensores e fixed-points do XModel compilado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xir

from common import ROOT, relative, sha256


def tensor_info(tensor) -> dict:
    return {
        "name": tensor.name,
        "dims": list(tensor.dims),
        "dtype": str(tensor.dtype),
        "fix_point": tensor.get_attr("fix_point") if tensor.has_attr("fix_point") else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xmodel",
        type=Path,
        default=ROOT / "artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/xmodel_inspection.json"
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
        device = subgraph.get_attr("device") if subgraph.has_attr("device") else None
        subgraphs.append(
            {
                "name": subgraph.get_name(),
                "device": device,
                "ops": len(subgraph.get_ops()),
                "inputs": [tensor_info(item) for item in subgraph.get_input_tensors()],
                "outputs": [tensor_info(item) for item in subgraph.get_output_tensors()],
            }
        )

    dpu_subgraphs = [item for item in subgraphs if item["device"] == "DPU"]
    report = {
        "status": "passed" if dpu_subgraphs else "failed",
        "xmodel": relative(args.xmodel),
        "xmodel_sha256": sha256(args.xmodel),
        "graph": graph.get_name(),
        "subgraph_count": len(subgraphs),
        "dpu_subgraph_count": len(dpu_subgraphs),
        "subgraphs": subgraphs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not dpu_subgraphs:
        raise RuntimeError("O XModel não contém subgrafo DPU")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
