#!/usr/bin/env python3
"""Analise analitica e reproduzivel do espaco de ReuseFactor da LeNet.

Este script nao converte o modelo, nao executa sintese e nao gera IP. Ele replica
as regras de validade/limite de multiplicadores do backend FPGA do hls4ml 1.3.0
e escreve tabelas para orientar o pequeno sweep que sera sintetizado depois.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HLS_ROOT = Path(__file__).resolve().parents[1]
REPORTS = HLS_ROOT / "reports"
ZCU104_DSPS = 1728

LAYERS = {
    "conv1": {"n_in": 25, "n_out": 6, "schedule_units": 24 * 24},
    "conv2": {"n_in": 150, "n_out": 16, "schedule_units": 8 * 8},
    "dense1": {"n_in": 256, "n_out": 120, "schedule_units": 120},
    "dense2": {"n_in": 120, "n_out": 84, "schedule_units": 84},
    "output": {"n_in": 84, "n_out": 10, "schedule_units": 10},
}


def valid_reuse_factors(n_in: int, n_out: int) -> list[int]:
    """Replica FPGABackend._validate_reuse_factor do hls4ml 1.3.0."""
    valid = []
    for rf in range(1, n_in * n_out + 1):
        multfactor = min(n_in, rf)
        multiplier_limit = math.ceil((n_in * n_out) / multfactor)
        ok = multiplier_limit % n_out == 0 or rf >= n_in
        ok = ok and (rf % n_in == 0 or rf < n_in)
        ok = ok and ((n_in * n_out) % rf == 0)
        if ok:
            valid.append(rf)
    return valid


def multiplier_limit(n_in: int, n_out: int, rf: int) -> int:
    """Limite estrutural usado pelo backend; nao equivale a DSP pos-sintese."""
    return math.ceil((n_in * n_out) / min(n_in, rf))


def schedule_proxy(units: int, rf: int) -> int:
    """Proxy derivado de set_target_reuse_factor: (RF + 6) * unidades."""
    return (rf + 6) * units


def closest(values: list[int], requested: int) -> int:
    return min(values, key=lambda value: (abs(value - requested), value))


def largest_nondominated_at_most(values: list[int], cap: int, n_in: int) -> int:
    return max(value for value in values if value <= min(cap, n_in))


def summarize_candidate(name: str, reuse: dict[str, int], note: str) -> dict:
    rows = []
    for layer_name, dims in LAYERS.items():
        rf = reuse[layer_name]
        rows.append(
            {
                "layer": layer_name,
                "reuse_factor": rf,
                "multiplier_limit": multiplier_limit(dims["n_in"], dims["n_out"], rf),
                "schedule_proxy_cycles": schedule_proxy(dims["schedule_units"], rf),
            }
        )
    total = sum(row["multiplier_limit"] for row in rows)
    return {
        "name": name,
        "note": note,
        "reuse": reuse,
        "layers": rows,
        "multiplier_limit_total": total,
        "multiplier_limit_as_zcu104_dsp_percent_if_one_to_one": 100.0 * total / ZCU104_DSPS,
        "two_dsp_per_multiplier_scenario_percent": 200.0 * total / ZCU104_DSPS,
        "maximum_schedule_proxy_cycles": max(row["schedule_proxy_cycles"] for row in rows),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    valid = {
        name: valid_reuse_factors(dims["n_in"], dims["n_out"])
        for name, dims in LAYERS.items()
    }

    candidates = []
    for cap in (16, 32, 64, 128, 256, 512):
        reuse = {
            name: largest_nondominated_at_most(valid[name], cap, dims["n_in"])
            for name, dims in LAYERS.items()
        }
        candidates.append(
            summarize_candidate(
                f"resource_cap_{cap}",
                reuse,
                "Maior RF valido, nao dominado e menor ou igual ao teto global.",
            )
        )

    candidates.extend(
        [
            summarize_candidate(
                "current_A",
                {"conv1": 5, "conv2": 30, "dense1": 128, "dense2": 120, "output": 84},
                "Baseline inicialmente congelado.",
            ),
            summarize_candidate(
                "requested_64_as_same_value",
                {name: closest(values, 64) for name, values in valid.items()},
                "O que o ajuste ao RF valido mais proximo de 64 produz; nao recomendado.",
            ),
            summarize_candidate(
                "recommended_E_cap64_rate_balanced",
                {"conv1": 5, "conv2": 50, "dense1": 64, "dense2": 60, "output": 42},
                "Teto 64 com RF explicito por camada e proxy de vazao balanceado.",
            ),
            summarize_candidate(
                "recommended_A_refined_same_proxy",
                {"conv1": 5, "conv2": 150, "dense1": 128, "dense2": 120, "output": 84},
                "Reduz o limite estrutural do baseline sem aumentar seu pior proxy de ciclos.",
            ),
        ]
    )

    payload = {
        "schema_version": 1,
        "method": {
            "hls4ml_version": "1.3.0",
            "validity_rule": "FPGABackend._validate_reuse_factor",
            "multiplier_limit": "ceil(n_in*n_out/min(n_in, reuse_factor))",
            "schedule_proxy": "(reuse_factor+6)*schedule_units, derived from set_target_reuse_factor",
            "warning": "Analytical estimates are not synthesis results.",
        },
        "device": {"board": "ZCU104", "part": "xczu7ev-ffvc1156-2-e", "dsp48e2": ZCU104_DSPS},
        "layers": {
            name: {**dims, "valid_reuse_factors": valid[name]}
            for name, dims in LAYERS.items()
        },
        "candidates": candidates,
    }

    json_path = REPORTS / "reuse_factor_space.json"
    csv_path = REPORTS / "reuse_factor_candidates.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate",
                "conv1_rf",
                "conv2_rf",
                "dense1_rf",
                "dense2_rf",
                "output_rf",
                "multiplier_limit_total",
                "one_dsp_scenario_percent",
                "two_dsp_scenario_percent",
                "maximum_schedule_proxy_cycles",
            ]
        )
        for item in candidates:
            writer.writerow(
                [
                    item["name"],
                    *[item["reuse"][name] for name in LAYERS],
                    item["multiplier_limit_total"],
                    f'{item["multiplier_limit_as_zcu104_dsp_percent_if_one_to_one"]:.3f}',
                    f'{item["two_dsp_per_multiplier_scenario_percent"]:.3f}',
                    item["maximum_schedule_proxy_cycles"],
                ]
            )

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    for item in candidates:
        print(
            f'{item["name"]:38s} multipliers={item["multiplier_limit_total"]:4d} '
            f'proxy={item["maximum_schedule_proxy_cycles"]:6d}'
        )


if __name__ == "__main__":
    main()
