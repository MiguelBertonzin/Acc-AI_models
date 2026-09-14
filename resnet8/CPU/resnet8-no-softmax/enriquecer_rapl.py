#!/usr/bin/env python3
"""Enrich CPU benchmark CSV files with privileged turbostat/RAPL telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_LOG = Path("/tmp/resnet8_cpu_turbostat.log")
DEFAULT_RESULTS = HERE / "resultados"
DEFAULT_BENCHMARK = HERE / "benchmark_no_softmax_cpu.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--interval-seconds", type=float, default=0.1)
    return p.parse_args()


def number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_log(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    header: list[str] | None = None
    required = {"Time_Of_Day_Seconds", "PkgWatt", "CorWatt", "Busy%", "Bzy_MHz", "PkgTmp"}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if "Time_Of_Day_Seconds" in fields:
            header = fields
            missing = required.difference(header)
            if missing:
                raise ValueError(f"turbostat header lacks {sorted(missing)}: {header}")
            continue
        if header is None or len(fields) != len(header):
            continue
        item = {name: number(value) for name, value in zip(header, fields, strict=True)}
        if all(math.isfinite(item[name]) for name in required):
            rows.append(item)
    if not rows:
        raise RuntimeError(f"No valid turbostat samples parsed from {path}")
    rows.sort(key=lambda x: x["Time_Of_Day_Seconds"])
    return rows


def overlaps(
    samples: list[dict[str, float]], start: float, end: float, interval: float
) -> list[tuple[dict[str, float], float]]:
    selected = []
    for sample in samples:
        sample_end = sample["Time_Of_Day_Seconds"]
        sample_start = sample_end - interval
        overlap = max(0.0, min(sample_end, end) - max(sample_start, start))
        if overlap > 0:
            selected.append((sample, overlap))
    return selected


def weighted_mean(selected: list[tuple[dict[str, float], float]], name: str) -> float:
    valid = [(x[name], weight) for x, weight in selected if math.isfinite(x[name])]
    total = sum(weight for _, weight in valid)
    return sum(value * weight for value, weight in valid) / total if total else math.nan


def integrated(selected: list[tuple[dict[str, float], float]], name: str) -> float:
    return sum(x[name] * weight for x, weight in selected if math.isfinite(x[name]))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def load_benchmark(path: Path):
    spec = importlib.util.spec_from_file_location("cpu_benchmark_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")
    samples = parse_log(args.log)
    metadata_path = args.results_dir / "metadados.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline_selected = overlaps(
        samples,
        float(metadata["baseline_start_epoch_s"]),
        float(metadata["baseline_end_epoch_s"]),
        args.interval_seconds,
    )
    if not baseline_selected:
        raise RuntimeError("No turbostat samples overlap the benchmark idle baseline")
    baseline_power = weighted_mean(baseline_selected, "PkgWatt")

    sizes = [int(x) for x in metadata["sample_sizes"]]
    total_matched = 0
    for n in sizes:
        path = args.results_dir / f"passagens_n{n}.csv"
        rows = read_csv(path)
        for row in rows:
            raw_psutil_frequency = number(str(row.get("cpu_frequency_psutil_mean_mhz", "nan")))
            if math.isfinite(raw_psutil_frequency) and raw_psutil_frequency < 100.0:
                row["cpu_frequency_psutil_mean_mhz"] = math.nan
            start = float(row["start_epoch_s"])
            end = float(row["end_epoch_s"])
            duration = float(row["wall_time_s"])
            selected = overlaps(samples, start, end, args.interval_seconds)
            coverage = sum(weight for _, weight in selected)
            if not selected or coverage < min(duration, args.interval_seconds) * 0.5:
                raise RuntimeError(
                    f"Insufficient turbostat coverage for n={n}, pass={row['repetition']}: "
                    f"coverage={coverage:.6f}s duration={duration:.6f}s"
                )
            package_power = weighted_mean(selected, "PkgWatt")
            package_energy = integrated(selected, "PkgWatt")
            if coverage < duration * 0.98:
                package_energy *= duration / coverage
            dynamic_power = max(package_power - baseline_power, 0.0)
            dynamic_energy = max(package_energy - baseline_power * duration, 0.0)
            row["rapl_samples"] = len(selected)
            row["rapl_coverage_s"] = coverage
            row["cpu_busy_percent_turbostat"] = weighted_mean(selected, "Busy%")
            row["cpu_busy_frequency_mean_mhz"] = weighted_mean(selected, "Bzy_MHz")
            row["cpu_package_temperature_mean_c"] = weighted_mean(selected, "PkgTmp")
            row["power_package_mean_w"] = package_power
            row["power_cores_mean_w"] = weighted_mean(selected, "CorWatt")
            row["power_dram_mean_w"] = math.nan
            row["power_dynamic_mean_w"] = dynamic_power
            row["energy_package_j"] = package_energy
            row["energy_per_inference_mj"] = package_energy * 1000.0 / n
            row["dynamic_energy_per_inference_mj"] = dynamic_energy * 1000.0 / n
            total_matched += len(selected)
        write_csv(path, rows)

    module = load_benchmark(args.benchmark_script)
    results = module.rebuild_results(
        args.results_dir,
        sizes,
        [int(x) for x in metadata["repetition_prefixes"]],
    )
    baseline_psutil_frequency = number(str(metadata.get("baseline_frequency_mhz", "nan")))
    psutil_frequency_valid = math.isfinite(baseline_psutil_frequency) and baseline_psutil_frequency >= 100.0
    if not psutil_frequency_valid:
        metadata["baseline_frequency_mhz"] = math.nan
    metadata.update(
        {
            "rapl_enriched": True,
            "psutil_frequency_valid": psutil_frequency_valid,
            "psutil_frequency_note": ("cpu_freq current was below 100 MHz and was discarded; turbostat Bzy_MHz is authoritative" if not psutil_frequency_valid else "valid"),
            "turbostat_log_source": str(args.log.resolve()),
            "turbostat_log_sha256": sha256(args.log),
            "turbostat_interval_seconds": args.interval_seconds,
            "turbostat_samples_parsed": len(samples),
            "turbostat_sample_references_to_passes": total_matched,
            "rapl_baseline_samples": len(baseline_selected),
            "rapl_baseline_package_power_w": baseline_power,
            "rapl_columns": ["PkgWatt", "CorWatt"],
            "turbostat_columns": [
                "Time_Of_Day_Seconds",
                "Busy%",
                "Bzy_MHz",
                "PkgTmp",
                "PkgWatt",
                "CorWatt",
            ],
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    module.compact_report(args.results_dir / "RESULTADOS.md", results, metadata)

    sample_csv = args.results_dir / "turbostat_amostras.csv"
    write_csv(sample_csv, samples)
    print(f"Parsed samples: {len(samples)}")
    print(f"Idle package power: {baseline_power:.6f} W")
    print(f"Updated: {args.results_dir / 'resultados.csv'}")
    print(f"Saved: {sample_csv}")


if __name__ == "__main__":
    main()
