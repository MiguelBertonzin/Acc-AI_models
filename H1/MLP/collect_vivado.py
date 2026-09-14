#!/usr/bin/env python3
"""Append Vivado OOC post-synthesis utilization and timing to the MLP H1 table."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RFS = (1, 4, 8, 16, 32)


def table_row(text, label):
    pattern = (r"^\|\s*" + re.escape(label) +
               r"\s*\|\s*([0-9.]+)\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|"
               r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
    match = re.search(pattern, text, re.M)
    if not match:
        raise RuntimeError(f"Vivado utilization row missing: {label}")
    used, available, percent = map(float, match.groups())
    used = int(used) if used.is_integer() else used
    available = int(available) if available.is_integer() else available
    return used, available, percent


def parse_rf(rf):
    report_dir = ROOT / f"RF{rf}" / "vivado_ooc"
    if (report_dir / "status.txt").read_text().splitlines()[0] != "status=completed":
        raise RuntimeError(f"RF{rf}: Vivado run did not complete")
    utilization = (report_dir / "utilization.rpt").read_text()
    timing = (report_dir / "timing_summary.rpt").read_text()
    values = {}
    labels = (
        ("clb_luts", "CLB LUTs*"), ("lut_logic", "LUT as Logic"),
        ("lut_memory", "LUT as Memory"), ("clb_registers", "CLB Registers"),
        ("bram_tiles", "Block RAM Tile"), ("ramb36", "RAMB36/FIFO*"),
        ("ramb18", "RAMB18"), ("uram", "URAM"), ("dsps", "DSPs"),
    )
    for key, label in labels:
        used, available, percent = table_row(utilization, label)
        values[f"vivado_{key}"] = used
        values[f"vivado_{key}_available"] = available
        values[f"vivado_{key}_percent"] = percent
    timing_match = re.search(
        r"WNS\(ns\).*?\n\s*-+.*?\n\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)",
        timing, re.S)
    clock_match = re.search(
        r"^ap_clk\s+\{[^}]+\}\s+([0-9.]+)\s+([0-9.]+)", timing, re.M)
    if not timing_match or not clock_match:
        raise RuntimeError(f"RF{rf}: constrained timing result missing")
    bram18eq = 2 * values["vivado_ramb36"] + values["vivado_ramb18"]
    clock_ns = float(clock_match.group(1))
    wns_ns = float(timing_match.group(1))
    values.update({
        "vivado_bram_18k_equivalent": bram18eq,
        "vivado_bram_18k_equivalent_percent": round(100 * bram18eq / 624, 2),
        "vivado_clock_period_ns": clock_ns,
        "vivado_clock_frequency_mhz": float(clock_match.group(2)),
        "vivado_wns_ns": wns_ns,
        "vivado_tns_ns": float(timing_match.group(2)),
        "vivado_post_synth_critical_path_ns": round(clock_ns - wns_ns, 3),
        "vivado_post_synth_fmax_mhz_theoretical": round(1000 / (clock_ns - wns_ns), 3),
        "vivado_timing_met": "All user specified timing constraints are met." in timing,
        "vivado_stage": "optimized_out_of_context_post_synthesis",
    })
    values["vivado_resource_fit"] = max(
        values["vivado_clb_luts_percent"], values["vivado_clb_registers_percent"],
        values["vivado_bram_tiles_percent"], values["vivado_uram_percent"],
        values["vivado_dsps_percent"]) <= 100
    return values


csv_path = ROOT / "resultados_hls.csv"
with csv_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if [int(row["rf_global"]) for row in rows] != list(RFS):
    raise RuntimeError("Unexpected RF rows in resultados_hls.csv")

for row in rows:
    rf = int(row["rf_global"])
    row.update(parse_rf(rf))
    clock_ns = float(row["clock_ns"])
    latency_cycles = float(row["latency_cycles_max"])
    ii_cycles = float(row["ii_max"])
    row["latency_us_at_target_clock"] = round(latency_cycles * clock_ns / 1000, 6)
    row["theoretical_inferences_per_second"] = round(1e9 / (ii_cycles * clock_ns), 3)
    comparisons = (
        ("dsp", "vivado_dsps"), ("lut", "vivado_clb_luts"),
        ("ff", "vivado_clb_registers"), ("bram_18k", "vivado_bram_18k_equivalent"),
        ("uram", "vivado_uram"),
    )
    for hls_name, vivado_name in comparisons:
        hls_value = float(row[f"{hls_name}_hls"])
        vivado_value = float(row[vivado_name])
        row[f"{hls_name}_vivado_minus_hls"] = vivado_value - hls_value
        row[f"{hls_name}_vivado_vs_hls_percent"] = round(100 * vivado_value / hls_value, 2) if hls_value else ""

with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {f"RF{rf}": parse_rf(rf) for rf in RFS}
(ROOT / "vivado_ooc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
