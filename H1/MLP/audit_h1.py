#!/usr/bin/env python3
"""Audit actual H1 artifacts and consolidate HLS estimates (not board results)."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[1] / "MLP/hls4ml/mlp_iris_apfixed16_6_rf1_100mhz"

def read_json(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

original = read_json(ORIGINAL / "configuration_manifest.json")
original_report = read_json(ORIGINAL / "build_report.json")
reference = read_json(ROOT / "RF1/configuration_manifest.json")
checks = ["backend", "part", "board", "target_frequency_mhz", "clock_period_ns",
          "clock_uncertainty", "precision", "quantization_framework", "strategy",
          "io_type", "softmax", "versions"]
rows = []
packages = []
for rf in (1, 4, 8, 16, 32):
    build = ROOT / f"RF{rf}"
    assert read_json(build / "run_status.json")["status"] == "completed"
    manifest = read_json(build / "configuration_manifest.json")
    for key in checks:
        assert manifest[key] == original[key], (rf, key)
    assert manifest["reuse_factor"] == rf
    config = copy.deepcopy(manifest["hls_config"])
    config["Model"]["ReuseFactor"] = 1
    assert config == original["hls_config"], (rf, "hls_config")
    for filename, key in (("iris_mlp_clean.h5", "model_sha256"), ("iris_scaler.joblib", "scaler_sha256")):
        assert manifest[key] == digest(ROOT / filename) == reference[key]
    effective = read_json(build / "effective_layers.json")
    dense = [layer for layer in effective if layer["class"] in ("Dense", "VitisDense")]
    assert len(dense) == 3
    assert all(layer["reuse_factor"] == rf and layer["strategy"] == "latency" for layer in dense)
    assert all(layer["reuse_factor"] == rf for layer in effective)
    parameters = (build / "firmware/parameters.h").read_text()
    normalized = re.sub(r"(static const unsigned reuse_factor = )\d+;", r"\g<1>1;", parameters)
    assert normalized == (ORIGINAL / "firmware/parameters.h").read_text(), (rf, "parameters")
    for source in [ORIGINAL / "firmware/defines.h", *sorted((ORIGINAL / "firmware/weights").glob("*"))]:
        if source.is_file():
            assert digest(source) == digest(build / source.relative_to(ORIGINAL)), (rf, source.name)
    validation = read_json(build / "validation_summary.json")
    report = read_json(build / "build_report.json")
    assert validation["samples"] == 30 and validation["hls_correct"] == 29
    assert validation["class_agreement_keras_hls"] == 1.0
    assert report["CosimReport"]["Status"] == "Pass"
    assert len(report["CSimResults"]) == len(report["CosimResults"]) == 30
    assert report["CSimResults"] == report["CosimResults"] == original_report["CSimResults"]
    if rf == 1:
        assert report["CSynthesisReport"] == original_report["CSynthesisReport"]
    ip_dir = build / "mlp_iris_prj/solution1/impl/ip"
    package = ip_dir / "xilinx_com_hls_mlp_iris_1_0.zip"
    assert (ip_dir / "component.xml").is_file()
    rtl_files = list((ip_dir / "hdl/verilog").glob("*"))
    assert rtl_files
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        for packaged in rtl_files:
            if packaged.is_file():
                assert archive.read(packaged.relative_to(ip_dir).as_posix()) == packaged.read_bytes()
                generated = build / "mlp_iris_prj/solution1/syn/verilog" / packaged.name
                assert generated.is_file() and digest(generated) == digest(packaged), (rf, packaged.name)
    packages.append({"rf": rf, "path": package.relative_to(ROOT).as_posix(), "sha256": digest(package)})
    syn = report["CSynthesisReport"]
    rows.append({"rf_global": rf, "strategy": manifest["strategy"],
        "clock_ns": manifest["clock_period_ns"], "latency_cycles_min": int(syn["BestLatency"]),
        "latency_cycles_max": int(syn["WorstLatency"]),
        "latency_ns_max": int(syn["WorstLatency"]) * manifest["clock_period_ns"],
        "ii_min": int(syn["IntervalMin"]), "ii_max": int(syn["IntervalMax"]),
        "dsp_hls": int(syn["DSP"]), "lut_hls": int(syn["LUT"]), "ff_hls": int(syn["FF"]),
        "bram_18k_hls": int(syn["BRAM_18K"]), "uram_hls": int(syn["URAM"]),
        "estimated_clock_ns": float(syn["EstimatedClockPeriod"]),
        "accuracy": validation["hls_accuracy"], "agreement": validation["class_agreement_keras_hls"],
        "rtl_cosim": report["CosimReport"]["Status"]})
with (ROOT / "resultados_hls.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "audit_summary.json").write_text(json.dumps({"status": "passed", "only_experimental_variable": "global ReuseFactor", "original_reference": str(ORIGINAL), "checks": ["original configuration and tool versions", "model and scaler hashes", "actual per-layer RF and Latency strategy", "generated parameters differ only by RF", "identical types and weights", "30/30 C and RTL outputs match original", "RF1 synthesis reproduces original", "ZIP integrity and exported Verilog matches synthesis"], "results": rows, "packages": packages}, indent=2) + "\n")
(ROOT / "IP_SHA256SUMS.txt").write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in packages))
print(json.dumps(rows, indent=2))
print("Audit passed; resultados_hls.csv, audit_summary.json and IP_SHA256SUMS.txt written.")
