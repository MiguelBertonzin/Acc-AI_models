#!/usr/bin/env python3
"""Audit LeNet H1 and consolidate evidence without rerunning synthesis."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from ip_audit import audit_ip
ROOT=Path(__file__).resolve().parent
BASELINE=ROOT.parents[1]/"LeNet/hls4ml/builds/cap64_q22_12_rate_balanced"
TOP="lenet_mnist_cap64_hls"
def read(path):
    return json.loads(path.read_text())
def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
def strip_rf(config):
    value=copy.deepcopy(config)
    value["Model"].pop("ReuseFactor",None)
    for layer in value["LayerName"].values():
        layer.pop("ReuseFactor",None)
    return value
expected={32:[25,30,32,30,28],64:[75,75,64,60,84],128:[150,150,128,120,168]}
layers=["conv1","conv2","dense1","dense2","output"]
ref_config=read(ROOT/"reference/hls_config.json")
ref_manifest=read(BASELINE/"build_manifest.json")
ref_data=np.load(BASELINE/"cpp_predictions_10000.npz")
ref_types=re.findall(r"typedef (ap_fixed<[^;]+) (\w+);",(BASELINE/"firmware/defines.h").read_text())
ref_source=(BASELINE/"firmware"/f"{TOP}.cpp").read_text()
rows=[]
packages=[]
all_logits=[]
for rf in (32,64,128):
    folder=ROOT/f"RF{rf}"
    assert read(folder/"run_status.json")["status"]=="completed"
    m=read(folder/"configuration_manifest.json")
    assert m["global_reuse_factor"]==rf
    assert strip_rf(m["hls_config"])==strip_rf(ref_config)
    assert all("ReuseFactor" not in x for x in m["hls_config"]["LayerName"].values())
    assert m["design"]==read(ROOT/"reference/design.json")
    assert m["software"]==ref_manifest["software"]
    assert m["model_sha256"]==sha(ROOT/"lenet_mnist_final.h5")==ref_manifest["model_sha256"]
    assert m["data_sha256"]==sha(ROOT/"data/mnist_test_uint8.npz")
    assert [m["effective_reuse"][name] for name in layers]==expected[rf]
    graph={item["name"]:item for item in read(folder/"effective_layers.json")}
    assert all(graph[name]["reuse_factor"]==m["effective_reuse"][name] and graph[name]["strategy"]=="resource" for name in layers)
    assert re.findall(r"typedef (ap_fixed<[^;]+) (\w+);",(folder/"firmware/defines.h").read_text())==ref_types
    assert (folder/"firmware"/f"{TOP}.cpp").read_text()==ref_source
    assert m["fifo_depths"]==read(ROOT/"RF32/fifo_depths.json") and not m["fifo_optimization"]
    v=read(folder/"cpp_validation.json")
    assert v["status"]=="passed" and v["samples"]==10000
    p=np.load(folder/"cpp_predictions_10000.npz")
    assert np.array_equal(p["labels"],ref_data["labels"])
    assert np.array_equal(p["keras_predictions"],ref_data["keras_predictions"])
    all_logits.append(p["hls_logits"].copy())
    report=read(folder/"build_report.json")
    assert report["CosimReport"]["Status"]=="Pass"
    indices=m["rtl_indices"]
    assert indices==read(ROOT/"RF32/rtl_test_subset.json")["indices"]
    csim=np.asarray(report["CSimResults"],float)
    rtl=np.asarray(report["CosimResults"],float)
    assert csim.shape==rtl.shape==(10,10) and np.array_equal(csim,rtl)
    assert np.allclose(csim,p["hls_logits"][indices],atol=1e-4,rtol=0)
    export=read(folder/"export_audit.json")
    assert export["status"]=="passed"
    package=folder/export["package"]
    assert sha(package)==export["sha256"]
    checked=audit_ip(folder,TOP)
    assert checked["sha256"]==export["sha256"] and checked["synthesis_datapath_matches"]
    packages.append({"rf":rf,"path":package.relative_to(ROOT).as_posix(),"sha256":sha(package)})
    syn=report["CSynthesisReport"]
    row={"global_rf_requested":rf,**{name+"_rf":m["effective_reuse"][name] for name in layers},
        "clock_ns":m["design"]["clock_period_ns"],"estimated_clock_ns":syn["EstimatedClockPeriod"],
        "hls_latency_min_cycles":syn["BestLatency"],"hls_latency_max_cycles":syn["WorstLatency"],
        "hls_ii_min_cycles":syn["IntervalMin"],"hls_ii_max_cycles":syn["IntervalMax"],
        **{name+"_hls":syn[name] for name in ("DSP","LUT","FF","BRAM_18K","URAM")},
        "cpp_samples":v["samples"],"keras_accuracy_percent":v["keras_accuracy_percent"],
        "hls_accuracy_percent":v["hls_accuracy_percent"],"keras_hls_agreement_percent":v["keras_hls_argmax_agreement_percent"],
        "logit_mae":v["logit_mean_absolute_error"],"logit_max_error":v["logit_max_absolute_error"],
        "rtl_samples":10,"rtl_status":report["CosimReport"]["Status"],
        "rtl_latency_min_cycles":report["CosimReport"].get("LatencyMin"),
        "rtl_latency_max_cycles":report["CosimReport"].get("LatencyMax"),
        "rtl_interval_min_cycles":report["CosimReport"].get("IntervalMin"),
        "rtl_interval_max_cycles":report["CosimReport"].get("IntervalMax")}
    rows.append(row)
with (ROOT/"resultados_hls.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summary={"status":"passed","results":rows,"packages":packages,
    "all_rf_cpp_logits_equal":all(np.array_equal(all_logits[0],p) for p in all_logits[1:]),
    "cpp_logits_equal_baseline":[bool(np.array_equal(p,ref_data["hls_logits"])) for p in all_logits],
    "checks":["same configuration except global RF and removed layer RF overrides", "same model, data and software versions",
    "expected effective weighted-layer RFs and Resource strategy", "same fixed-point types and top C++ including FIFOs",
    "10000-image numerical validation passed", "10-image RTL agrees with C simulation and C++", "ZIP and packaged RTL integrity"]}
(ROOT/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(ROOT/"IP_SHA256SUMS.txt").write_text("".join(f"{p['sha256']}  {p['path']}\n" for p in packages))
print(json.dumps(summary,indent=2))
