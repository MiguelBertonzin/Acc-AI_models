#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, json, math
import numpy as np

def mean_sd_ci95(vals):
    a=np.asarray(vals,float); n=len(a)
    m=float(np.mean(a))
    sd=float(np.std(a,ddof=1)) if n>1 else 0.0
    # n=5 => df=4; t(0.975)=2.7764451051977987
    t=2.7764451051977987 if n==5 else 1.959963984540054
    se=sd/math.sqrt(n) if n>0 else float("nan")
    return m,sd,100*sd/m if m else float("nan"),m-t*se,m+t*se

def loadj(p): return json.load(open(p,encoding="utf-8"))

ap=argparse.ArgumentParser()
ap.add_argument("--project-root",type=Path,default=Path("."))
ap.add_argument("--output-dir",type=Path,default=Path("results/final_summary"))
a=ap.parse_args()
root=a.project_root
out=a.output_dir
out.mkdir(parents=True,exist_ok=True)

rows=[]
serial=root/"results/serial_official_20260908"
for n in [30000,100000]:
    reps=[]
    for r in range(1,6):
        p=serial/str(n)/f"rep{r:02d}"
        if not (p/"metrics_summary.json").exists(): continue
        m=loadj(p/"metrics_summary.json"); w=loadj(p/"power_summary.json")
        row={
            "campaign":"serial","inferences":n,"rep":r,
            "accelerator_mean_ms":m["accelerator"]["mean_ms"],
            "device_call_mean_ms":m["device_call"]["mean_ms"],
            "end_to_end_mean_ms":m["application_end_to_end"]["mean_ms"],
            "throughput_inf_s":m["throughput_effective_inf_s"],
            "idle_power_w":w["idle_mean_power_w"],
            "active_power_w":w["active_mean_power_w"],
            "dynamic_power_w":w["dynamic_mean_power_w"],
            "total_mj_inf":w["total_mj_per_inference"],
            "dynamic_mj_inf":w["dynamic_mj_per_inference"],
        }
        rows.append(row); reps.append(row)

    if len(reps)==5:
        agg={"campaign":"serial_aggregate","inferences":n,"rep":"mean_of_5"}
        for k in [
            "accelerator_mean_ms","device_call_mean_ms","end_to_end_mean_ms",
            "throughput_inf_s","idle_power_w","active_power_w","dynamic_power_w",
            "total_mj_inf","dynamic_mj_inf"
        ]:
            m,sd,cv,lo,hi=mean_sd_ci95([x[k] for x in reps])
            agg[k+"_mean"]=m; agg[k+"_sd"]=sd; agg[k+"_cv_percent"]=cv
            agg[k+"_ci95_lower"]=lo; agg[k+"_ci95_upper"]=hi
        rows.append(agg)

robust_dirs=sorted(p for p in (root/"results").glob("robust_vitis_ai_*") if p.is_dir())
if robust_dirs:
    rp=robust_dirs[-1]
    s=rp/"all_configs_summary.json"
    if s.exists():
        for x in loadj(s):
            m=x["metrics"]; w=x["power"]
            rows.append({
                "campaign":"robust_12x10k",
                "scenario":x["scenario"],
                "threads_runners":x["threads_runners"],
                "inferences":m["inferences"],
                "latency_mean_ms":m["latency"]["mean_ms"],
                "latency_median_ms":m["latency"]["median_ms"],
                "latency_p90_ms":m["latency"]["p90_ms"],
                "latency_p95_ms":m["latency"]["p95_ms"],
                "latency_p99_ms":m["latency"]["p99_ms"],
                "latency_min_ms":m["latency"]["minimum_ms"],
                "latency_max_ms":m["latency"]["maximum_ms"],
                "latency_std_ms":m["latency"]["std_sample_ms"],
                "latency_cv_percent":m["latency"]["cv_percent"],
                "latency_ci95_lower_ms":m["latency"]["mean_ci95_lower_ms"],
                "latency_ci95_upper_ms":m["latency"]["mean_ci95_upper_ms"],
                "throughput_inf_s":m["throughput_effective_inf_s"],
                "idle_power_w":w["idle_mean_power_w"],
                "active_power_w":w["active_mean_power_w"],
                "dynamic_power_w":w["dynamic_mean_power_w"],
                "idle_mj_inf":w["idle_mj_per_inference"],
                "total_mj_inf":w["total_mj_per_inference"],
                "dynamic_mj_inf":w["dynamic_mj_per_inference"],
            })

# JSON
(out/"summary_all.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")

# CSV union of all keys
keys=[]
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with (out/"summary_all.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=keys)
    w.writeheader(); w.writerows(rows)

print(json.dumps({"rows":len(rows),"json":str(out/"summary_all.json"),"csv":str(out/"summary_all.csv")},indent=2))
