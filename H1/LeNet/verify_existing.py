"""Recheck an exported RF after a post-export audit failure, without synthesis."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ip_audit import audit_ip
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(); parser.add_argument("--rf",type=int,choices=(32,64,128),required=True)
rf=parser.parse_args().rf
folder=root/f"RF{rf}"
report=json.loads((folder/"build_report.json").read_text())
manifest=json.loads((folder/"configuration_manifest.json").read_text())
validation=json.loads((folder/"cpp_validation.json").read_text())
pred=np.load(folder/"cpp_predictions_10000.npz")
assert validation["status"]=="passed" and validation["samples"]==10000
assert report["CosimReport"]["Status"]=="Pass"
a=np.asarray(report["CSimResults"],float); b=np.asarray(report["CosimResults"],float)
assert a.shape==b.shape==(10,10) and np.array_equal(a,b)
assert np.allclose(a,pred["hls_logits"][manifest["rtl_indices"]],atol=1e-4,rtol=0)
result=audit_ip(folder,"lenet_mnist_cap64_hls")
result.update(rtl_samples=10,cosim_matches_csim=True,csim_matches_cpp=True)
(folder/"export_audit.json").write_text(json.dumps(result,indent=2)+"\n")
state_path=folder/"run_status.json"
state=json.loads(state_path.read_text())
if state["status"]=="failed":
    history=folder/"run_status.initial_failure.json"
    if not history.exists(): history.write_bytes(state_path.read_bytes())
    state.update(status="completed",original_returncode=state["returncode"],returncode=0,
        recovered_at_unix=time.time(),recovery="Post-export audit corrected to recognize simulation-only deadlock instrumentation. Existing IP verified; synthesis not rerun.")
    state_path.write_text(json.dumps(state,indent=2)+"\n")
print(json.dumps({"rf":rf,"status":"passed","additional_simulation_files":result["simulation_auxiliary_files"],"normalized_top":result["normalized_top_files"],"sha256":result["sha256"]},indent=2))
