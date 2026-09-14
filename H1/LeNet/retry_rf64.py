#!/usr/bin/env python3
"""Retry only RF64 RTL cosimulation/export, then continue the H1 queue."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/"RF64"
STAGE=Path("/tmp/h1_lenet_rf64_anv2rv3c")
TOP="lenet_mnist_cap64_hls"
TOOL="/opt/Xilinx/Vitis/2024.2/bin/vitis-run"
def write(path,data):
    path.write_text(json.dumps(data,indent=2)+"\n")
def hashes():
    directory=STAGE/f"{TOP}_prj/solution1/syn"
    return {str(p.relative_to(STAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix in {".v",".vh",".dat",".rpt",".xml"}}
state_path=FOLDER/"run_status.json"
old=json.loads(state_path.read_text())
assert old["status"]=="failed"
write(FOLDER/"run_status.before_cosim_retry.json",old)
before=hashes()
assert before
write(FOLDER/"synthesis_hashes_before_retry.json",before)
tcl=STAGE/"retry_cosim_export.tcl"
tcl.write_text('source project.tcl\n'
    'open_project ${project_name}_prj\n'
    'open_solution solution1\n'
    'add_files -tb ${project_name}_test.cpp -cflags "-std=c++0x -DRTL_SIM"\n'
    'cosim_design -rtl verilog -trace_level none\n'
    'set cfile [open tb_data/csim_results.log r]\n'
    'set cresults [read $cfile]\nclose $cfile\n'
    'set rfile [open tb_data/rtl_cosim_results.log r]\n'
    'set rresults [read $rfile]\nclose $rfile\n'
    'if {$cresults ne $rresults} {error "C/RTL output mismatch"}\n'
    'export_design -format ip_catalog -version $version\n'
    'exit\n')
state=dict(old)
state.update(status="running",phase="cosim_export_retry_without_synthesis",retry_started_at_unix=time.time(),
             retry_command=[TOOL,"--tcl",str(tcl),"--mode","hls"])
write(state_path,state)
print(f"RF64: retrying only RTL cosimulation/export; log={STAGE / 'retry_cosim_export.log'}",flush=True)
error=None
try:
    with (STAGE/"retry_cosim_export.log").open("w") as log:
        process=subprocess.Popen(state["retry_command"],cwd=STAGE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            raise RuntimeError("Cosimulation/export timeout after 900s; process group stopped, synthesis preserved")
    assert code==0,f"Retry tool failed with return code {code}"
    assert hashes()==before,"Synthesis outputs changed during retry"
    from hls4ml.report.vivado_report import parse_vivado_report
    report=parse_vivado_report(str(STAGE))
    write(STAGE/"build_report.json",report)
    assert report["CosimReport"]["Status"]=="Pass"
    manifest=json.loads((STAGE/"configuration_manifest.json").read_text())
    manifest["simulation_retry"]={"rtl_samples":10,"mode":"direct cosim_design, Verilog, trace_level none",
        "synthesis_rerun":False,"synthesis_hashes_unchanged":True}
    write(STAGE/"configuration_manifest.json",manifest)
except Exception as exc:
    error=str(exc)
finally:
    shutil.copytree(STAGE,FOLDER,dirs_exist_ok=True)
    state.update(status="failed" if error else "verifying",retry_finished_at_unix=time.time(),retry_error=error)
    write(state_path,state)
if error:
    raise RuntimeError(error)
# Functional and package verification must succeed before marking complete.
subprocess.run([sys.executable,str(ROOT/"verify_existing.py"),"--rf","64"],check=True)
state.update(status="completed",returncode=0,phase="completed_after_cosim_retry",synthesis_rerun=False,
             synthesis_hashes_unchanged=True)
write(state_path,state)
print("RF64: retry passed, IP verified, synthesis unchanged. Starting remaining RFs.",flush=True)
subprocess.run([sys.executable,str(ROOT/"run_h1.py")],check=True)
subprocess.run([sys.executable,str(ROOT/"audit_h1.py")],check=True)
subprocess.run([sys.executable,str(ROOT/"write_report.py")],check=True)
print("LeNet H1: all RFs completed and final report written.",flush=True)
