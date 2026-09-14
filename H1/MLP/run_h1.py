#!/usr/bin/env python3
"""Build the five H1 MLP IPs using the original Latency flow."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
for rf in (1, 4, 8, 16, 32):
    destination = ROOT / f"RF{rf}"
    destination.mkdir(exist_ok=True)
    status_path = destination / "run_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        print(f"RF{rf}: already completed; preserving artifacts", flush=True)
        continue
    if any(destination.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty build: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f"h1_mlp_rf{rf}_"))
    command = [sys.executable, str(ROOT / "generate_mlp_h1.py"),
               "--reuse-factor", str(rf), "--output-dir", str(staging), "--cosim"]
    state = {"reuse_factor": rf, "status": "running", "staging": str(staging),
             "command": command, "started_at_unix": time.time()}
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"RF{rf}: starting; log={staging / 'run.log'}", flush=True)
    with (staging / "run.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    shutil.copytree(staging, destination, dirs_exist_ok=True)
    state.update(status="completed" if result.returncode == 0 else "failed",
                 returncode=result.returncode, finished_at_unix=time.time())
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"RF{rf}: {state['status']}; published to {destination}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
