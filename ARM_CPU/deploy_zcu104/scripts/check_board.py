#!/usr/bin/env python3
"""Diagnóstico não destrutivo do ambiente ARM antes da campanha."""

import json
import os
import platform
import sys
import numpy as np

runtime = None
runtime_error = None
try:
    import tflite_runtime.interpreter as interpreter_module
    runtime = "tflite_runtime"
except Exception as exc:
    try:
        import tensorflow.lite as interpreter_module
        runtime = "tensorflow.lite"
    except Exception as fallback_exc:
        runtime_error = f"tflite_runtime: {exc}; tensorflow.lite: {fallback_exc}"

report = {
    "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
    "numpy": np.__version__, "cpu_count": os.cpu_count(),
    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "runtime": runtime, "runtime_error": runtime_error,
    "cpu_only_contract": "Scripts não importam VART e não carregam overlay/delegate DPU.",
}
print(json.dumps(report, indent=2, ensure_ascii=False))
raise SystemExit(0 if runtime else 1)
