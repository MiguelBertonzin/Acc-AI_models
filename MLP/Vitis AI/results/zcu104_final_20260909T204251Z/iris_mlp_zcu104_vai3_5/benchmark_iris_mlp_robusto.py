#!/usr/bin/env python3
"""
MLP/Iris — benchmark robusto Vitis AI/ZCU104 equivalente à campanha LeNet.

Matriz:
  inference_only : T1,T2,T3,T4
  end_to_end     : T1,T2,T3,T4
  saturated      : T1,T2,T3,T4

Cada configuração:
  - exatamente 10.000 inferências medidas
  - batch 1
  - 100 warm-ups por runner
  - 10 blocos x 1.000 conclusões
  - baseline idle 5 s após warm-up
  - telemetria INA226 nominal 10 ms
  - nenhum outlier removido

IMPORTANTE:
  T1..T4 = número de VART runners/threads concorrentes.
  A placa continua tendo 2 cores físicos DPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

EXPECTED_XMODEL_SHA256 = "47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce"
EXPECTED_TEST_SHA256 = "3972a24323d73613dfca346b56333a5843d45e399cdf80d05ac04b1257ed86e1"
EXPECTED_REF_SHA256 = "b0ab40e38bd10bf685badc8552e2a66030c247447df626ab05a12e796364ba0e"

EXPECTED_INPUT = (1, 4)
EXPECTED_OUTPUT = (1, 3)
EXPECTED_INPUT_FIX = 5
EXPECTED_OUTPUT_FIX = 3

POWER = Path("/sys/class/hwmon/hwmon0/power1_input")
CURRENT = Path("/sys/class/hwmon/hwmon0/curr1_input")
VOLTAGE = Path("/sys/class/hwmon/hwmon0/in2_input")
AMS = Path("/sys/bus/iio/devices/iio:device0")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def run_command(cmd, timeout=60):
    try:
        p = subprocess.run(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False, timeout=timeout
        )
        return {"command": list(cmd), "returncode": p.returncode, "output": p.stdout}
    except Exception as e:
        return {"command": list(cmd), "returncode": None, "output": repr(e)}


def read_float(path: Path):
    try:
        return float(path.read_text().strip())
    except Exception:
        return None


def temp_c(prefix):
    raw = read_float(AMS / f"{prefix}_raw")
    off = read_float(AMS / f"{prefix}_offset")
    scale = read_float(AMS / f"{prefix}_scale")
    if raw is None or off is None or scale is None:
        return None
    return (raw + off) * scale / 1000.0


def temperatures():
    return {
        "ps_temp_c": temp_c("in_temp0_ps_temp"),
        "remote_temp_c": temp_c("in_temp1_remote_temp"),
        "pl_temp_c": temp_c("in_temp2_pl_temp"),
    }


def only_dpu_subgraph(graph):
    children = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [
        s for s in children
        if s.has_attr("device") and str(s.get_attr("device")).upper() == "DPU"
    ]
    if len(dpu) != 1:
        raise RuntimeError(f"Esperado 1 subgrafo DPU; encontrados {len(dpu)}")
    return dpu[0]


def fix(t):
    if not t.has_attr("fix_point"):
        raise RuntimeError(f"{t.name} sem fix_point")
    return int(t.get_attr("fix_point"))


def dims(t):
    return tuple(int(v) for v in t.dims)


class RunnerCtx:
    def __init__(self, xmodel: Path):
        import xir
        import vart
        self.graph = xir.Graph.deserialize(str(xmodel))
        self.runner = vart.Runner.create_runner(only_dpu_subgraph(self.graph), "run")
        self.it = self.runner.get_input_tensors()[0]
        self.ot = self.runner.get_output_tensors()[0]
        self.input_fix = fix(self.it)
        self.output_fix = fix(self.ot)
        if dims(self.it) != EXPECTED_INPUT or dims(self.ot) != EXPECTED_OUTPUT:
            raise RuntimeError(f"Shapes: {dims(self.it)} -> {dims(self.ot)}")
        if (self.input_fix, self.output_fix) != (EXPECTED_INPUT_FIX, EXPECTED_OUTPUT_FIX):
            raise RuntimeError(f"fix_point: {self.input_fix}, {self.output_fix}")
        self.ib = np.empty(dims(self.it), dtype=np.int8)
        self.ob = np.empty(dims(self.ot), dtype=np.int8)

    def quantize(self, normalized):
        return np.clip(
            np.rint(np.asarray(normalized, dtype=np.float32) * (2 ** self.input_fix)),
            -128, 127
        ).astype(np.int8).reshape(EXPECTED_INPUT)

    def decode(self, qout):
        return np.asarray(qout, dtype=np.float32).reshape(-1) / (2 ** self.output_fix)

    def execute_only(self):
        jid = self.runner.execute_async([self.ib], [self.ob])
        self.runner.wait(jid)


def softmax(logits):
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def wilson(correct, n, z=1.959963984540054):
    p = correct / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return [center-half, center+half]


def stat_summary(values, block_means):
    a = np.asarray(values, dtype=np.float64)
    bm = np.asarray(block_means, dtype=np.float64)
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise RuntimeError("latências inválidas")
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1))
    # Protocolo congelado: 10 blocos => df=9.
    t975_df9 = 2.2621571627409915
    se = float(np.std(bm, ddof=1) / math.sqrt(len(bm)))
    bmean = float(np.mean(bm))
    return {
        "count": int(a.size),
        "mean_ms": mean,
        "median_ms": float(np.median(a)),
        "std_sample_ms": sd,
        "cv_percent": float(100 * sd / mean),
        "minimum_ms": float(np.min(a)),
        "maximum_ms": float(np.max(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ci95_lower_ms": bmean - t975_df9 * se,
        "mean_ci95_upper_ms": bmean + t975_df9 * se,
        "ci_unit": "means of 10 blocks of 1000 completed inferences",
        "ci_blocks": int(len(bm)),
    }


class Telemetry:
    def __init__(self, interval_s):
        self.interval_s = float(interval_s)
        self.phase = "setup"
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def _read(self, phase=None):
        t0 = time.perf_counter_ns()
        p = read_float(POWER)
        c = read_float(CURRENT)
        v = read_float(VOLTAGE)
        row = {
            "perf_counter_ns": t0,
            "unix_s": time.time(),
            "phase": self.phase if phase is None else phase,
            "board_input_power_w": None if p is None else p/1e6,
            "board_input_current_a": None if c is None else c/1000.0,
            "board_input_voltage_v": None if v is None else v/1000.0,
            "read_duration_ns": time.perf_counter_ns() - t0,
        }
        with self.lock:
            self.rows.append(row)

    def sample_now(self, phase=None):
        self._read(phase)

    def loop(self):
        nxt = time.perf_counter()
        while not self.stop_event.is_set():
            self._read()
            nxt += self.interval_s
            delay = nxt - time.perf_counter()
            if delay > 0:
                self.stop_event.wait(delay)
            else:
                nxt = time.perf_counter()

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)


def integrate_exact(rows, key, start_ns, end_ns):
    r = [x for x in rows if x.get(key) is not None and math.isfinite(float(x[key]))]
    if len(r) < 2:
        return None
    r.sort(key=lambda x: x["perf_counter_ns"])
    t = np.asarray([x["perf_counter_ns"] for x in r], dtype=np.float64)
    y = np.asarray([x[key] for x in r], dtype=np.float64)
    if t[0] > start_ns or t[-1] < end_ns:
        return {"status": "insufficient_bracketing"}
    inside = (t >= start_ns) & (t <= end_ns)
    ti, yi = t[inside], y[inside]
    ys = float(np.interp(float(start_ns), t, y))
    ye = float(np.interp(float(end_ns), t, y))
    tx = np.concatenate(([float(start_ns)], ti, [float(end_ns)]))
    yx = np.concatenate(([ys], yi, [ye]))
    order = np.argsort(tx)
    tx, yx = tx[order], yx[order]
    keep = np.concatenate(([True], np.diff(tx) > 0))
    tx, yx = tx[keep], yx[keep]
    rel_s = (tx - start_ns) / 1e9
    integ = float(np.trapz(yx, rel_s))
    duration_s = (end_ns-start_ns)/1e9
    gaps = np.diff(tx)/1e6
    return {
        "status": "measured",
        "bracketed": True,
        "samples_inside": int(np.sum(inside)),
        "duration_s": duration_s,
        "mean": integ/duration_s,
        "median_samples_inside": float(np.median(yi)) if yi.size else None,
        "std_sample_inside": float(np.std(yi, ddof=1)) if yi.size > 1 else 0.0,
        "minimum_samples_inside": float(np.min(yi)) if yi.size else None,
        "maximum_samples_inside": float(np.max(yi)) if yi.size else None,
        "max_sample_gap_ms": float(np.max(gaps)) if gaps.size else None,
        "integral": integ,
    }


def deterministic_order(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        out.extend(rng.permutation(30).tolist())
    return np.asarray(out[:n], dtype=np.int16)


def validate_unique30(xmodel, normalized, labels, ref):
    r = RunnerCtx(xmodel)
    preds = []
    qouts = []
    for x in normalized:
        q = r.quantize(x)
        r.ib[...] = q
        r.execute_only()
        qo = r.ob.copy()
        qouts.append(qo.reshape(-1))
        preds.append(int(np.argmax(r.decode(qo))))
    preds = np.asarray(preds, dtype=np.int64)
    correct = int(np.sum(preds == labels))
    agreement = int(np.sum(preds == ref))
    ci = wilson(correct, 30)
    return {
        "samples_unique": 30,
        "correct": correct,
        "accuracy": correct/30,
        "accuracy_percent": 100*correct/30,
        "wilson95": ci,
        "wilson95_percent": [100*ci[0], 100*ci[1]],
        "reference_agreement": agreement,
        "reference_agreement_percent": 100*agreement/30,
        "divergences": int(np.sum(preds != ref)),
        "passed": bool(correct == 29 and agreement == 30),
    }


def warmup_runner(r, scenario, normalized, qinputs, count, worker_id):
    for i in range(count):
        idx = (i + worker_id) % 30
        if scenario == "end_to_end":
            q = r.quantize(normalized[idx])
            r.ib[...] = q
            r.execute_only()
            qo = r.ob.copy()
            _ = int(np.argmax(softmax(r.decode(qo))))
        else:
            r.ib[...] = qinputs[idx]
            r.execute_only()


def run_config(
    scenario, threads, xmodel, normalized, ref, qinputs, order_idx,
    outdir, warmup, baseline_s, telemetry_ms, dpu_query, script_sha
):
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Recuso sobrescrever {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(order_idx)
    if n != 10000:
        raise RuntimeError("protocolo robusto exige exatamente 10000")
    blocks = 10
    block_size = 1000

    runners = [RunnerCtx(xmodel) for _ in range(threads)]

    # Saturated: um input fixo por runner, preparado uma única vez.
    if scenario == "saturated":
        for wid, r in enumerate(runners):
            r.ib[...] = qinputs[wid % 30]

    temp_pre = temperatures()
    for wid, r in enumerate(runners):
        warmup_runner(r, scenario, normalized, qinputs, warmup, wid)
    temp_warm = temperatures()

    tel = Telemetry(telemetry_ms/1000.0)
    tel.start()

    tel.phase = "baseline"
    tel.sample_now("baseline_boundary_before")
    b0 = time.perf_counter_ns()
    time.sleep(baseline_s)
    b1 = time.perf_counter_ns()
    tel.sample_now("baseline_boundary_after")
    base_power = integrate_exact(tel.rows, "board_input_power_w", b0, b1)
    if not base_power or base_power.get("status") != "measured":
        tel.stop()
        raise RuntimeError(f"baseline inválido: {base_power}")
    idle_w = float(base_power["mean"])

    lat = np.empty(n, dtype=np.float64)
    preds = np.full(n, -1, dtype=np.int16)
    block_rows = []
    errors = []
    err_lock = threading.Lock()

    ready = threading.Barrier(threads + 1)
    done_barrier = threading.Barrier(threads + 1)
    go_events = [threading.Event() for _ in range(blocks)]

    # Posições determinísticas por worker dentro de cada bloco.
    work = []
    for wid in range(threads):
        per_block = []
        for b in range(blocks):
            start = b*block_size
            stop = start+block_size
            per_block.append(list(range(start+wid, stop, threads)))
        work.append(per_block)

    def worker(wid):
        r = runners[wid]
        for b in range(blocks):
            ready.wait()
            go_events[b].wait()
            try:
                for pos in work[wid][b]:
                    idx = int(order_idx[pos])
                    if scenario == "inference_only":
                        # input copy fora da latência; pós-processamento também fora.
                        r.ib[...] = qinputs[idx]
                        t0 = time.perf_counter_ns()
                        jid = r.runner.execute_async([r.ib], [r.ob])
                        r.runner.wait(jid)
                        t1 = time.perf_counter_ns()
                        lat[pos] = (t1-t0)/1e6
                        qo = r.ob.copy()
                        preds[pos] = int(np.argmax(r.decode(qo)))
                    elif scenario == "end_to_end":
                        x = normalized[idx]
                        t0 = time.perf_counter_ns()
                        q = r.quantize(x)
                        r.ib[...] = q
                        r.execute_only()
                        qo = r.ob.copy()
                        logits = r.decode(qo)
                        probs = softmax(logits)
                        preds[pos] = int(np.argmax(probs))
                        t1 = time.perf_counter_ns()
                        lat[pos] = (t1-t0)/1e6
                    elif scenario == "saturated":
                        t0 = time.perf_counter_ns()
                        jid = r.runner.execute_async([r.ib], [r.ob])
                        r.runner.wait(jid)
                        t1 = time.perf_counter_ns()
                        lat[pos] = (t1-t0)/1e6
                    else:
                        raise RuntimeError(scenario)
            except Exception as e:
                with err_lock:
                    errors.append(f"worker {wid} block {b}: {repr(e)}")
            done_barrier.wait()

    worker_threads = [threading.Thread(target=worker, args=(wid,), daemon=True) for wid in range(threads)]
    for t in worker_threads:
        t.start()

    print(
        f"[{scenario} T{threads}] benchmark 10000 iniciado; "
        "sem prints dentro dos blocos medidos",
        flush=True
    )
    tel.phase = "benchmark"
    tel.sample_now("benchmark_boundary_before")
    benchmark_start_ns = time.perf_counter_ns()

    for b in range(blocks):
        ready.wait()
        block_start_ns = time.perf_counter_ns()
        go_events[b].set()
        done_barrier.wait()
        block_end_ns = time.perf_counter_ns()
        br = {
            "block": b,
            "samples": block_size,
            "duration_s": (block_end_ns-block_start_ns)/1e9,
            "throughput_inf_s": block_size/((block_end_ns-block_start_ns)/1e9),
        }
        block_rows.append(br)

    benchmark_end_ns = time.perf_counter_ns()
    tel.sample_now("benchmark_boundary_after")
    tel.phase = "post"
    time.sleep(2*telemetry_ms/1000.0)
    tel.stop()

    for t in worker_threads:
        t.join(timeout=5)

    if errors:
        raise RuntimeError("; ".join(errors))
    if not np.all(np.isfinite(lat)) or np.any(lat <= 0):
        raise RuntimeError("latências inválidas")

    if scenario != "saturated":
        expected = ref[order_idx.astype(np.int64)]
        divergent = np.flatnonzero(preds.astype(np.int64) != expected)
        determinism = {
            "checked": True,
            "passed": bool(divergent.size == 0),
            "divergences": int(divergent.size),
            "divergent_positions_first100": divergent[:100].tolist(),
        }
        if divergent.size:
            raise RuntimeError(f"{scenario} divergências={divergent.size}")
    else:
        divergent = np.asarray([], dtype=np.int64)
        determinism = {
            "checked": False,
            "reason": "saturated is performance-only; no postprocessing/prediction in measured loop"
        }

    block_means = [
        float(np.mean(lat[b*block_size:(b+1)*block_size]))
        for b in range(blocks)
    ]
    wall_s = (benchmark_end_ns-benchmark_start_ns)/1e9

    metrics = {
        "scenario": scenario,
        "threads_runners": threads,
        "physical_dpu_cores": 2,
        "inferences": n,
        "blocks": blocks,
        "samples_per_block": block_size,
        "benchmark_wall_s": wall_s,
        "throughput_effective_inf_s": n/wall_s,
        "block_throughput_arithmetic_mean_inf_s": float(np.mean([x["throughput_inf_s"] for x in block_rows])),
        "latency": stat_summary(lat, block_means),
        "determinism": determinism,
    }

    active_power = integrate_exact(tel.rows, "board_input_power_w", benchmark_start_ns, benchmark_end_ns)
    active_current = integrate_exact(tel.rows, "board_input_current_a", benchmark_start_ns, benchmark_end_ns)
    active_voltage = integrate_exact(tel.rows, "board_input_voltage_v", benchmark_start_ns, benchmark_end_ns)
    if not active_power or active_power.get("status") != "measured":
        raise RuntimeError(f"active power inválido: {active_power}")
    total_j = float(active_power["integral"])
    dyn_w = max(float(active_power["mean"]) - idle_w, 0.0)
    dyn_j = max(total_j - idle_w*wall_s, 0.0)
    idle_j = idle_w*wall_s

    power = {
        "status": "measured",
        "scope": "ZCU104 board-input ~12V rail via INA226 hwmon0",
        "sensor_path": str(POWER),
        "sensor_iio_sampling_frequency_hz": 114,
        "telemetry_interval_ms": telemetry_ms,
        "baseline": base_power,
        "benchmark": active_power,
        "current_benchmark": active_current,
        "voltage_benchmark": active_voltage,
        "idle_mean_power_w": idle_w,
        "active_mean_power_w": float(active_power["mean"]),
        "dynamic_mean_power_w": dyn_w,
        "idle_energy_j": idle_j,
        "total_energy_j": total_j,
        "dynamic_energy_j": dyn_j,
        "idle_mj_per_inference": 1000*idle_j/n,
        "total_mj_per_inference": 1000*total_j/n,
        "dynamic_mj_per_inference": 1000*dyn_j/n,
    }

    boundaries = {
        "inference_only": "INT8 input copy before t0; t0 -> execute_async+wait -> t1; output copy/dequant/argmax after t1. Global throughput includes inter-request host copy/postprocessing.",
        "end_to_end": "normalized float input -> quantization -> input copy -> execute_async+wait -> output copy -> dequantization -> software softmax -> argmax",
        "saturated": "persistent prequantized input buffer reused; t0 -> execute_async+wait -> t1; no preprocessing/dequantization/softmax/argmax",
    }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "protocol": "MLP robust Vitis AI equivalent to LeNet 12x10000 campaign",
        "scenario": scenario,
        "threads_runners": threads,
        "physical_dpu_cores": 2,
        "batch": 1,
        "warmup_per_runner": warmup,
        "baseline_seconds": baseline_s,
        "telemetry_ms": telemetry_ms,
        "seed": 20260831,
        "outliers_removed": 0,
        "timing_boundary": boundaries[scenario],
        "standard_scaler_inside_end_to_end": False,
        "softmax_inside_end_to_end": scenario == "end_to_end",
        "temperature_pre": temp_pre,
        "temperature_after_warmup": temp_warm,
        "temperature_after_benchmark": temperatures(),
        "xmodel_sha256": sha256_file(xmodel),
        "test_data_sha256": EXPECTED_TEST_SHA256,
        "reference_sha256": EXPECTED_REF_SHA256,
        "script_sha256": script_sha,
        "xdputil_query_before_any_runner": dpu_query,
    }

    np.save(outdir/"latencies_ms.npy", lat)
    np.save(outdir/"request_sample_indices.npy", order_idx)
    if scenario != "saturated":
        np.save(outdir/"predictions.npy", preds)
        np.save(outdir/"divergent_positions.npy", divergent.astype(np.int64))

    with (outdir/"blocks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        w.writeheader()
        w.writerows(block_rows)

    fields = [
        "perf_counter_ns", "unix_s", "phase",
        "board_input_power_w", "board_input_current_a",
        "board_input_voltage_v", "read_duration_ns"
    ]
    with (outdir/"telemetry.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(tel.rows)

    write_json(outdir/"metrics_summary.json", metrics)
    write_json(outdir/"power_summary.json", power)
    write_json(outdir/"metadata.json", metadata)
    write_json(outdir/"config_summary.json", {"metrics": metrics, "power": power, "metadata": metadata})

    files = sorted(p for p in outdir.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (outdir/"SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files),
        encoding="utf-8"
    )

    print(
        f"[{scenario} T{threads}] OK: "
        f"FPS={metrics['throughput_effective_inf_s']:.3f}, "
        f"lat={metrics['latency']['mean_ms']:.6f} ms, "
        f"Pdyn={power['dynamic_mean_power_w']:.6f} W, "
        f"Edyn={power['dynamic_mj_per_inference']:.6f} mJ/inf",
        flush=True
    )
    return {"metrics": metrics, "power": power, "metadata": metadata}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmodel", type=Path, default=HERE/"iris_mlp.xmodel")
    ap.add_argument("--test-data", type=Path, default=HERE/"iris_test.npz")
    ap.add_argument("--reference", type=Path, default=HERE/"quantized_test_outputs.npz")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--baseline-seconds", type=float, default=5.0)
    ap.add_argument("--telemetry-ms", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260831)
    a = ap.parse_args()

    if a.warmup != 100:
        raise SystemExit("protocolo robusto exige --warmup 100 por runner")
    if abs(a.baseline_seconds - 5.0) > 1e-12:
        raise SystemExit("protocolo robusto exige --baseline-seconds 5")
    if a.telemetry_ms != 10:
        raise SystemExit("protocolo robusto exige --telemetry-ms 10")
    if a.seed != 20260831:
        raise SystemExit("protocolo robusto exige --seed 20260831")

    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit(f"Recuso sobrescrever {a.output_dir}")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    hashes = {
        "xmodel": sha256_file(a.xmodel),
        "test_data": sha256_file(a.test_data),
        "reference": sha256_file(a.reference),
        "script": sha256_file(Path(__file__).resolve()),
    }
    expected = {
        "xmodel": EXPECTED_XMODEL_SHA256,
        "test_data": EXPECTED_TEST_SHA256,
        "reference": EXPECTED_REF_SHA256,
    }
    bad = {k: [hashes[k], expected[k]] for k in expected if hashes[k] != expected[k]}
    if bad:
        raise RuntimeError(f"hash mismatch: {bad}")

    dpu_query = run_command(["xdputil", "query"])
    if dpu_query.get("returncode") != 0:
        raise RuntimeError(f"xdputil query falhou: {dpu_query}")

    data = np.load(a.test_data)
    normalized = np.asarray(data["features"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    ref = np.asarray(np.load(a.reference)["predictions"], dtype=np.int64).reshape(-1)
    if normalized.shape != (30,4) or labels.shape != (30,) or ref.shape != (30,):
        raise RuntimeError("formas inesperadas")

    validation = validate_unique30(a.xmodel, normalized, labels, ref)
    write_json(a.output_dir/"validation_unique_30.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"validação falhou: {validation}")

    # Pré-quantização canônica para inference_only e saturated.
    # Evita manter um runner auxiliar aberto durante T1..T4.
    qinputs = np.clip(
        np.rint(normalized * (2 ** EXPECTED_INPUT_FIX)),
        -128, 127
    ).astype(np.int8).reshape(30, 1, 4)

    order_idx = deterministic_order(10000, a.seed)
    np.save(a.output_dir/"selected_request_indices_10000.npy", order_idx)
    write_json(a.output_dir/"preflight.json", {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": hashes,
        "validation": validation,
        "xdputil_query": dpu_query,
        "temperature": temperatures(),
        "protocol": {
            "configs": 12,
            "scenarios": ["inference_only","end_to_end","saturated"],
            "threads_runners": [1,2,3,4],
            "inferences_per_config": 10000,
            "blocks_per_config": 10,
            "samples_per_block": 1000,
            "warmup_per_runner": a.warmup,
            "baseline_seconds": a.baseline_seconds,
            "telemetry_ms": a.telemetry_ms,
            "seed": a.seed,
        }
    })

    all_results = []
    for scenario in ["inference_only", "end_to_end", "saturated"]:
        for threads in [1,2,3,4]:
            # Cooling padronizado entre configurações.
            if all_results:
                print("[COOLDOWN] 30 s entre configurações", flush=True)
                time.sleep(30)
            cfg_dir = a.output_dir / f"{scenario}_T{threads}"
            result = run_config(
                scenario=scenario,
                threads=threads,
                xmodel=a.xmodel,
                normalized=normalized,
                ref=ref,
                qinputs=qinputs,
                order_idx=order_idx,
                outdir=cfg_dir,
                warmup=a.warmup,
                baseline_s=a.baseline_seconds,
                telemetry_ms=a.telemetry_ms,
                dpu_query=dpu_query,
                script_sha=hashes["script"],
            )
            all_results.append({
                "scenario": scenario,
                "threads_runners": threads,
                "metrics": result["metrics"],
                "power": result["power"],
            })

    write_json(a.output_dir/"all_configs_summary.json", all_results)

    integrity = {
        "expected_configs": 12,
        "completed_configs": len(all_results),
        "all_inferences_10000": all(x["metrics"]["inferences"] == 10000 for x in all_results),
        "all_latency_count_10000": all(x["metrics"]["latency"]["count"] == 10000 for x in all_results),
        "all_positive_fps": all(x["metrics"]["throughput_effective_inf_s"] > 0 for x in all_results),
        "all_power_measured": all(x["power"]["status"] == "measured" for x in all_results),
        "all_power_bracketed": all(x["power"]["benchmark"].get("bracketed") is True for x in all_results),
    }
    integrity["passed"] = all([
        integrity["completed_configs"] == 12,
        integrity["all_inferences_10000"],
        integrity["all_latency_count_10000"],
        integrity["all_positive_fps"],
        integrity["all_power_measured"],
        integrity["all_power_bracketed"],
    ])
    write_json(a.output_dir/"integrity_summary.json", integrity)

    root_files = sorted(
        p for p in a.output_dir.iterdir()
        if p.is_file() and p.name != "SHA256SUMS_ROOT.txt"
    )
    (a.output_dir/"SHA256SUMS_ROOT.txt").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in root_files),
        encoding="utf-8"
    )

    print(json.dumps(integrity, indent=2), flush=True)


if __name__ == "__main__":
    main()
