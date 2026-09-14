#!/usr/bin/env python3
"""
Benchmark serial canônico MLP/Iris na ZCU104 — revisão metodológica.

Uso principal:
  - 5 campanhas independentes de 30.000 inferências
  - 5 campanhas independentes de 100.000 inferências
  - batch 1, serial, síncrono, uma inferência em voo
  - 200 warm-ups
  - baseline idle de 5 s APÓS warm-up
  - telemetria nominal de 100 ms
  - 30 amostras por passagem, seed 20260831
  - nenhum outlier removido

Janelas:
  accelerator:
      execute_async + wait apenas.
  device_call:
      quantização da entrada normalizada -> DPU -> dequantização.
  application_end_to_end:
      entrada JÁ normalizada -> quantização -> DPU -> dequantização
      -> softmax em ARM/Python -> argmax.

O StandardScaler NÃO entra em application_end_to_end, pois a metodologia
CPU/GPU da MLP usa a entrada já normalizada.

Energia:
  - baseline medido após warm-up;
  - integração trapezoidal no intervalo exato do benchmark, usando amostras
    que envolvem as duas bordas;
  - potência dinâmica = potência ativa - baseline;
  - energia dinâmica = E_total - P_idle * duração exata.
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
from typing import Any

import numpy as np

from vart_common import IrisDpu

HERE = Path(__file__).resolve().parent

EXPECTED_XMODEL_SHA256 = "47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce"
EXPECTED_TEST_SHA256 = "3972a24323d73613dfca346b56333a5843d45e399cdf80d05ac04b1257ed86e1"
EXPECTED_REF_SHA256 = "b0ab40e38bd10bf685badc8552e2a66030c247447df626ab05a12e796364ba0e"

DEFAULT_POWER = Path("/sys/class/hwmon/hwmon0/power1_input")
DEFAULT_CURRENT = Path("/sys/class/hwmon/hwmon0/curr1_input")
DEFAULT_VOLTAGE = Path("/sys/class/hwmon/hwmon0/in2_input")
AMS = Path("/sys/bus/iio/devices/iio:device0")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_float(path: Path):
    try:
        return float(path.read_text().strip())
    except Exception:
        return None


def run_command(cmd):
    try:
        p = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=60,
        )
        return {"command": list(cmd), "returncode": p.returncode, "output": p.stdout}
    except Exception as e:
        return {"command": list(cmd), "returncode": None, "output": repr(e)}


def temp_c(prefix: str):
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


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def wilson(correct: int, n: int, z: float = 1.959963984540054):
    p = correct / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [center - half, center + half]


def tcrit_975_large_df(df: int) -> float:
    # Cornish-Fisher; aqui df >= 999 nas campanhas oficiais.
    z = 1.959963984540054
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def stat_summary(values, passage_means):
    a = np.asarray(values, dtype=np.float64)
    pm = np.asarray(passage_means, dtype=np.float64)
    if a.size == 0:
        raise RuntimeError("vetor de latência vazio")
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise RuntimeError("latência inválida encontrada")
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1))
    if len(pm) > 1:
        tc = tcrit_975_large_df(len(pm) - 1)
        se = float(np.std(pm, ddof=1) / math.sqrt(len(pm)))
        lo = float(np.mean(pm) - tc * se)
        hi = float(np.mean(pm) + tc * se)
    else:
        lo = hi = float(np.mean(pm))
    return {
        "count": int(a.size),
        "mean_ms": mean,
        "median_ms": float(np.median(a)),
        "std_sample_ms": sd,
        "cv_percent": float(100.0 * sd / mean),
        "minimum_ms": float(np.min(a)),
        "maximum_ms": float(np.max(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ci95_lower_ms": lo,
        "mean_ci95_upper_ms": hi,
        "ci_unit": "means of complete 30-sample passages",
        "ci_passages": int(len(pm)),
    }


class TelemetrySampler:
    """INA226 polling thread. Stores raw time-aligned board-input telemetry."""

    def __init__(self, interval_s: float):
        self.interval_s = float(interval_s)
        self.phase = "setup"
        self.rows = []
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def _read(self, phase=None):
        t0 = time.perf_counter_ns()
        row = {
            "perf_counter_ns": t0,
            "unix_s": time.time(),
            "phase": self.phase if phase is None else phase,
            "board_input_power_w": None,
            "board_input_current_a": None,
            "board_input_voltage_v": None,
            "read_duration_ns": None,
        }
        p = read_float(DEFAULT_POWER)
        c = read_float(DEFAULT_CURRENT)
        v = read_float(DEFAULT_VOLTAGE)
        if p is not None:
            row["board_input_power_w"] = p / 1e6
        if c is not None:
            row["board_input_current_a"] = c / 1000.0
        if v is not None:
            row["board_input_voltage_v"] = v / 1000.0
        row["read_duration_ns"] = time.perf_counter_ns() - t0
        with self._lock:
            self.rows.append(row)
        return row

    def sample_now(self, phase=None):
        return self._read(phase)

    def _loop(self):
        next_t = time.perf_counter()
        while not self._stop.is_set():
            self._read()
            next_t += self.interval_s
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.perf_counter()

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def numeric_rows(rows, key):
    return [r for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]


def integrate_exact_window(rows, key, start_ns: int, end_ns: int):
    r = numeric_rows(rows, key)
    if len(r) < 2:
        return None
    r = sorted(r, key=lambda x: x["perf_counter_ns"])
    t = np.asarray([x["perf_counter_ns"] for x in r], dtype=np.float64)
    y = np.asarray([x[key] for x in r], dtype=np.float64)

    if t[0] > start_ns or t[-1] < end_ns:
        return {
            "status": "insufficient_bracketing",
            "first_sample_ns": int(t[0]),
            "last_sample_ns": int(t[-1]),
            "start_ns": int(start_ns),
            "end_ns": int(end_ns),
        }

    inside = (t >= start_ns) & (t <= end_ns)
    ti = t[inside]
    yi = y[inside]

    start_y = float(np.interp(float(start_ns), t, y))
    end_y = float(np.interp(float(end_ns), t, y))
    tx = np.concatenate(([float(start_ns)], ti, [float(end_ns)]))
    yx = np.concatenate(([start_y], yi, [end_y]))

    order = np.argsort(tx)
    tx, yx = tx[order], yx[order]
    keep = np.concatenate(([True], np.diff(tx) > 0))
    tx, yx = tx[keep], yx[keep]

    rel_s = (tx - float(start_ns)) / 1e9
    energy = float(np.trapz(yx, rel_s))
    duration_s = (end_ns - start_ns) / 1e9
    gaps = np.diff(tx) / 1e6 if len(tx) > 1 else np.asarray([], dtype=np.float64)

    return {
        "status": "measured",
        "samples_inside": int(np.sum(inside)),
        "bracketed": True,
        "duration_s": duration_s,
        "mean": energy / duration_s,
        "median_samples_inside": float(np.median(yi)) if yi.size else None,
        "std_sample_inside": float(np.std(yi, ddof=1)) if yi.size > 1 else 0.0,
        "minimum_samples_inside": float(np.min(yi)) if yi.size else None,
        "maximum_samples_inside": float(np.max(yi)) if yi.size else None,
        "max_sample_gap_ms": float(np.max(gaps)) if gaps.size else None,
        "integral": energy,
    }


def write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def validate_unique_30(dpu, normalized, labels, ref):
    preds = []
    raw_qouts = []
    for x in normalized:
        qin = dpu.quantize(x)
        qout = dpu.execute_quantized(qin)
        raw_qouts.append(np.asarray(qout, dtype=np.int8).reshape(-1).copy())
        preds.append(int(np.argmax(dpu.decode(qout))))
    preds = np.asarray(preds, dtype=np.int64)
    raw_qouts = np.asarray(raw_qouts, dtype=np.int8)
    correct = int(np.sum(preds == labels))
    agreement = int(np.sum(preds == ref))
    ci = wilson(correct, len(labels))
    return preds, raw_qouts, {
        "samples_unique": int(len(labels)),
        "correct": correct,
        "accuracy": float(correct / len(labels)),
        "accuracy_percent": float(100.0 * correct / len(labels)),
        "wilson95": ci,
        "wilson95_percent": [100.0 * ci[0], 100.0 * ci[1]],
        "reference_agreement": agreement,
        "reference_agreement_percent": float(100.0 * agreement / len(labels)),
        "divergences": int(np.sum(preds != ref)),
        "passed": bool(correct == 29 and agreement == 30),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmodel", type=Path, default=HERE / "iris_mlp.xmodel")
    ap.add_argument("--test-data", type=Path, default=HERE / "iris_test.npz")
    ap.add_argument("--reference", type=Path, default=HERE / "quantized_test_outputs.npz")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--inferences", type=int, choices=[30000, 100000])
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--baseline-seconds", type=float, default=5.0)
    ap.add_argument("--telemetry-ms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--strict-hashes", action="store_true", default=True)
    ap.add_argument("--no-strict-hashes", dest="strict_hashes", action="store_false")
    a = ap.parse_args()

    if not a.preflight_only and a.inferences is None:
        raise SystemExit("--inferences é obrigatório exceto com --preflight-only")
    if a.warmup != 200:
        raise SystemExit("protocolo oficial exige --warmup 200")
    if a.baseline_seconds < 5.0:
        raise SystemExit("protocolo exige baseline >= 5 s")
    if a.telemetry_ms != 100:
        raise SystemExit("protocolo serial CPU/GPU exige --telemetry-ms 100")

    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit(f"Recuso sobrescrever {a.output_dir}")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    hashes = {
        "xmodel": sha256_file(a.xmodel),
        "test_data": sha256_file(a.test_data),
        "reference": sha256_file(a.reference),
    }
    expected = {
        "xmodel": EXPECTED_XMODEL_SHA256,
        "test_data": EXPECTED_TEST_SHA256,
        "reference": EXPECTED_REF_SHA256,
    }
    if a.strict_hashes:
        bad = {k: [hashes[k], expected[k]] for k in hashes if hashes[k] != expected[k]}
        if bad:
            raise RuntimeError(f"hash mismatch: {bad}")

    data = np.load(a.test_data)
    normalized = np.asarray(data["features"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    if normalized.shape != (30, 4) or labels.shape != (30,):
        raise RuntimeError(f"dataset inesperado: features={normalized.shape}, labels={labels.shape}")

    ref = np.asarray(np.load(a.reference)["predictions"], dtype=np.int64).reshape(-1)
    if ref.shape != (30,):
        raise RuntimeError(f"referência inesperada: {ref.shape}")

    # Consulta a DPU ANTES de criar o VART Runner. Algumas imagens/versões
    # de xdputil podem bloquear enquanto o mesmo processo mantém um runner aberto.
    xdputil_query_before_runner = run_command(["xdputil", "query"])
    if xdputil_query_before_runner.get("returncode") != 0:
        raise RuntimeError(
            "xdputil query falhou antes da criação do runner: "
            + str(xdputil_query_before_runner)
        )

    dpu = IrisDpu(a.xmodel)

    preds_unique, raw_unique, validation = validate_unique_30(dpu, normalized, labels, ref)
    np.save(a.output_dir / "validation_predictions_unique30.npy", preds_unique)
    np.save(a.output_dir / "validation_raw_output_int8_unique30.npy", raw_unique)
    write_json(a.output_dir / "validation_unique_30.json", validation)
    if not validation["passed"]:
        raise SystemExit("Validação física 29/30 e 30/30 falhou; benchmark bloqueado")

    preflight = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": hashes,
        "validation": validation,
        "temperature": temperatures(),
        "power_now_w": (read_float(DEFAULT_POWER) / 1e6) if read_float(DEFAULT_POWER) is not None else None,
        "current_now_a": (read_float(DEFAULT_CURRENT) / 1000.0) if read_float(DEFAULT_CURRENT) is not None else None,
        "voltage_now_v": (read_float(DEFAULT_VOLTAGE) / 1000.0) if read_float(DEFAULT_VOLTAGE) is not None else None,
        "xdputil_query": xdputil_query_before_runner,
        "platform": platform.platform(),
    }
    write_json(a.output_dir / "preflight.json", preflight)

    if a.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    # Warm-up PRIMEIRO. Não entra no benchmark nem no baseline idle.
    warm_rng = np.random.default_rng(a.seed ^ 0x5A5A5A5A)
    for _ in range(a.warmup):
        x = normalized[int(warm_rng.integers(0, 30))]
        dpu.execute_quantized(dpu.quantize(x))

    temp_after_warmup = temperatures()

    sampler = TelemetrySampler(a.telemetry_ms / 1000.0)
    sampler.start()

    # Baseline idle APÓS warm-up.
    sampler.phase = "baseline"
    sampler.sample_now("baseline_boundary_before")
    baseline_start_ns = time.perf_counter_ns()
    time.sleep(a.baseline_seconds)
    baseline_end_ns = time.perf_counter_ns()
    sampler.sample_now("baseline_boundary_after")

    baseline_power = integrate_exact_window(
        sampler.rows, "board_input_power_w", baseline_start_ns, baseline_end_ns
    )
    if not baseline_power or baseline_power.get("status") != "measured":
        sampler.stop()
        raise RuntimeError(f"baseline power inválido: {baseline_power}")
    idle_power_w = float(baseline_power["mean"])

    # Benchmark order independent of warm-up.
    rng = np.random.default_rng(a.seed)

    n = int(a.inferences)
    latency_acc = np.empty(n, dtype=np.float64)
    latency_device = np.empty(n, dtype=np.float64)
    latency_e2e = np.empty(n, dtype=np.float64)
    sample_indices = np.empty(n, dtype=np.int16)
    predictions = np.empty(n, dtype=np.int8)

    passage_rows = []
    done = 0

    sampler.phase = "benchmark"
    sampler.sample_now("benchmark_boundary_before")
    benchmark_start_ns = time.perf_counter_ns()
    benchmark_start_unix = time.time()

    while done < n:
        count = min(30, n - done)
        order = rng.permutation(30)[:count]
        p0 = time.perf_counter_ns()

        acc_block = []
        dev_block = []
        e2e_block = []

        for local_i, idx_np in enumerate(order):
            idx = int(idx_np)
            x = normalized[idx]

            # Entrada já normalizada antes de t0.
            # application_end_to_end começa em t0.
            t0 = time.perf_counter_ns()

            # Quantização faz parte de device_call/e2e.
            qin = dpu.quantize(x)

            # Copia a entrada INT8 para o buffer ANTES do cronômetro accelerator.
            dpu.ib[...] = qin

            # accelerator = SOMENTE execute_async + wait.
            t_acc0 = time.perf_counter_ns()
            jid = dpu.runner.execute_async([dpu.ib], [dpu.ob])
            status = dpu.runner.wait(jid)
            t_acc1 = time.perf_counter_ns()
            if status not in (0, None):
                raise RuntimeError(f"runner.wait retornou {status}")

            # Cópia/dequantização ficam fora do accelerator, mas dentro de device/e2e.
            qout = dpu.ob.copy()
            logits = dpu.decode(qout)
            t_device_end = time.perf_counter_ns()

            probs = softmax(logits)
            pred = int(np.argmax(probs))
            t_e2e_end = time.perf_counter_ns()

            pos = done + local_i
            latency_acc[pos] = (t_acc1 - t_acc0) / 1e6
            latency_device[pos] = (t_device_end - t0) / 1e6
            latency_e2e[pos] = (t_e2e_end - t0) / 1e6
            sample_indices[pos] = idx
            predictions[pos] = pred

            acc_block.append(latency_acc[pos])
            dev_block.append(latency_device[pos])
            e2e_block.append(latency_e2e[pos])

        p1 = time.perf_counter_ns()
        passage_rows.append({
            "passage": len(passage_rows),
            "samples": count,
            "accelerator_mean_ms": float(np.mean(acc_block)),
            "device_call_mean_ms": float(np.mean(dev_block)),
            "application_e2e_mean_ms": float(np.mean(e2e_block)),
            "passage_duration_ms": (p1 - p0) / 1e6,
            "passage_throughput_inf_s": count / ((p1 - p0) / 1e9),
        })
        done += count

    benchmark_end_ns = time.perf_counter_ns()
    benchmark_end_unix = time.time()
    sampler.sample_now("benchmark_boundary_after")
    sampler.phase = "post"
    time.sleep(2.0 * a.telemetry_ms / 1000.0)
    sampler.stop()

    if done != n:
        raise RuntimeError(f"contagem incorreta: {done} != {n}")
    if np.any(latency_acc <= 0) or np.any(latency_device <= 0) or np.any(latency_e2e <= 0):
        raise RuntimeError("latência <= 0")

    # Determinismo: cada repetição deve reproduzir a referência INT8 da amostra.
    expected_repeated = ref[sample_indices.astype(np.int64)]
    divergent = np.flatnonzero(predictions.astype(np.int64) != expected_repeated)
    deterministic = bool(divergent.size == 0)

    full = [r for r in passage_rows if r["samples"] == 30]
    wall_s = (benchmark_end_ns - benchmark_start_ns) / 1e9

    metrics = {
        "inferences": n,
        "full_passages": len(full),
        "final_passage_samples": int(passage_rows[-1]["samples"]),
        "benchmark_wall_s": wall_s,
        "throughput_effective_inf_s": n / wall_s,
        "passage_throughput_arithmetic_mean_inf_s": float(np.mean([r["passage_throughput_inf_s"] for r in passage_rows])),
        "accelerator": stat_summary(
            latency_acc, [r["accelerator_mean_ms"] for r in full]
        ),
        "device_call": stat_summary(
            latency_device, [r["device_call_mean_ms"] for r in full]
        ),
        "application_end_to_end": stat_summary(
            latency_e2e, [r["application_e2e_mean_ms"] for r in full]
        ),
        "determinism": {
            "passed": deterministic,
            "divergences": int(divergent.size),
            "divergent_positions_first100": divergent[:100].tolist(),
        },
    }
    if not deterministic:
        raise RuntimeError(f"determinismo falhou em {divergent.size} inferências")

    active_power = integrate_exact_window(
        sampler.rows, "board_input_power_w", benchmark_start_ns, benchmark_end_ns
    )
    active_current = integrate_exact_window(
        sampler.rows, "board_input_current_a", benchmark_start_ns, benchmark_end_ns
    )
    active_voltage = integrate_exact_window(
        sampler.rows, "board_input_voltage_v", benchmark_start_ns, benchmark_end_ns
    )
    if not active_power or active_power.get("status") != "measured":
        raise RuntimeError(f"active power inválido: {active_power}")

    total_energy_j = float(active_power["integral"])
    dynamic_power_w = max(float(active_power["mean"]) - idle_power_w, 0.0)
    dynamic_energy_j = max(total_energy_j - idle_power_w * wall_s, 0.0)

    power = {
        "status": "measured",
        "scope": "ZCU104 board-input ~12V rail via INA226 hwmon0",
        "sensor_path": str(DEFAULT_POWER),
        "sensor_iio_sampling_frequency_hz": 114,
        "telemetry_interval_ms": a.telemetry_ms,
        "baseline": baseline_power,
        "benchmark": active_power,
        "current_benchmark": active_current,
        "voltage_benchmark": active_voltage,
        "idle_mean_power_w": idle_power_w,
        "active_mean_power_w": float(active_power["mean"]),
        "dynamic_mean_power_w": dynamic_power_w,
        "total_energy_j": total_energy_j,
        "dynamic_energy_j": dynamic_energy_j,
        "total_mj_per_inference": 1000.0 * total_energy_j / n,
        "dynamic_mj_per_inference": 1000.0 * dynamic_energy_j / n,
    }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_start_unix": benchmark_start_unix,
        "benchmark_end_unix": benchmark_end_unix,
        "platform": platform.platform(),
        "protocol": "METODOLOGIA_BENCHMARK_MLP_IRIS.md + revision for exact stated boundaries",
        "batch": 1,
        "serial_synchronous": True,
        "one_inference_in_flight": True,
        "warmup": a.warmup,
        "baseline_seconds": a.baseline_seconds,
        "baseline_after_warmup": True,
        "telemetry_ms": a.telemetry_ms,
        "seed_benchmark": a.seed,
        "seed_warmup": int(a.seed ^ 0x5A5A5A5A),
        "samples_per_passage": 30,
        "outliers_removed": 0,
        "xmodel_sha256": hashes["xmodel"],
        "test_data_sha256": hashes["test_data"],
        "reference_sha256": hashes["reference"],
        "temperature_preflight": preflight["temperature"],
        "temperature_after_warmup": temp_after_warmup,
        "temperature_after_benchmark": temperatures(),
        "timing_boundaries": {
            "accelerator": "INT8 input copied to persistent DPU buffer before t0; t0 -> runner.execute_async + runner.wait -> t1; excludes input copy, output copy, dequantization, softmax and argmax",
            "device_call": "normalized input -> input quantization -> DPU execute/wait -> output dequantization",
            "application_end_to_end": "normalized input -> input quantization -> DPU -> dequantization -> software softmax -> argmax",
            "standard_scaler_inside_application_e2e": False,
            "softmax_inside_application_e2e": True,
        },
        "xdputil_query_before_runner": preflight["xdputil_query"],
        "xdputil_status_after": run_command(["xdputil", "status"]),
    }

    np.save(a.output_dir / "latencies_accelerator_ms.npy", latency_acc)
    np.save(a.output_dir / "latencies_device_call_ms.npy", latency_device)
    np.save(a.output_dir / "latencies_application_e2e_ms.npy", latency_e2e)
    np.save(a.output_dir / "sample_indices.npy", sample_indices)
    np.save(a.output_dir / "predictions.npy", predictions)
    np.save(a.output_dir / "divergent_positions.npy", divergent.astype(np.int64))

    with (a.output_dir / "passages.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(passage_rows[0].keys()))
        w.writeheader()
        w.writerows(passage_rows)

    telemetry_fields = [
        "perf_counter_ns", "unix_s", "phase",
        "board_input_power_w", "board_input_current_a", "board_input_voltage_v",
        "read_duration_ns",
    ]
    with (a.output_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=telemetry_fields)
        w.writeheader()
        w.writerows(sampler.rows)

    write_json(a.output_dir / "metrics_summary.json", metrics)
    write_json(a.output_dir / "power_summary.json", power)
    write_json(a.output_dir / "metadata.json", metadata)

    summary = {
        "validation": validation,
        "metrics": metrics,
        "power": power,
        "output_dir": str(a.output_dir),
    }
    write_json(a.output_dir / "benchmark_summary.json", summary)

    generated = sorted(p for p in a.output_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (a.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in generated),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
