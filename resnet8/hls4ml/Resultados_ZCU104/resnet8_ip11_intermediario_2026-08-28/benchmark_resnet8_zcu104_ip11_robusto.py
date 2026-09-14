#!/usr/bin/env python3
"""
Benchmark robusto ResNet8 hls4ml / ZCU104 / PYNQ.

Objetivos:
- batch 1
- CIFAR-10 oficial
- acurácia separada do benchmark de desempenho
- inference-only e end-to-end separados
- 100 warm-ups
- 100 ciclos por tamanho (100, 1000, 10000)
- prefixos 10/20/50/100
- latência: média, IC95 entre ciclos, mediana, desvio, p95, p99, min, max
- throughput: média, desvio, IC95, global efetivo
- PMBus: potência idle/ativa/dinâmica e energia total/dinâmica por inferência
- telemetria ARM/PS: CPU, RSS, frequência e temperatura quando psutil estiver disponível
- checkpoints por ciclo e arquivos brutos
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pynq import Overlay, allocate

try:
    from pynq.pmbus import get_rails, DataRecorder
except Exception:
    from pynq import get_rails, DataRecorder

try:
    import psutil
except Exception:
    psutil = None


FIXED_W = 22
FIXED_I = 12
FIXED_F = FIXED_W - FIXED_I
FIXED_SCALE = 1 << FIXED_F
FIXED_MASK = (1 << FIXED_W) - 1
FIXED_SIGN = 1 << (FIXED_W - 1)
RAW_MIN = -(1 << (FIXED_W - 1))
RAW_MAX = (1 << (FIXED_W - 1)) - 1

N_PIXELS = 32 * 32
INPUT_BYTES = N_PIXELS * 16
OUTPUT_BYTES = 64
N_CLASSES = 10
Z95 = 1.959963984540054


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_interval(correct: int, total: int, z: float = Z95):
    p = correct / total
    d = 1.0 + z * z / total
    c = (p + z * z / (2.0 * total)) / d
    h = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / d
    return c - h, c + h


def pack_image_u8(image_u8: np.ndarray, destination):
    if image_u8.shape != (32, 32, 3):
        raise ValueError(f"Imagem com shape inválido: {image_u8.shape}")
    x = image_u8.astype(np.float32) / np.float32(255.0)
    q = np.rint(x * np.float32(FIXED_SCALE)).astype(np.int64)
    np.clip(q, RAW_MIN, RAW_MAX, out=q)
    q = (q & FIXED_MASK).astype(np.uint32).reshape(N_PIXELS, 3)
    destination[:, 0] = q[:, 0]
    destination[:, 1] = q[:, 1]
    destination[:, 2] = q[:, 2]
    destination[:, 3] = 0


def pack_dataset_uint8(x_test: np.ndarray, chunk: int = 250) -> np.ndarray:
    packed = np.zeros((len(x_test), N_PIXELS, 4), dtype=np.uint32)
    for start in range(0, len(x_test), chunk):
        end = min(start + chunk, len(x_test))
        x = x_test[start:end].astype(np.float32) / np.float32(255.0)
        q = np.rint(x * np.float32(FIXED_SCALE)).astype(np.int64)
        np.clip(q, RAW_MIN, RAW_MAX, out=q)
        q = (q & FIXED_MASK).astype(np.uint32).reshape(end - start, N_PIXELS, 3)
        packed[start:end, :, 0:3] = q
    return packed


def decode_logits(source) -> np.ndarray:
    raw = (np.asarray(source[:10], dtype=np.uint32) & FIXED_MASK).astype(np.int64)
    neg = (raw & FIXED_SIGN) != 0
    raw[neg] -= (1 << FIXED_W)
    return raw.astype(np.float32) / np.float32(FIXED_SCALE)


def confusion_matrix(y_true, y_pred):
    m = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def make_nested_stratified_indices(labels: np.ndarray, seed: int):
    """
    Gera conjuntos aninhados, estratificados e determinísticos.
    O conjunto de 10k contém TODO o test set; portanto é diretamente comparável.
    """
    rng = np.random.default_rng(seed)
    per_class = {}
    for c in range(N_CLASSES):
        idx = np.flatnonzero(labels == c).copy()
        if len(idx) != 1000:
            raise RuntimeError(f"Classe {c}: esperado 1000, obtido {len(idx)}")
        rng.shuffle(idx)
        per_class[c] = idx

    result = {}
    for n in (100, 1000, 10000):
        k = n // N_CLASSES
        sel = np.concatenate([per_class[c][:k] for c in range(N_CLASSES)])
        local = np.random.default_rng(seed + n)
        local.shuffle(sel)
        result[n] = sel.astype(np.int64)

    if not set(result[100]).issubset(set(result[1000])):
        raise RuntimeError("Conjunto 100 não está contido em 1000")
    if not set(result[1000]).issubset(set(result[10000])):
        raise RuntimeError("Conjunto 1000 não está contido em 10000")
    if len(np.unique(result[10000])) != 10000:
        raise RuntimeError("Conjunto de 10k não cobre exatamente o test set")
    return result


def dma_status(dma):
    return {
        "send_running": bool(dma.sendchannel.running),
        "send_idle": bool(dma.sendchannel.idle),
        "send_error": bool(dma.sendchannel.error),
        "recv_running": bool(dma.recvchannel.running),
        "recv_idle": bool(dma.recvchannel.idle),
        "recv_error": bool(dma.recvchannel.error),
    }


def transfer_once(dma, input_buffer, output_buffer):
    # Ordem obrigatória: recepção antes do envio.
    dma.recvchannel.transfer(output_buffer, nbytes=OUTPUT_BYTES)
    dma.sendchannel.transfer(input_buffer, nbytes=INPUT_BYTES)
    dma.sendchannel.wait()
    dma.recvchannel.wait()


def available_power_rails():
    rails = get_rails()
    info = {}
    for name, rail in rails.items():
        power = getattr(rail, "power", None)
        voltage = getattr(rail, "voltage", None)
        current = getattr(rail, "current", None)
        info[name] = {
            "has_power": power is not None,
            "power_sensor": getattr(power, "name", None),
            "power_w": float(power.value) if power is not None else None,
            "voltage_v": float(voltage.value) if voltage is not None else None,
            "current_a": float(current.value) if current is not None else None,
        }
    return rails, info


def choose_power_rail(rails, requested: str):
    powered = [k for k, r in rails.items() if getattr(r, "power", None) is not None]
    if requested.lower() != "auto":
        if requested not in rails:
            raise RuntimeError(
                f"Rail '{requested}' não existe. Rails com potência: {powered}"
            )
        if rails[requested].power is None:
            raise RuntimeError(f"Rail '{requested}' não possui sensor de potência.")
        return requested

    exact = [k for k in powered if k.lower() == "12v"]
    if len(exact) == 1:
        return exact[0]

    containing = [k for k in powered if "12v" in k.lower()]
    if len(containing) == 1:
        return containing[0]

    raise RuntimeError(
        "Não foi possível escolher automaticamente o rail de entrada/placa. "
        f"Rails com potência: {powered}. Rode novamente com --power-rail NOME."
    )


def read_max_temperature_c():
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return math.nan
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        values = []
        for entries in temps.values():
            for e in entries:
                if e.current is not None and math.isfinite(float(e.current)):
                    values.append(float(e.current))
        return max(values) if values else math.nan
    except Exception:
        return math.nan


class HostTelemetry:
    """Telemetria do ARM/PS a 100 ms. Métrica de overhead do host, não de uso do PL."""

    def __init__(self, interval_s=0.1):
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.process = psutil.Process() if psutil is not None else None

    def start(self):
        if psutil is None:
            return
        self.samples = []
        self._stop.clear()
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        ncpu = psutil.cpu_count(logical=True) or 1
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                system_cpu = float(psutil.cpu_percent(interval=None))
                proc_cpu = float(self.process.cpu_percent(interval=None))
                proc_norm = proc_cpu / ncpu
                rss_mib = float(self.process.memory_info().rss) / (1024.0 ** 2)
                freq = psutil.cpu_freq()
                freq_mhz = float(freq.current) if freq and freq.current is not None else math.nan
                temp_c = read_max_temperature_c()
                self.samples.append({
                    "perf_counter_s": t,
                    "system_cpu_percent": system_cpu,
                    "process_cpu_percent": proc_cpu,
                    "process_cpu_normalized_percent": proc_norm,
                    "rss_mib": rss_mib,
                    "cpu_freq_mhz": freq_mhz,
                    "temperature_max_c": temp_c,
                })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def summary(self):
        if not self.samples:
            return {
                "host_samples": 0,
                "system_cpu_percent_mean": math.nan,
                "process_cpu_percent_mean": math.nan,
                "process_cpu_normalized_percent_mean": math.nan,
                "rss_mib_mean": math.nan,
                "cpu_freq_mhz_mean": math.nan,
                "temperature_max_c_mean": math.nan,
            }

        def finite_mean(key):
            vals = np.asarray([s[key] for s in self.samples], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if len(vals) else math.nan

        return {
            "host_samples": len(self.samples),
            "system_cpu_percent_mean": finite_mean("system_cpu_percent"),
            "process_cpu_percent_mean": finite_mean("process_cpu_percent"),
            "process_cpu_normalized_percent_mean": finite_mean("process_cpu_normalized_percent"),
            "rss_mib_mean": finite_mean("rss_mib"),
            "cpu_freq_mhz_mean": finite_mean("cpu_freq_mhz"),
            "temperature_max_c_mean": finite_mean("temperature_max_c"),
        }


def append_host_samples(path: Path, scenario: str, n: int, cycle: int, samples):
    if not samples:
        return
    new_file = not path.exists()
    fields = ["scenario", "n", "cycle"] + list(samples[0].keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for s in samples:
            row = {"scenario": scenario, "n": n, "cycle": cycle}
            row.update(s)
            w.writerow(row)


def append_power_frame(path: Path, scenario: str, n: int, cycle: int, frame):
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        fieldnames = ["scenario", "n", "cycle", "timestamp"] + list(frame.columns)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for ts, row in frame.iterrows():
            out = {
                "scenario": scenario,
                "n": n,
                "cycle": cycle,
                "timestamp": str(ts),
            }
            for c in frame.columns:
                out[c] = float(row[c])
            w.writerow(out)


def power_mean_from_frame(frame, sensor_name: str):
    if frame is None or len(frame) == 0:
        return math.nan, 0
    vals = np.asarray(frame[sensor_name], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return (float(np.mean(vals)) if len(vals) else math.nan), int(len(vals))


def save_rows_csv(path: Path, rows):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def load_rows_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return math.nan


def aggregate_prefix(
    latencies,
    cycle_rows,
    prefix,
    n,
    unique_correct,
    idle_power_w,
):
    lat = np.asarray(latencies[:prefix], dtype=np.float64).reshape(-1)
    cycle_means = np.asarray([to_float(r, "latency_mean_ms") for r in cycle_rows[:prefix]])
    fps = np.asarray([to_float(r, "throughput_fps") for r in cycle_rows[:prefix]])
    wall = np.asarray([to_float(r, "wall_time_s") for r in cycle_rows[:prefix]])
    proc_time = np.asarray([to_float(r, "process_time_s") for r in cycle_rows[:prefix]])

    def mean_finite(vals):
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else math.nan

    def sd_finite(vals):
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan

    mean_ms = float(np.mean(lat))
    cycle_mean_mean = mean_finite(cycle_means)
    cycle_mean_sd = sd_finite(cycle_means)
    ci_half = Z95 * cycle_mean_sd / math.sqrt(prefix) if math.isfinite(cycle_mean_sd) else math.nan

    fps_mean = mean_finite(fps)
    fps_sd = sd_finite(fps)
    fps_ci_half = Z95 * fps_sd / math.sqrt(prefix) if math.isfinite(fps_sd) else math.nan

    total_inf = prefix * n
    total_wall = float(np.sum(wall))
    global_fps = total_inf / total_wall

    active_powers = np.asarray([to_float(r, "active_power_w") for r in cycle_rows[:prefix]])
    dynamic_powers = np.asarray([to_float(r, "dynamic_power_w") for r in cycle_rows[:prefix]])
    energy_total = np.asarray([to_float(r, "energy_total_per_inf_mj") for r in cycle_rows[:prefix]])
    energy_dyn = np.asarray([to_float(r, "energy_dynamic_per_inf_mj") for r in cycle_rows[:prefix]])

    host_system = np.asarray([to_float(r, "system_cpu_percent_mean") for r in cycle_rows[:prefix]])
    host_proc = np.asarray([to_float(r, "process_cpu_percent_mean") for r in cycle_rows[:prefix]])
    host_proc_norm = np.asarray([to_float(r, "process_cpu_normalized_percent_mean") for r in cycle_rows[:prefix]])
    host_rss = np.asarray([to_float(r, "rss_mib_mean") for r in cycle_rows[:prefix]])
    host_freq = np.asarray([to_float(r, "cpu_freq_mhz_mean") for r in cycle_rows[:prefix]])
    host_temp = np.asarray([to_float(r, "temperature_max_c_mean") for r in cycle_rows[:prefix]])

    low, high = wilson_interval(unique_correct, n)

    return {
        "unique_images": n,
        "cycles": prefix,
        "timed_inferences": total_inf,
        "correct_unique": unique_correct,
        "accuracy": unique_correct / n,
        "accuracy_percent": 100.0 * unique_correct / n,
        "accuracy_wilson95_low_percent": 100.0 * low,
        "accuracy_wilson95_high_percent": 100.0 * high,
        "latency_mean_ms": mean_ms,
        "latency_mean_cycle_ci95_low_ms": cycle_mean_mean - ci_half,
        "latency_mean_cycle_ci95_high_ms": cycle_mean_mean + ci_half,
        "latency_median_ms": float(np.median(lat)),
        "latency_std_ms": float(np.std(lat, ddof=1)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "latency_min_ms": float(np.min(lat)),
        "latency_max_ms": float(np.max(lat)),
        "throughput_mean_fps": fps_mean,
        "throughput_std_fps": fps_sd,
        "throughput_ci95_low_fps": fps_mean - fps_ci_half,
        "throughput_ci95_high_fps": fps_mean + fps_ci_half,
        "throughput_global_fps": global_fps,
        "total_wall_time_s": total_wall,
        "process_time_total_s": float(np.sum(proc_time)),
        "process_time_per_inf_ms": float(np.sum(proc_time)) * 1000.0 / total_inf,
        "idle_power_w": idle_power_w,
        "active_power_w": mean_finite(active_powers),
        "dynamic_power_w": mean_finite(dynamic_powers),
        "energy_total_per_inf_mj": mean_finite(energy_total),
        "energy_dynamic_per_inf_mj": mean_finite(energy_dyn),
        "power_samples_total": int(sum(int(float(r.get("power_samples", 0))) for r in cycle_rows[:prefix])),
        "host_samples_total": int(sum(int(float(r.get("host_samples", 0))) for r in cycle_rows[:prefix])),
        "system_cpu_percent_mean": mean_finite(host_system),
        "process_cpu_percent_mean": mean_finite(host_proc),
        "process_cpu_normalized_percent_mean": mean_finite(host_proc_norm),
        "rss_mib_mean": mean_finite(host_rss),
        "arm_cpu_freq_mhz_mean": mean_finite(host_freq),
        "temperature_max_c_mean": mean_finite(host_temp),
    }


def write_markdown_summary(path: Path, scenario: str, summaries):
    lines = [
        f"# Resultados — {scenario}",
        "",
        "| Imagens | Ciclos | Inferências | Acc (%) | Lat média (ms) | Mediana | p95 | p99 | FPS médio | P idle (W) | P ativa (W) | P dinâmica (W) | E/inf (mJ) | E din/inf (mJ) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['unique_images']} | {r['cycles']} | {r['timed_inferences']} | "
            f"{r['accuracy_percent']:.4f} | {r['latency_mean_ms']:.4f} | "
            f"{r['latency_median_ms']:.4f} | {r['latency_p95_ms']:.4f} | "
            f"{r['latency_p99_ms']:.4f} | {r['throughput_mean_fps']:.3f} | "
            f"{r['idle_power_w']:.4f} | {r['active_power_w']:.4f} | "
            f"{r['dynamic_power_w']:.4f} | {r['energy_total_per_inf_mj']:.4f} | "
            f"{r['energy_dynamic_per_inf_mj']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_accuracy(
    dma,
    input_buffer,
    output_buffer,
    x_test,
    y_test,
    results_dir,
    warmup=100,
):
    pred_path = results_dir / "predictions_accuracy_10000.npy"
    json_path = results_dir / "accuracy_10000.json"
    conf_path = results_dir / "confusion_matrix.npy"

    if pred_path.exists() and json_path.exists() and conf_path.exists():
        print("[ACCURACY] reutilizando validação já salva.")
        preds = np.load(pred_path)
        acc = json.loads(json_path.read_text())
        return preds, acc

    print(f"[ACCURACY] warm-up funcional: {warmup}")
    for i in range(warmup):
        pack_image_u8(x_test[i % len(x_test)], input_buffer)
        transfer_once(dma, input_buffer, output_buffer)

    preds = np.empty(len(x_test), dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(len(x_test)):
        pack_image_u8(x_test[i], input_buffer)
        transfer_once(dma, input_buffer, output_buffer)
        logits = decode_logits(output_buffer)
        preds[i] = np.argmax(logits)
        if (i + 1) % 1000 == 0:
            partial = float(np.mean(preds[:i+1] == y_test[:i+1]))
            print(f"[ACCURACY] {i+1}/10000 | parcial={partial*100:.3f}%")

    elapsed = time.perf_counter() - t0
    correct = int(np.sum(preds == y_test))
    low, high = wilson_interval(correct, len(y_test))
    conf = confusion_matrix(y_test, preds)

    per_class = {}
    for c in range(N_CLASSES):
        total = int(conf[c].sum())
        per_class[str(c)] = {
            "correct": int(conf[c, c]),
            "total": total,
            "accuracy": float(conf[c, c] / total),
        }

    acc = {
        "unique_images": len(y_test),
        "correct": correct,
        "accuracy": correct / len(y_test),
        "accuracy_percent": 100.0 * correct / len(y_test),
        "wilson95_low_percent": 100.0 * low,
        "wilson95_high_percent": 100.0 * high,
        "elapsed_validation_s_not_benchmark": elapsed,
        "per_class": per_class,
    }
    np.save(pred_path, preds)
    np.save(conf_path, conf)
    json_path.write_text(json.dumps(acc, indent=2))
    print(
        f"[ACCURACY] {correct}/10000 = {100*correct/10000:.4f}% | "
        f"Wilson95=[{100*low:.4f}%, {100*high:.4f}%]"
    )
    return preds, acc


def run_scenario(
    scenario: str,
    dma,
    input_buffer,
    output_buffer,
    x_test,
    y_test,
    packed_test,
    subsets,
    reference_preds,
    root_results,
    power_sensor,
    power_sensor_name,
    idle_seconds,
    telemetry_interval,
    warmup,
    cycles,
    prefixes,
    restart,
    seed,
):
    scenario_dir = root_results / scenario
    if restart and scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== CENÁRIO: {scenario} ==========")
    print(f"Warm-up: {warmup} inferências")
    idx10k = subsets[10000]
    for i in range(warmup):
        idx = int(idx10k[i % len(idx10k)])
        if scenario == "inference_only":
            input_buffer[:] = packed_test[idx]
            transfer_once(dma, input_buffer, output_buffer)
            _ = int(np.argmax(decode_logits(output_buffer)))
        else:
            pack_image_u8(x_test[idx], input_buffer)
            transfer_once(dma, input_buffer, output_buffer)
            _ = int(np.argmax(decode_logits(output_buffer)))

    # Baseline idle após warm-up, com overlay carregado e DMA parado.
    baseline_rec = DataRecorder(power_sensor)
    print(f"Baseline idle: {idle_seconds:.1f}s @ {telemetry_interval:.3f}s")
    with baseline_rec.record(telemetry_interval):
        time.sleep(idle_seconds)
    baseline_frame = baseline_rec.frame.copy()
    baseline_frame.to_csv(scenario_dir / "power_idle_raw.csv")
    idle_power_w, idle_samples = power_mean_from_frame(baseline_frame, power_sensor_name)
    print(f"Idle power ({power_sensor_name}): {idle_power_w:.4f} W ({idle_samples} amostras)")

    all_summaries = []

    for n in (100, 1000, 10000):
        indices = subsets[n]
        lat_path = scenario_dir / f"latencias_n{n}.npy"
        cycles_csv = scenario_dir / f"passagens_n{n}.csv"
        power_csv = scenario_dir / f"power_samples_n{n}.csv"
        host_csv = scenario_dir / f"host_samples_n{n}.csv"

        if lat_path.exists() and cycles_csv.exists() and not restart:
            latencies = np.load(lat_path, mmap_mode="r+")
            cycle_rows = load_rows_csv(cycles_csv)
            start_cycle = len(cycle_rows)
            if latencies.shape != (cycles, n):
                raise RuntimeError(
                    f"Checkpoint incompatível {lat_path}: {latencies.shape}, "
                    f"esperado {(cycles, n)}. Use --restart."
                )
        else:
            latencies = np.lib.format.open_memmap(
                lat_path, mode="w+", dtype=np.float32, shape=(cycles, n)
            )
            latencies[:] = np.nan
            latencies.flush()
            cycle_rows = []
            start_cycle = 0

        unique_correct = int(np.sum(reference_preds[indices] == y_test[indices]))

        print(
            f"\n[{scenario}] N={n} | acurácia única={100*unique_correct/n:.4f}% | "
            f"retomando no ciclo {start_cycle + 1 if start_cycle < cycles else 'concluído'}"
        )

        for cycle in range(start_cycle, cycles):
            # Permutação determinística diferente em cada ciclo.
            rng = np.random.default_rng(seed + n * 1000 + cycle)
            order = indices[rng.permutation(n)]

            correct = 0
            mismatches_vs_reference = 0

            recorder = DataRecorder(power_sensor)
            host = HostTelemetry(telemetry_interval)
            host.start()

            process_t0 = time.process_time()
            with recorder.record(telemetry_interval):
                wall_t0 = time.perf_counter()

                for j, idx_val in enumerate(order):
                    idx = int(idx_val)

                    if scenario == "inference_only":
                        # Preparação/copiar input pré-quantizado fora da latência individual.
                        input_buffer[:] = packed_test[idx]

                        t0 = time.perf_counter_ns()
                        transfer_once(dma, input_buffer, output_buffer)
                        t1 = time.perf_counter_ns()

                        # Materialização/argmax fora da latência individual,
                        # mas dentro do tempo de parede do ciclo.
                        logits = decode_logits(output_buffer)
                        pred = int(np.argmax(logits))
                    else:
                        # End-to-end: normalização + packing + DMA + decode + argmax.
                        t0 = time.perf_counter_ns()
                        pack_image_u8(x_test[idx], input_buffer)
                        transfer_once(dma, input_buffer, output_buffer)
                        logits = decode_logits(output_buffer)
                        pred = int(np.argmax(logits))
                        t1 = time.perf_counter_ns()

                    latencies[cycle, j] = (t1 - t0) / 1_000_000.0
                    correct += int(pred == int(y_test[idx]))
                    mismatches_vs_reference += int(pred != int(reference_preds[idx]))

                wall_t1 = time.perf_counter()

            process_t1 = time.process_time()
            host.stop()

            if dma.sendchannel.error or dma.recvchannel.error:
                raise RuntimeError(f"DMA em erro após ciclo {cycle+1}: {dma_status(dma)}")
            if mismatches_vs_reference != 0:
                raise RuntimeError(
                    f"Determinismo violado: {mismatches_vs_reference} predições "
                    f"divergiram da validação de referência no ciclo {cycle+1}."
                )

            power_frame = recorder.frame.copy()
            active_power_w, power_samples = power_mean_from_frame(
                power_frame, power_sensor_name
            )
            wall_time_s = wall_t1 - wall_t0
            process_time_s = process_t1 - process_t0
            throughput_fps = n / wall_time_s
            dynamic_power_w = max(active_power_w - idle_power_w, 0.0)
            energy_total_per_inf_mj = active_power_w * wall_time_s * 1000.0 / n
            energy_dynamic_per_inf_mj = dynamic_power_w * wall_time_s * 1000.0 / n

            hsum = host.summary()
            lat_cycle = np.asarray(latencies[cycle], dtype=np.float64)

            row = {
                "cycle": cycle + 1,
                "n": n,
                "correct": correct,
                "mismatches_vs_reference": mismatches_vs_reference,
                "wall_time_s": wall_time_s,
                "process_time_s": process_time_s,
                "process_time_per_inf_ms": process_time_s * 1000.0 / n,
                "throughput_fps": throughput_fps,
                "latency_mean_ms": float(np.mean(lat_cycle)),
                "latency_median_ms": float(np.median(lat_cycle)),
                "latency_std_ms": float(np.std(lat_cycle, ddof=1)) if n > 1 else 0.0,
                "latency_p95_ms": float(np.percentile(lat_cycle, 95)),
                "latency_p99_ms": float(np.percentile(lat_cycle, 99)),
                "latency_min_ms": float(np.min(lat_cycle)),
                "latency_max_ms": float(np.max(lat_cycle)),
                "idle_power_w": idle_power_w,
                "active_power_w": active_power_w,
                "dynamic_power_w": dynamic_power_w,
                "energy_total_per_inf_mj": energy_total_per_inf_mj,
                "energy_dynamic_per_inf_mj": energy_dynamic_per_inf_mj,
                "power_samples": power_samples,
            }
            row.update(hsum)
            cycle_rows.append(row)

            latencies.flush()
            save_rows_csv(cycles_csv, cycle_rows)
            append_power_frame(power_csv, scenario, n, cycle + 1, power_frame)
            append_host_samples(host_csv, scenario, n, cycle + 1, host.samples)

            print(
                f"[{scenario}] n={n:5d} ciclo={cycle+1:3d}/{cycles} | "
                f"lat={row['latency_mean_ms']:.4f} ms | "
                f"FPS={throughput_fps:.2f} | "
                f"P={active_power_w:.3f} W | "
                f"Pdyn={dynamic_power_w:.3f} W | "
                f"E={energy_total_per_inf_mj:.3f} mJ"
            )

        # Agrega prefixos cumulativos 10/20/50/100.
        cycle_rows = load_rows_csv(cycles_csv)
        latencies_ro = np.load(lat_path, mmap_mode="r")
        summaries = []
        for p in prefixes:
            if p <= len(cycle_rows):
                r = aggregate_prefix(
                    latencies_ro, cycle_rows, p, n, unique_correct, idle_power_w
                )
                r["scenario"] = scenario
                summaries.append(r)
                all_summaries.append(r)

        summary_csv = scenario_dir / f"resultados_n{n}.csv"
        save_rows_csv(summary_csv, summaries)

    # 12 linhas = 3 tamanhos x 4 prefixos
    save_rows_csv(scenario_dir / "resultados.csv", all_summaries)
    write_markdown_summary(scenario_dir / "RESULTADOS.md", scenario, all_summaries)
    return all_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="/home/xilinx/jupyter_notebooks/resnet8_hls_ip11",
        help="Pasta contendo .bit/.hwh e cifar10_test_uint8.npz",
    )
    ap.add_argument("--bit", default="resnet8_hls_ip11.bit")
    ap.add_argument("--dataset", default="cifar10_test_uint8.npz")
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--idle-seconds", type=float, default=5.0)
    ap.add_argument("--telemetry-interval", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--power-rail", default="auto")
    ap.add_argument(
        "--scenario",
        choices=["inference_only", "end_to_end", "both"],
        default="both",
    )
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--list-rails", action="store_true")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    bit = base / args.bit
    hwh = bit.with_suffix(".hwh")
    dataset_path = base / args.dataset
    results_root = base / "resultados_robustos"

    for p in (bit, hwh, dataset_path):
        if not p.exists():
            raise FileNotFoundError(p)

    rails, rail_info = available_power_rails()
    print("===== RAILS PMBus =====")
    for name, info in rail_info.items():
        print(name, info)

    if args.list_rails:
        return

    primary_rail_name = choose_power_rail(rails, args.power_rail)
    power_sensor = rails[primary_rail_name].power
    power_sensor_name = power_sensor.name
    print(f"\nRail primário para custo energético: {primary_rail_name}")
    print(f"Sensor: {power_sensor_name} | leitura atual={power_sensor.value:.4f} W")

    if args.restart and results_root.exists():
        shutil.rmtree(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    print("\n===== ARTEFATOS =====")
    print("BIT :", bit)
    print("HWH :", hwh)
    print("DATA:", dataset_path)

    print("\n===== CARREGANDO OVERLAY =====")
    overlay = Overlay(str(bit), download=True)
    if "axi_dma_0" not in overlay.ip_dict:
        raise RuntimeError("axi_dma_0 ausente no HWH")
    dma = overlay.axi_dma_0
    print("DMA:", dma_status(dma))

    data = np.load(dataset_path)
    x_test = data["x"]
    y_test = data["y"].reshape(-1)
    if x_test.shape != (10000, 32, 32, 3) or x_test.dtype != np.uint8:
        raise RuntimeError(f"x_test inesperado: {x_test.shape} {x_test.dtype}")
    if y_test.shape != (10000,):
        raise RuntimeError(f"y_test inesperado: {y_test.shape}")
    if not np.all(np.bincount(y_test, minlength=10) == 1000):
        raise RuntimeError("CIFAR-10 de teste não está balanceado em 1000 imagens/classe")

    subsets = make_nested_stratified_indices(y_test, args.seed)
    np.save(results_root / "indices_n100.npy", subsets[100])
    np.save(results_root / "indices_n1000.npy", subsets[1000])
    np.save(results_root / "indices_n10000.npy", subsets[10000])

    input_buffer = allocate(shape=(N_PIXELS, 4), dtype=np.uint32)
    output_buffer = allocate(shape=(16,), dtype=np.uint32)

    try:
        # Smoke test do build atual.
        pack_image_u8(x_test[0], input_buffer)
        output_buffer[:] = 0
        transfer_once(dma, input_buffer, output_buffer)
        logits = decode_logits(output_buffer)
        pred0 = int(np.argmax(logits))
        padding = np.asarray(output_buffer[10:16], dtype=np.uint32).copy()
        if np.any(padding != 0):
            raise RuntimeError(f"Padding 320->512 não zero: {padding}")
        if dma.sendchannel.error or dma.recvchannel.error:
            raise RuntimeError(f"DMA em erro no smoke test: {dma_status(dma)}")
        print(
            f"Smoke: y0={int(y_test[0])} pred0={pred0} | "
            f"padding={padding.tolist()} | DMA={dma_status(dma)}"
        )

        # Metadados estáticos.
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": "AMD/Xilinx ZCU104",
            "accelerator": "ResNet8 hls4ml IP 1.1",
            "bitstream": str(bit),
            "hwh": str(hwh),
            "dataset": str(dataset_path),
            "bitstream_sha256": sha256_file(bit),
            "hwh_sha256": sha256_file(hwh),
            "dataset_sha256": sha256_file(dataset_path),
            "python": sys.version,
            "numpy": np.__version__,
            "pynq": __import__("pynq").__version__,
            "kernel": platform.release(),
            "machine": platform.machine(),
            "batch": 1,
            "seed": args.seed,
            "warmup": args.warmup,
            "cycles": args.cycles,
            "sizes": [100, 1000, 10000],
            "prefixes": [10, 20, 50, 100],
            "telemetry_interval_s": args.telemetry_interval,
            "idle_baseline_s": args.idle_seconds,
            "power_scope": (
                f"rail '{primary_rail_name}' ({power_sensor_name}); "
                "interpretar como potência medida no escopo desse rail"
            ),
            "power_rail_info_start": rail_info,
            "precision": "ap_fixed<22,12,AP_RND_CONV,AP_SAT>",
            "fixed_fractional_bits": 10,
            "fixed_scale": 1024,
            "input_dma_bytes_per_image": INPUT_BYTES,
            "output_dma_bytes_per_image": OUTPUT_BYTES,
            "pl_clock_mhz": 100.0,
            "timing_post_route_wns_ns": 0.122,
            "timing_post_route_whs_ns": 0.009,
            "timing_post_route_tns_ns": 0.0,
            "timing_post_route_ths_ns": 0.0,
            "inference_only_boundary": (
                "input já quantizado/empacotado e copiado para o PynqBuffer; "
                "cronômetro cobre programação DMA + MM2S/DDR + acelerador + "
                "S2MM/DDR + polling/wait; decode/argmax fora da latência individual"
            ),
            "end_to_end_boundary": (
                "cronômetro cobre uint8 CIFAR-10 em RAM -> normalização -> "
                "quantização/packing -> DMA -> acelerador -> retorno -> decode -> argmax; "
                "exclui leitura de arquivo, load do overlay e inicialização"
            ),
            "note_subset_comparability": (
                "10k cobre o test set oficial inteiro e é a comparação principal. "
                "Os subconjuntos 100/1000 são estratificados/aninhados e reprodutíveis "
                "neste script; para comparação bit-a-bit com subconjuntos antigos, "
                "usar os mesmos índices salvos pelos ensaios antigos."
            ),
        }
        (results_root / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Acurácia é separada do benchmark de desempenho.
        reference_preds, accuracy = run_accuracy(
            dma,
            input_buffer,
            output_buffer,
            x_test,
            y_test,
            results_root,
            warmup=args.warmup,
        )

        print("\n===== PRÉ-PACKING PARA INFERENCE-ONLY =====")
        t_pack = time.perf_counter()
        packed_test = pack_dataset_uint8(x_test)
        print(
            f"Packed: {packed_test.nbytes / (1024**2):.2f} MiB | "
            f"{time.perf_counter() - t_pack:.2f}s"
        )

        scenarios = (
            ["inference_only", "end_to_end"]
            if args.scenario == "both"
            else [args.scenario]
        )
        all_results = []
        for scenario in scenarios:
            summaries = run_scenario(
                scenario=scenario,
                dma=dma,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                x_test=x_test,
                y_test=y_test,
                packed_test=packed_test,
                subsets=subsets,
                reference_preds=reference_preds,
                root_results=results_root,
                power_sensor=power_sensor,
                power_sensor_name=power_sensor_name,
                idle_seconds=args.idle_seconds,
                telemetry_interval=args.telemetry_interval,
                warmup=args.warmup,
                cycles=args.cycles,
                prefixes=[p for p in (10, 20, 50, 100) if p <= args.cycles],
                restart=args.restart,
                seed=args.seed,
            )
            all_results.extend(summaries)

        save_rows_csv(results_root / "resultados_todos.csv", all_results)
        (results_root / "final_status.json").write_text(
            json.dumps(
                {
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "accuracy": accuracy,
                    "dma_final": dma_status(dma),
                    "primary_power_rail": primary_rail_name,
                    "primary_power_sensor": power_sensor_name,
                    "result_rows": len(all_results),
                    "status": "complete",
                },
                indent=2,
            )
        )
        print("\n========== BENCHMARK CONCLUÍDO ==========")
        print("Resultados:", results_root)
        print("DMA final:", dma_status(dma))
        print(
            "Resultado principal para comparação CPU/GPU: "
            "unique_images=10000, cycles=100, por cenário."
        )

    finally:
        try:
            input_buffer.close()
        except Exception:
            pass
        try:
            output_buffer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
