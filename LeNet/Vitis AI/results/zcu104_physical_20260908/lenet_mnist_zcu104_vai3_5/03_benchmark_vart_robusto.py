#!/usr/bin/env python3
"""
Benchmark robusto LeNet/MNIST no Vitis AI / ZCU104.

Campanha congelada:
  - inference_only: 1, 2, 3, 4 threads/runners
  - end_to_end:     1, 2, 3, 4 threads/runners
  - saturated:      1, 2, 3, 4 threads/runners
  - 10.000 inferencias TOTAIS por configuracao
  - batch 1
  - 100 warm-ups por runner, excluidos
  - 10 blocos de 1.000 inferencias para IC95/estabilidade
  - um vart.Runner independente por thread
  - nenhuma remocao de outliers

Fronteiras:
  inference_only:
    entrada INT8 pre-quantizada e copiada ao buffer antes de t0;
    t0 -> execute_async + wait -> t1;
    argmax e armazenamento ficam fora da latencia individual.

  end_to_end:
    t0 -> uint8->float32 -> /255 -> quantizacao INT8 -> copia de entrada ->
    execute_async + wait -> dequantizacao -> argmax -> t1.

  saturated:
    mesma entrada INT8 e buffers reutilizados;
    t0 -> execute_async + wait -> t1;
    sem preprocessamento/argmax dentro da regiao medida.

Telemetria:
  - INA226: /sys/class/hwmon/hwmon0/{power1_input,curr1_input,in2_input}
  - AMS/IIO: PS, PL e remote temperature quando disponiveis
  - sampler paralelo; intervalo padrao 10 ms
  - baseline idle antes de cada configuracao
  - energia por integracao trapezoidal da potencia
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
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Z95 = 1.959963984540054
EXPECTED_XMODEL_SHA256 = "bfef577b297939e19cabf74b1ab8d30ab65d694cd6a1ec9b28197577a3ce52c5"
EXPECTED_DATASET_SHA256 = "be1c7a4955d7da22ea2fa55f9c0350a34f33cdee1697a6da500153c2eafa73cd"
EXPECTED_REFERENCE_SHA256 = "cb7f13cff28ae247001a638d7635dea8fd7066d12078dd049beb93bd856b0d32"
EXPECTED_SELECTED_INDEX_SHA256_N10000 = "65a5248b6e84b1b5bc6bd1068aba612e7884ac5c781f686d75d3843cc7ebd8a9"
EXPECTED_HOST_PRED_SHA256_OFFICIAL = "eaa5a7b4bad2e55523a42b6555292efbc15c34545a0fa0d17a66eabf68b5f250"

DEFAULT_POWER_PATH = Path("/sys/class/hwmon/hwmon0/power1_input")
DEFAULT_CURRENT_PATH = Path("/sys/class/hwmon/hwmon0/curr1_input")
DEFAULT_VOLTAGE_PATH = Path("/sys/class/hwmon/hwmon0/in2_input")
DEFAULT_AMS = Path("/sys/bus/iio/devices/iio:device0")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None


def read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def read_float(path: Path) -> Optional[float]:
    try:
        return float(path.read_text().strip())
    except Exception:
        return None


def run_command(cmd: Sequence[str], timeout: float = 30.0) -> Dict[str, Any]:
    try:
        cp = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"command": list(cmd), "returncode": cp.returncode, "output": cp.stdout}
    except Exception as exc:
        return {"command": list(cmd), "returncode": None, "output": repr(exc)}


def proc_status() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(errors="replace").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "Threads:")):
                k, v = line.split(":", 1)
                result[k] = v.strip()
    try:
        result["loadavg_1_5_15"] = list(os.getloadavg())
    except Exception:
        result["loadavg_1_5_15"] = None
    return result


def cpu_governors_and_freqs() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cpufreq = cpu / "cpufreq"
        if not cpufreq.exists():
            continue
        out[cpu.name] = {
            "governor": read_text(cpufreq / "scaling_governor"),
            "cur_freq_khz": read_int(cpufreq / "scaling_cur_freq"),
            "min_freq_khz": read_int(cpufreq / "scaling_min_freq"),
            "max_freq_khz": read_int(cpufreq / "scaling_max_freq"),
        }
    return out


def wilson_interval(correct: int, n: int, z: float = Z95) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = correct / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (center, min(1.0, center + half)) if center - half < 0 else (center - half, min(1.0, center + half))


def finite_stats(values: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    if not np.all(np.isfinite(x)):
        raise RuntimeError("estatistica recebeu NaN/Inf")
    if np.any(x <= 0):
        raise RuntimeError("latencia/valor <= 0 encontrado")
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return {
        "count": int(x.size),
        "mean": mean,
        "median": float(np.median(x)),
        "sample_sd_ddof1": sd,
        "cv_percent": float(100.0 * sd / mean) if mean != 0 else None,
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def ci95_from_observations(values: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    mean = float(np.mean(x))
    if x.size == 1:
        return {"n": 1, "mean": mean, "sample_sd": 0.0, "low": mean, "high": mean}
    sd = float(np.std(x, ddof=1))
    half = Z95 * sd / math.sqrt(x.size)
    return {"n": int(x.size), "mean": mean, "sample_sd": sd, "low": mean - half, "high": mean + half}


def throughput_stats(block_fps: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(block_fps, dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    out = {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sample_sd_ddof1": sd,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    out["ci95_mean"] = ci95_from_observations(x.tolist())
    return out


def make_selected_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    labels = labels.reshape(-1)
    if labels.size != 10000:
        raise RuntimeError(f"dataset esperado com 10000 labels, encontrado {labels.size}")
    rng = np.random.default_rng(seed)
    groups = []
    for class_id in range(10):
        idx = np.flatnonzero(labels == class_id)
        rng.shuffle(idx)
        groups.append(idx)
    balanced_thousand = np.stack([g[:100] for g in groups], axis=1).reshape(-1)
    remainder = np.concatenate([g[100:] for g in groups])
    rng.shuffle(remainder)
    selected = np.concatenate([balanced_thousand, remainder]).astype(np.int64, copy=False)
    if selected.size != 10000 or np.unique(selected).size != 10000:
        raise RuntimeError("indices N=10000 invalidos")
    return np.ascontiguousarray(selected)


def make_run_order(selected: np.ndarray, seed: int) -> np.ndarray:
    # Reaproveita a semente de permutacao definida no README para N=10000.
    rng_cycle = np.random.default_rng(seed + 10000 * 1009)
    perm = rng_cycle.permutation(10000)
    return np.ascontiguousarray(selected[perm], dtype=np.int64)


def split_block_positions(block: int, threads: int, block_size: int = 1000) -> List[np.ndarray]:
    start = block * block_size
    stop = start + block_size
    base = block_size // threads
    rem = block_size % threads
    counts = [base] * threads
    # Rotaciona os extras entre workers para balancear o total ao longo de 10 blocos.
    for r in range(rem):
        counts[(block + r) % threads] += 1
    chunks = []
    p = start
    for c in counts:
        chunks.append(np.arange(p, p + c, dtype=np.int64))
        p += c
    if p != stop:
        raise AssertionError("particao de bloco inconsistente")
    return chunks


def ams_temp_c(channel_prefix: str, ams_dir: Path = DEFAULT_AMS) -> Optional[float]:
    raw = read_float(ams_dir / f"{channel_prefix}_raw")
    offset = read_float(ams_dir / f"{channel_prefix}_offset")
    scale = read_float(ams_dir / f"{channel_prefix}_scale")
    if raw is None or offset is None or scale is None:
        return None
    # AMS nesta imagem expoe o valor processado em miligraus C.
    return (raw + offset) * scale / 1000.0


def read_temperatures() -> Dict[str, Optional[float]]:
    return {
        "ps_temp_c": ams_temp_c("in_temp0_ps_temp"),
        "remote_temp_c": ams_temp_c("in_temp1_remote_temp"),
        "pl_temp_c": ams_temp_c("in_temp2_pl_temp"),
    }


@dataclass
class TelemetrySample:
    perf_ns: int
    utc_ns: int
    power_w: Optional[float]
    current_a: Optional[float]
    voltage_v: Optional[float]
    ps_temp_c: Optional[float]
    pl_temp_c: Optional[float]
    remote_temp_c: Optional[float]
    read_duration_ns: int


class TelemetrySampler:
    def __init__(self, interval_s: float = 0.01, temp_every: int = 10):
        self.interval_s = float(interval_s)
        self.temp_every = max(1, int(temp_every))
        self.samples: List[TelemetrySample] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_one(self, i: int) -> TelemetrySample:
        t0 = time.perf_counter_ns()
        utc = time.time_ns()
        p = read_int(DEFAULT_POWER_PATH)
        c = read_int(DEFAULT_CURRENT_PATH)
        v = read_int(DEFAULT_VOLTAGE_PATH)
        temps = {"ps_temp_c": None, "pl_temp_c": None, "remote_temp_c": None}
        if i % self.temp_every == 0:
            temps = read_temperatures()
        t1 = time.perf_counter_ns()
        return TelemetrySample(
            perf_ns=t0,
            utc_ns=utc,
            power_w=(p / 1e6) if p is not None else None,
            current_a=(c / 1000.0) if c is not None else None,
            voltage_v=(v / 1000.0) if v is not None else None,
            ps_temp_c=temps["ps_temp_c"],
            pl_temp_c=temps["pl_temp_c"],
            remote_temp_c=temps["remote_temp_c"],
            read_duration_ns=t1 - t0,
        )

    def _loop(self) -> None:
        next_t = time.perf_counter()
        i = 0
        while not self._stop.is_set():
            try:
                self.samples.append(self._read_one(i))
            except Exception:
                self.errors.append(traceback.format_exc())
            i += 1
            next_t += self.interval_s
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.perf_counter()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler ja iniciado")
        self._thread = threading.Thread(target=self._loop, name="telemetry-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def as_rows(self) -> List[Dict[str, Any]]:
        return [s.__dict__.copy() for s in self.samples]


def summarize_telemetry(samples: List[TelemetrySample]) -> Dict[str, Any]:
    def vals(attr: str) -> np.ndarray:
        return np.asarray(
            [getattr(s, attr) for s in samples if getattr(s, attr) is not None],
            dtype=np.float64,
        )
    out: Dict[str, Any] = {"sample_count": len(samples)}
    for attr in ("power_w", "current_a", "voltage_v", "ps_temp_c", "pl_temp_c", "remote_temp_c"):
        x = vals(attr)
        if x.size:
            out[attr] = {
                "count": int(x.size),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "sample_sd_ddof1": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
        else:
            out[attr] = None
    read_ms = np.asarray([s.read_duration_ns / 1e6 for s in samples], dtype=np.float64)
    if read_ms.size:
        out["sampler_read_duration_ms"] = {
            "mean": float(np.mean(read_ms)),
            "median": float(np.median(read_ms)),
            "p95": float(np.percentile(read_ms, 95)),
            "max": float(np.max(read_ms)),
        }
    if len(samples) >= 2:
        dt_ms = np.diff(np.asarray([s.perf_ns for s in samples], dtype=np.float64)) / 1e6
        out["sampler_interval_ms"] = {
            "mean": float(np.mean(dt_ms)),
            "median": float(np.median(dt_ms)),
            "p95": float(np.percentile(dt_ms, 95)),
            "max": float(np.max(dt_ms)),
        }
    return out


def integrate_power(samples: List[TelemetrySample], start_ns: int, end_ns: int) -> Dict[str, Any]:
    if end_ns <= start_ns:
        raise ValueError("intervalo de energia invalido")
    valid = [s for s in samples if s.power_w is not None]
    if len(valid) < 2:
        return {
            "samples_in_interval": 0,
            "bracketed": False,
            "coverage_fraction": 0.0,
            "max_sample_gap_ms": None,
            "energy_j": None,
            "mean_power_w": None,
        }

    t = np.asarray([s.perf_ns for s in valid], dtype=np.float64)
    p = np.asarray([s.power_w for s in valid], dtype=np.float64)

    # Requer amostras que envolvam as duas bordas para integracao completa.
    bracketed = bool(t[0] <= start_ns and t[-1] >= end_ns)
    inside_mask = (t >= start_ns) & (t <= end_ns)
    inside_count = int(np.sum(inside_mask))

    first_inside = t[inside_mask][0] if inside_count else None
    last_inside = t[inside_mask][-1] if inside_count else None
    duration_ns = float(end_ns - start_ns)
    if inside_count >= 2:
        coverage = float((last_inside - first_inside) / duration_ns)
    else:
        coverage = 0.0

    # Inclui bordas interpoladas + todas as amostras internas.
    internal_t = t[inside_mask]
    internal_p = p[inside_mask]
    start_p = float(np.interp(float(start_ns), t, p))
    end_p = float(np.interp(float(end_ns), t, p))
    ti = np.concatenate(([float(start_ns)], internal_t, [float(end_ns)]))
    pi = np.concatenate(([start_p], internal_p, [end_p]))
    order = np.argsort(ti)
    ti = ti[order]
    pi = pi[order]
    # Remove timestamps duplicados.
    keep = np.concatenate(([True], np.diff(ti) > 0))
    ti = ti[keep]
    pi = pi[keep]
    ts = (ti - ti[0]) / 1e9
    energy_j = float(np.trapz(pi, ts))
    duration_s = (end_ns - start_ns) / 1e9
    gaps_ms = np.diff(ti) / 1e6 if ti.size >= 2 else np.asarray([], dtype=np.float64)

    return {
        "samples_in_interval": inside_count,
        "bracketed": bracketed,
        "coverage_fraction": coverage,
        "max_sample_gap_ms": float(np.max(gaps_ms)) if gaps_ms.size else None,
        "energy_j": energy_j,
        "mean_power_w": energy_j / duration_s,
    }


def save_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def save_telemetry_csv(path: Path, samples: List[TelemetrySample]) -> None:
    fields = [
        "perf_ns", "utc_ns", "power_w", "current_a", "voltage_v",
        "ps_temp_c", "pl_temp_c", "remote_temp_c", "read_duration_ns",
    ]
    save_csv(path, fields, [s.__dict__ for s in samples])


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def raw_output_distribution(raw: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(raw, dtype=np.int8)
    return {
        "count_values": int(x.size),
        "count_minus128": int(np.sum(x == -128)),
        "count_plus127": int(np.sum(x == 127)),
        "min": int(x.min()) if x.size else None,
        "max": int(x.max()) if x.size else None,
        "mean": float(x.astype(np.float64).mean()) if x.size else None,
        "sample_sd_ddof1": float(x.astype(np.float64).std(ddof=1)) if x.size > 1 else 0.0,
    }


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, nclasses: int = 10) -> np.ndarray:
    cm = np.zeros((nclasses, nclasses), dtype=np.int64)
    np.add.at(cm, (labels.astype(np.int64), preds.astype(np.int64)), 1)
    return cm


def verify_inputs(xmodel: Path, dataset: Path, reference: Path, strict_hashes: bool) -> Dict[str, str]:
    hashes = {
        "xmodel": sha256_file(xmodel),
        "dataset": sha256_file(dataset),
        "reference": sha256_file(reference),
    }
    expected = {
        "xmodel": EXPECTED_XMODEL_SHA256,
        "dataset": EXPECTED_DATASET_SHA256,
        "reference": EXPECTED_REFERENCE_SHA256,
    }
    if strict_hashes:
        bad = {k: (hashes[k], expected[k]) for k in hashes if hashes[k] != expected[k]}
        if bad:
            raise RuntimeError(f"hashes de entrada divergentes: {bad}")
    return hashes


def inspect_xmodel(xmodel: Path) -> Tuple[Any, Dict[str, Any]]:
    import xir
    import vart

    graph = xir.Graph.deserialize(str(xmodel))
    dpu = [
        sg for sg in graph.get_root_subgraph().toposort_child_subgraph()
        if sg.has_attr("device") and sg.get_attr("device") == "DPU"
    ]
    if len(dpu) != 1:
        raise RuntimeError(f"esperado 1 subgrafo DPU, encontrado {len(dpu)}")
    probe = vart.Runner.create_runner(dpu[0], "run")
    inputs = probe.get_input_tensors()
    outputs = probe.get_output_tensors()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"esperado 1 input/1 output, encontrados {len(inputs)}/{len(outputs)}")
    it, ot = inputs[0], outputs[0]
    input_shape = tuple(it.dims)
    output_shape = tuple(ot.dims)
    if input_shape != (1, 28, 28, 1):
        raise RuntimeError(f"input shape inesperado: {input_shape}")
    if output_shape != (1, 10):
        raise RuntimeError(f"output shape inesperado: {output_shape}")
    input_fix = int(it.get_attr("fix_point"))
    output_fix = int(ot.get_attr("fix_point"))
    if input_fix != 6 or output_fix != 2:
        raise RuntimeError(f"fix_points inesperados: input={input_fix}, output={output_fix}")
    meta = {
        "input_name": it.name,
        "input_shape": list(input_shape),
        "input_fix_point": input_fix,
        "output_name": ot.name,
        "output_shape": list(output_shape),
        "output_fix_point": output_fix,
        "batch": int(input_shape[0]),
    }
    # Mantemos graph vivo enquanto dpu for usado.
    return (graph, dpu[0]), meta


def prepare_quantized(images: np.ndarray, input_fix: int) -> np.ndarray:
    q = np.clip(
        np.rint((images.astype(np.float32) / 255.0) * (2 ** input_fix)),
        -128,
        127,
    ).astype(np.int8)
    return np.ascontiguousarray(q[..., np.newaxis])


@dataclass
class ScenarioConfig:
    scenario: str
    threads: int
    samples: int = 10000
    blocks: int = 10
    block_size: int = 1000
    warmup_per_runner: int = 100


def run_scenario(
    cfg: ScenarioConfig,
    dpu_subgraph: Any,
    tensor_meta: Dict[str, Any],
    images: np.ndarray,
    labels: np.ndarray,
    host_predictions: np.ndarray,
    host_logits: np.ndarray,
    run_order: np.ndarray,
    prequantized: np.ndarray,
    outdir: Path,
    idle_seconds: float,
    telemetry_interval: float,
) -> Dict[str, Any]:
    import vart

    if cfg.samples != 10000 or cfg.blocks * cfg.block_size != cfg.samples:
        raise RuntimeError("esta campanha exige exatamente 10 blocos x 1000 = 10000")
    if cfg.threads not in (1, 2, 3, 4):
        raise RuntimeError("threads deve ser 1,2,3,4")
    if cfg.scenario not in ("inference_only", "end_to_end", "saturated"):
        raise RuntimeError(f"cenario invalido: {cfg.scenario}")

    outdir.mkdir(parents=True, exist_ok=False)

    # Snapshot do estado dos cores antes da configuracao, fora da regiao medida.
    dpu_status_before = run_command(["xdputil", "status"])

    n = cfg.samples
    latencies_ns = np.zeros(n, dtype=np.int64)
    thread_id_for_pos = np.full(n, -1, dtype=np.int16)
    predictions = np.full(n, -1, dtype=np.int64)
    raw_outputs = np.zeros((n, 10), dtype=np.int8)
    worker_counts = np.zeros(cfg.threads, dtype=np.int64)

    block_start_ns = np.zeros(cfg.blocks, dtype=np.int64)
    block_end_ns = np.zeros(cfg.blocks, dtype=np.int64)
    block_start_utc_ns = np.zeros(cfg.blocks, dtype=np.int64)
    block_end_utc_ns = np.zeros(cfg.blocks, dtype=np.int64)

    errors: List[Dict[str, Any]] = []
    errors_lock = threading.Lock()
    ready_barrier = threading.Barrier(cfg.threads + 1)
    active_gate = threading.Event()

    def start_action_factory(b: int):
        def action():
            block_start_ns[b] = time.perf_counter_ns()
            block_start_utc_ns[b] = time.time_ns()
        return action

    def end_action_factory(b: int):
        def action():
            block_end_ns[b] = time.perf_counter_ns()
            block_end_utc_ns[b] = time.time_ns()
        return action

    start_barriers = [threading.Barrier(cfg.threads, action=start_action_factory(b)) for b in range(cfg.blocks)]
    end_barriers = [threading.Barrier(cfg.threads, action=end_action_factory(b)) for b in range(cfg.blocks)]

    def abort_barriers():
        for br in start_barriers + end_barriers:
            try:
                br.abort()
            except Exception:
                pass
        try:
            ready_barrier.abort()
        except Exception:
            pass
        active_gate.set()

    input_shape = tuple(tensor_meta["input_shape"])
    output_shape = tuple(tensor_meta["output_shape"])
    input_fix = int(tensor_meta["input_fix_point"])
    output_fix = int(tensor_meta["output_fix_point"])

    def worker(worker_id: int):
        try:
            runner = vart.Runner.create_runner(dpu_subgraph, "run")
            it = runner.get_input_tensors()[0]
            ot = runner.get_output_tensors()[0]
            if tuple(it.dims) != input_shape or tuple(ot.dims) != output_shape:
                raise RuntimeError(f"worker {worker_id}: tensor shape divergente")
            if int(it.get_attr("fix_point")) != input_fix or int(ot.get_attr("fix_point")) != output_fix:
                raise RuntimeError(f"worker {worker_id}: fix_point divergente")

            input_buffer = np.empty(input_shape, dtype=np.int8, order="C")
            output_buffer = np.empty(output_shape, dtype=np.int8, order="C")
            input_buffer[...] = prequantized[run_order[0]].reshape(input_shape)

            def execute_once():
                job = runner.execute_async([input_buffer], [output_buffer])
                status = runner.wait(job)
                if status not in (0, None):
                    raise RuntimeError(f"worker {worker_id}: runner.wait retornou {status}")

            for _ in range(cfg.warmup_per_runner):
                execute_once()

            ready_barrier.wait()
            active_gate.wait()

            for b in range(cfg.blocks):
                start_barriers[b].wait()
                chunks = split_block_positions(b, cfg.threads, cfg.block_size)
                positions = chunks[worker_id]

                if cfg.scenario == "saturated":
                    input_buffer[...] = prequantized[run_order[0]].reshape(input_shape)

                for pos_np in positions:
                    pos = int(pos_np)
                    thread_id_for_pos[pos] = worker_id

                    if cfg.scenario == "inference_only":
                        data_idx = int(run_order[pos])
                        input_buffer[...] = prequantized[data_idx].reshape(input_shape)
                        t0 = time.perf_counter_ns()
                        execute_once()
                        t1 = time.perf_counter_ns()
                        latencies_ns[pos] = t1 - t0
                        # Pos-processamento fora da fronteira individual.
                        raw = output_buffer.reshape(-1).copy()
                        raw_outputs[pos] = raw
                        predictions[pos] = int(np.argmax(raw))

                    elif cfg.scenario == "end_to_end":
                        data_idx = int(run_order[pos])
                        image = images[data_idx]
                        t0 = time.perf_counter_ns()
                        f32 = image.astype(np.float32) / 255.0
                        q = np.clip(
                            np.rint(f32 * (2 ** input_fix)),
                            -128,
                            127,
                        ).astype(np.int8)
                        input_buffer[...] = q.reshape(input_shape)
                        execute_once()
                        logits = output_buffer.reshape(-1).astype(np.float32) * (2.0 ** (-output_fix))
                        pred = int(np.argmax(logits))
                        t1 = time.perf_counter_ns()
                        latencies_ns[pos] = t1 - t0
                        raw_outputs[pos] = output_buffer.reshape(-1)
                        predictions[pos] = pred

                    else:  # saturated
                        t0 = time.perf_counter_ns()
                        execute_once()
                        t1 = time.perf_counter_ns()
                        latencies_ns[pos] = t1 - t0

                    worker_counts[worker_id] += 1

                end_barriers[b].wait()

        except BaseException as exc:
            with errors_lock:
                errors.append({
                    "worker_id": worker_id,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                })
            abort_barriers()

    threads: List[threading.Thread] = []
    for wid in range(cfg.threads):
        th = threading.Thread(target=worker, args=(wid,), name=f"vart-worker-{wid}")
        threads.append(th)
        th.start()

    # Espera criacao dos runners + warm-up de todos.
    try:
        ready_barrier.wait()
    except threading.BrokenBarrierError:
        for th in threads:
            th.join(timeout=5.0)
        raise RuntimeError(f"falha antes do baseline: {errors}")

    # DPU/runner inicializados; agora mede idle.
    idle_sampler = TelemetrySampler(interval_s=telemetry_interval)
    idle_sampler.start()
    time.sleep(idle_seconds)
    idle_sampler.stop()
    idle_summary = summarize_telemetry(idle_sampler.samples)
    idle_power_w = None
    if idle_summary.get("power_w"):
        idle_power_w = float(idle_summary["power_w"]["mean"])

    # Telemetria ativa com amostras antes/depois das bordas medidas.
    active_sampler = TelemetrySampler(interval_s=telemetry_interval)
    active_sampler.start()
    time.sleep(max(0.05, 3.0 * telemetry_interval))
    active_gate.set()

    for th in threads:
        th.join()

    time.sleep(max(0.05, 3.0 * telemetry_interval))
    active_sampler.stop()

    if errors:
        json_dump(outdir / "ERROR.json", {"errors": errors})
        save_telemetry_csv(outdir / "telemetry_active_partial.csv", active_sampler.samples)
        raise RuntimeError(f"erro em worker(s): {errors}")

    if np.any(latencies_ns <= 0):
        bad = np.flatnonzero(latencies_ns <= 0)
        raise RuntimeError(f"latencias invalidas em {bad[:20].tolist()}")
    if np.any(thread_id_for_pos < 0):
        raise RuntimeError("posicoes sem worker")
    if int(worker_counts.sum()) != n:
        raise RuntimeError(f"contagem total incorreta: {worker_counts.sum()} != {n}")

    scenario_start_ns = int(block_start_ns[0])
    scenario_end_ns = int(block_end_ns[-1])
    wall_s = (scenario_end_ns - scenario_start_ns) / 1e9
    if wall_s <= 0:
        raise RuntimeError("wall time invalido")

    latency_ms = latencies_ns.astype(np.float64) / 1e6
    block_rows: List[Dict[str, Any]] = []
    block_means_ms: List[float] = []
    block_fps: List[float] = []

    for b in range(cfg.blocks):
        a = b * cfg.block_size
        z = a + cfg.block_size
        block_lat = latency_ms[a:z]
        dur_s = (int(block_end_ns[b]) - int(block_start_ns[b])) / 1e9
        fps = cfg.block_size / dur_s
        block_means_ms.append(float(np.mean(block_lat)))
        block_fps.append(float(fps))
        st = finite_stats(block_lat)
        pwr = integrate_power(active_sampler.samples, int(block_start_ns[b]), int(block_end_ns[b]))
        dynamic_energy_j = None
        dynamic_power_w = None
        total_mj_per_inf = None
        dyn_mj_per_inf = None
        if pwr["energy_j"] is not None:
            total_mj_per_inf = 1000.0 * pwr["energy_j"] / cfg.block_size
            if idle_power_w is not None:
                dynamic_power_w = max(float(pwr["mean_power_w"]) - idle_power_w, 0.0)
                dynamic_energy_j = max(float(pwr["energy_j"]) - idle_power_w * dur_s, 0.0)
                dyn_mj_per_inf = 1000.0 * dynamic_energy_j / cfg.block_size

        block_rows.append({
            "scenario": cfg.scenario,
            "threads": cfg.threads,
            "block": b + 1,
            "inferences": cfg.block_size,
            "start_perf_ns": int(block_start_ns[b]),
            "end_perf_ns": int(block_end_ns[b]),
            "start_utc_ns": int(block_start_utc_ns[b]),
            "end_utc_ns": int(block_end_utc_ns[b]),
            "wall_time_s": dur_s,
            "fps": fps,
            "latency_mean_ms": st["mean"],
            "latency_median_ms": st["median"],
            "latency_sd_ms": st["sample_sd_ddof1"],
            "latency_cv_percent": st["cv_percent"],
            "latency_p90_ms": st["p90"],
            "latency_p95_ms": st["p95"],
            "latency_p99_ms": st["p99"],
            "latency_min_ms": st["min"],
            "latency_max_ms": st["max"],
            "power_samples": pwr["samples_in_interval"],
            "power_bracketed": pwr["bracketed"],
            "power_coverage_fraction": pwr["coverage_fraction"],
            "power_max_sample_gap_ms": pwr["max_sample_gap_ms"],
            "active_mean_power_w": pwr["mean_power_w"],
            "idle_mean_power_w": idle_power_w,
            "dynamic_mean_power_w": dynamic_power_w,
            "total_energy_j": pwr["energy_j"],
            "dynamic_energy_j": dynamic_energy_j,
            "total_energy_mj_per_inference": total_mj_per_inf,
            "dynamic_energy_mj_per_inference": dyn_mj_per_inf,
        })

    overall_power = integrate_power(active_sampler.samples, scenario_start_ns, scenario_end_ns)
    overall_dynamic_power = None
    overall_dynamic_energy = None
    total_energy_mj_per_inf = None
    dynamic_energy_mj_per_inf = None
    if overall_power["energy_j"] is not None:
        total_energy_mj_per_inf = 1000.0 * float(overall_power["energy_j"]) / n
        if idle_power_w is not None:
            overall_dynamic_power = max(float(overall_power["mean_power_w"]) - idle_power_w, 0.0)
            overall_dynamic_energy = max(float(overall_power["energy_j"]) - idle_power_w * wall_s, 0.0)
            dynamic_energy_mj_per_inf = 1000.0 * overall_dynamic_energy / n

    overall_latency = finite_stats(latency_ms)
    latency_ci = ci95_from_observations(block_means_ms)
    tp_stats = throughput_stats(block_fps)
    fps_global = n / wall_s

    per_thread: List[Dict[str, Any]] = []
    for wid in range(cfg.threads):
        mask = thread_id_for_pos == wid
        l = latency_ms[mask]
        per_thread.append({
            "thread_id": wid,
            "count": int(mask.sum()),
            "latency_ms": finite_stats(l) if l.size else {"count": 0},
            "nominal_fps_from_sum_latency": float(mask.sum() / (np.sum(l) / 1000.0)) if l.size else None,
        })

    accuracy: Optional[Dict[str, Any]] = None
    if cfg.scenario in ("inference_only", "end_to_end"):
        run_labels = labels[run_order]
        run_host = host_predictions[run_order]
        correct = int(np.sum(predictions == run_labels))
        agreement = int(np.sum(predictions == run_host))
        low, high = wilson_interval(correct, n)
        divergent_pos = np.flatnonzero(predictions != run_host).astype(np.int64)
        divergent_dataset_indices = run_order[divergent_pos].astype(np.int64)
        cm = confusion_matrix(run_labels, predictions)
        np.save(outdir / "confusion_matrix.npy", cm)
        np.save(outdir / "divergent_positions.npy", divergent_pos)
        np.save(outdir / "divergent_dataset_indices.npy", divergent_dataset_indices)
        np.save(outdir / "predictions_dpu.npy", predictions)
        np.save(outdir / "raw_outputs_int8.npy", raw_outputs)

        # Logits dequantizados do DPU para auditoria.
        logits_dpu = raw_outputs.astype(np.float32) * (2.0 ** (-output_fix))
        np.save(outdir / "logits_dequantized.npy", logits_dpu)
        np.save(outdir / "labels.npy", run_labels)
        np.save(outdir / "run_order.npy", run_order)

        # Diferencas contra fake-quant host sao informativas, nao criterio de igualdade.
        host_logits_run = host_logits[run_order].astype(np.float32)
        diff = logits_dpu.astype(np.float64) - host_logits_run.astype(np.float64)
        accuracy = {
            "unique_images": n,
            "correct": correct,
            "accuracy_percent": 100.0 * correct / n,
            "wilson95_percent": [100.0 * low, 100.0 * high],
            "host_int8_argmax_agreement": agreement,
            "host_int8_argmax_agreement_percent": 100.0 * agreement / n,
            "divergence_count": int(divergent_pos.size),
            "prediction_sha256_run_order": sha256_array(predictions.astype(np.int64)),
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
            "raw_output_distribution": raw_output_distribution(raw_outputs),
            "dpu_vs_host_fakequant_logits": {
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "max_abs": float(np.max(np.abs(diff))),
            },
        }

    np.save(outdir / "latencies_ns.npy", latencies_ns)
    np.save(outdir / "thread_id_for_position.npy", thread_id_for_pos)

    block_fields = list(block_rows[0].keys())
    save_csv(outdir / "blocks.csv", block_fields, block_rows)
    save_telemetry_csv(outdir / "telemetry_idle.csv", idle_sampler.samples)
    save_telemetry_csv(outdir / "telemetry_active.csv", active_sampler.samples)

    # Resumo DPU antes/depois é coletado fora da regiao medida.
    dpu_status_after = run_command(["xdputil", "status"])

    summary: Dict[str, Any] = {
        "scenario": cfg.scenario,
        "threads": cfg.threads,
        "samples_total": n,
        "batch": 1,
        "warmup_per_runner": cfg.warmup_per_runner,
        "warmup_total_excluded": cfg.warmup_per_runner * cfg.threads,
        "blocks": cfg.blocks,
        "block_size": cfg.block_size,
        "start_utc": datetime.fromtimestamp(block_start_utc_ns[0] / 1e9, tz=timezone.utc).isoformat(),
        "end_utc": datetime.fromtimestamp(block_end_utc_ns[-1] / 1e9, tz=timezone.utc).isoformat(),
        "wall_time_s": wall_s,
        "fps_global_weighted": fps_global,
        "throughput_block_stats": tp_stats,
        "latency_ms": overall_latency,
        "latency_mean_ci95_from_10_block_means_ms": latency_ci,
        "block_latency_means_ms": block_means_ms,
        "block_fps": block_fps,
        "worker_counts": worker_counts.tolist(),
        "per_thread": per_thread,
        "idle_telemetry": idle_summary,
        "active_telemetry": summarize_telemetry(active_sampler.samples),
        "power_energy_overall": {
            **overall_power,
            "idle_mean_power_w": idle_power_w,
            "dynamic_mean_power_w": overall_dynamic_power,
            "dynamic_energy_j": overall_dynamic_energy,
            "total_energy_mj_per_inference": total_energy_mj_per_inf,
            "dynamic_energy_mj_per_inference": dynamic_energy_mj_per_inf,
        },
        "accuracy": accuracy,
        "dpu_status_before": dpu_status_before,
        "dpu_status_after": dpu_status_after,
        "latency_boundary": {
            "inference_only": "input INT8 pronto e copiado antes de t0; t0 -> execute_async + wait -> t1; argmax fora",
            "end_to_end": "t0 -> uint8->float32 /255 -> quantizacao -> copia -> execute_async+wait -> dequantizacao -> argmax -> t1",
            "saturated": "mesma entrada INT8/buffers reutilizados; t0 -> execute_async + wait -> t1; sem preprocessamento/argmax",
        }[cfg.scenario],
        "throughput_scope": "tempo global do primeiro inicio de bloco ao termino do ultimo bloco; exclui criacao de runner, warm-up, baseline idle e escrita em disco",
        "telemetry_paths": {
            "power": str(DEFAULT_POWER_PATH),
            "current": str(DEFAULT_CURRENT_PATH),
            "voltage": str(DEFAULT_VOLTAGE_PATH),
            "ams": str(DEFAULT_AMS),
            "requested_interval_s": telemetry_interval,
            "known_iio_sampling_frequency_hz": 114,
        },
        "system_after": proc_status(),
    }

    json_dump(outdir / "summary.json", summary)

    # Checksums do cenario depois de todos os artefatos.
    checksum_lines = []
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            checksum_lines.append(f"{sha256_file(p)}  {p.name}")
    (outdir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return summary


def build_final_summary(root: Path, summaries: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    rows = []
    for s in summaries:
        p = s["power_energy_overall"]
        rows.append({
            "scenario": s["scenario"],
            "threads": s["threads"],
            "inferences": s["samples_total"],
            "wall_time_s": s["wall_time_s"],
            "fps_global": s["fps_global_weighted"],
            "latency_mean_ms": s["latency_ms"]["mean"],
            "latency_median_ms": s["latency_ms"]["median"],
            "latency_sd_ms": s["latency_ms"]["sample_sd_ddof1"],
            "latency_cv_percent": s["latency_ms"]["cv_percent"],
            "latency_p90_ms": s["latency_ms"]["p90"],
            "latency_p95_ms": s["latency_ms"]["p95"],
            "latency_p99_ms": s["latency_ms"]["p99"],
            "latency_min_ms": s["latency_ms"]["min"],
            "latency_max_ms": s["latency_ms"]["max"],
            "latency_ci95_low_ms": s["latency_mean_ci95_from_10_block_means_ms"]["low"],
            "latency_ci95_high_ms": s["latency_mean_ci95_from_10_block_means_ms"]["high"],
            "idle_power_w": p["idle_mean_power_w"],
            "active_power_w": p["mean_power_w"],
            "dynamic_power_w": p["dynamic_mean_power_w"],
            "total_energy_mj_per_inference": p["total_energy_mj_per_inference"],
            "dynamic_energy_mj_per_inference": p["dynamic_energy_mj_per_inference"],
            "power_samples": p["samples_in_interval"],
            "power_coverage_fraction": p["coverage_fraction"],
            "accuracy_percent": s["accuracy"]["accuracy_percent"] if s["accuracy"] else None,
            "host_agreement_percent": s["accuracy"]["host_int8_argmax_agreement_percent"] if s["accuracy"] else None,
            "divergences": s["accuracy"]["divergence_count"] if s["accuracy"] else None,
        })

    fields = list(rows[0].keys())
    save_csv(root / "FINAL_SUMMARY.csv", fields, rows)
    json_dump(root / "FINAL_SUMMARY.json", {"metadata": metadata, "results": rows})

    lines = [
        "LeNet/MNIST - Vitis AI / ZCU104",
        "Campanha: 12 configuracoes x 10.000 inferencias",
        "",
        "Cenarios:",
        "  inference_only: 1,2,3,4 threads",
        "  end_to_end:     1,2,3,4 threads",
        "  saturated:      1,2,3,4 threads",
        "",
        "Consultar FINAL_SUMMARY.csv para a tabela principal.",
        "Cada configuracao contem summary.json, blocks.csv, latencies_ns.npy,",
        "telemetria idle/ativa e SHA256SUMS.txt.",
        "",
        "Acuracia e concordancia sao calculadas somente para as 10.000 imagens unicas",
        "nos cenarios inference_only e end_to_end; saturated caracteriza desempenho.",
        "",
        "Potencia/energia: INA226 no rail de aproximadamente 12 V exposto via sysfs.",
        "Energia ativa: integral trapezoidal da potencia dentro do intervalo medido.",
        "Energia dinamica: max(E_total - P_idle * tempo, 0).",
        "",
        "Nenhum outlier e removido.",
    ]
    (root / "README_RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checksum_lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel = p.relative_to(root)
            checksum_lines.append(f"{sha256_file(p)}  {rel}")
    (root / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xmodel", type=Path, default=Path("lenet_mnist_no_softmax.xmodel"))
    p.add_argument("--dataset", type=Path, default=Path("mnist_test_uint8.npz"))
    p.add_argument("--reference", type=Path, default=Path("quantized_test_outputs.npz"))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Diretorio novo. Se omitido, cria results/benchmark_<UTC>.")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--scenarios", nargs="+",
                   choices=["inference_only", "end_to_end", "saturated"],
                   default=["inference_only", "end_to_end", "saturated"])
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--warmup", type=int, default=100,
                   help="Warm-ups POR runner; excluidos.")
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--idle-seconds", type=float, default=5.0)
    p.add_argument("--telemetry-interval", type=float, default=0.01)
    p.add_argument("--strict-hashes", action="store_true", default=True)
    p.add_argument("--no-strict-hashes", dest="strict_hashes", action="store_false")
    p.add_argument("--restart", action="store_true",
                   help="APAGA explicitamente output-dir existente antes de iniciar.")
    p.add_argument("--dry-run", action="store_true",
                   help="Valida ambiente/dados/XModel e sai sem executar a campanha.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.samples != 10000:
        raise SystemExit("ERRO: protocolo congelado exige --samples 10000")
    if args.blocks != 10:
        raise SystemExit("ERRO: protocolo congelado exige --blocks 10")
    if sorted(set(args.threads)) != sorted(args.threads) or any(t not in (1, 2, 3, 4) for t in args.threads):
        raise SystemExit("ERRO: --threads deve conter valores unicos entre 1 e 4")
    if args.idle_seconds < 5.0:
        raise SystemExit("ERRO: --idle-seconds deve ser >= 5")
    if args.telemetry_interval <= 0:
        raise SystemExit("ERRO: --telemetry-interval deve ser > 0")

    for p in (args.xmodel, args.dataset, args.reference):
        if not p.is_file():
            raise SystemExit(f"ERRO: arquivo ausente: {p}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_dir or Path("results") / f"benchmark_{ts}"
    if root.exists():
        if not args.restart:
            raise SystemExit(f"ERRO: output-dir ja existe: {root}; use outro diretorio ou --restart explicitamente")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    start_utc = utc_now_iso()
    input_hashes = verify_inputs(args.xmodel, args.dataset, args.reference, args.strict_hashes)

    data = np.load(args.dataset)
    images = np.asarray(data["images"])
    labels = np.asarray(data["labels"]).reshape(-1).astype(np.int64)
    if images.shape != (10000, 28, 28) or images.dtype != np.uint8:
        raise RuntimeError(f"dataset inesperado: images shape={images.shape} dtype={images.dtype}")
    if labels.shape != (10000,):
        raise RuntimeError(f"labels inesperados: {labels.shape}")

    ref = np.load(args.reference)
    host_predictions = np.asarray(ref["predictions"]).reshape(-1).astype(np.int64)
    host_logits = np.asarray(ref["logits"]).astype(np.float32)
    ref_labels = np.asarray(ref["labels"]).reshape(-1).astype(np.int64)
    if host_predictions.shape != (10000,) or host_logits.shape != (10000, 10):
        raise RuntimeError("referencia quantizada com shapes inesperados")
    if not np.array_equal(labels, ref_labels):
        raise RuntimeError("labels do dataset e da referencia nao coincidem")

    # Hash de predicoes oficiais do host.
    host_pred_hash = sha256_array(np.ascontiguousarray(host_predictions, dtype=np.int64))
    if args.strict_hashes and host_pred_hash != EXPECTED_HOST_PRED_SHA256_OFFICIAL:
        raise RuntimeError(
            f"hash de predictions host inesperado: {host_pred_hash} != {EXPECTED_HOST_PRED_SHA256_OFFICIAL}"
        )

    selected = make_selected_indices(labels, args.seed)
    selected_hash = sha256_array(selected)
    if args.strict_hashes and selected_hash != EXPECTED_SELECTED_INDEX_SHA256_N10000:
        raise RuntimeError(
            f"hash de indices N10000 inesperado: {selected_hash} != {EXPECTED_SELECTED_INDEX_SHA256_N10000}"
        )
    run_order = make_run_order(selected, args.seed)

    np.save(root / "selected_indices_n10000.npy", selected)
    np.save(root / "run_order_n10000.npy", run_order)

    (graph, dpu_subgraph), tensor_meta = inspect_xmodel(args.xmodel)
    prequantized = prepare_quantized(images, tensor_meta["input_fix_point"])

    # Confirma que argmax da referencia local da campanha ainda e 9895/10000.
    host_correct = int(np.sum(host_predictions == labels))
    if args.strict_hashes and host_correct != 9895:
        raise RuntimeError(f"referencia INT8 esperada 9895 acertos, encontrada {host_correct}")

    environment = {
        "campaign_start_utc": start_utc,
        "command": sys.argv,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "cwd": str(Path.cwd()),
        "input_hashes": input_hashes,
        "host_prediction_sha256_official_order": host_pred_hash,
        "selected_indices_sha256": selected_hash,
        "run_order_sha256": sha256_array(run_order),
        "tensor_meta": tensor_meta,
        "host_int8_reference": {
            "correct": host_correct,
            "accuracy_percent": 100.0 * host_correct / 10000,
        },
        "xdputil_query": run_command(["xdputil", "query"]),
        "xdputil_status_before": run_command(["xdputil", "status"]),
        "uname": run_command(["uname", "-a"]),
        "os_release": read_text(Path("/etc/os-release")),
        "cpu_governors_and_freqs": cpu_governors_and_freqs(),
        "process_system_before": proc_status(),
        "telemetry_discovery": {
            "hwmon_name": read_text(Path("/sys/class/hwmon/hwmon0/name")),
            "power_path": str(DEFAULT_POWER_PATH),
            "current_path": str(DEFAULT_CURRENT_PATH),
            "voltage_path": str(DEFAULT_VOLTAGE_PATH),
            "iio_ina226_sampling_frequency_hz": read_int(Path("/sys/bus/iio/devices/iio:device1/in_sampling_frequency")),
            "iio_ina226_oversampling_ratio": read_int(Path("/sys/bus/iio/devices/iio:device1/in_oversampling_ratio")),
            "temperature_initial": read_temperatures(),
        },
        "protocol": {
            "samples_per_configuration": 10000,
            "blocks": 10,
            "block_size": 1000,
            "warmup_per_runner": args.warmup,
            "threads": args.threads,
            "scenarios": args.scenarios,
            "seed": args.seed,
            "idle_seconds": args.idle_seconds,
            "telemetry_interval_s": args.telemetry_interval,
            "outlier_removal": False,
            "softmax": False,
            "decision": "argmax logits",
        },
    }
    json_dump(root / "METADATA.json", environment)

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_passed",
            "output_dir": str(root),
            "input_hashes": input_hashes,
            "selected_indices_sha256": selected_hash,
            "tensor_meta": tensor_meta,
            "host_int8_correct": host_correct,
            "temperature": read_temperatures(),
        }, indent=2))
        return 0

    summaries: List[Dict[str, Any]] = []
    for scenario in args.scenarios:
        for threads_count in args.threads:
            config_dir = root / scenario / f"threads_{threads_count}"
            print(f"[START] scenario={scenario} threads={threads_count} samples=10000", flush=True)
            s = run_scenario(
                ScenarioConfig(
                    scenario=scenario,
                    threads=threads_count,
                    samples=10000,
                    blocks=10,
                    block_size=1000,
                    warmup_per_runner=args.warmup,
                ),
                dpu_subgraph=dpu_subgraph,
                tensor_meta=tensor_meta,
                images=images,
                labels=labels,
                host_predictions=host_predictions,
                host_logits=host_logits,
                run_order=run_order,
                prequantized=prequantized,
                outdir=config_dir,
                idle_seconds=args.idle_seconds,
                telemetry_interval=args.telemetry_interval,
            )
            summaries.append(s)
            print(
                f"[DONE] {scenario} T={threads_count} "
                f"lat_mean={s['latency_ms']['mean']:.6f} ms "
                f"fps={s['fps_global_weighted']:.3f} "
                f"P={s['power_energy_overall']['mean_power_w']} W",
                flush=True,
            )

    final_meta = {
        **environment,
        "campaign_end_utc": utc_now_iso(),
        "temperature_final": read_temperatures(),
        "process_system_after": proc_status(),
        "xdputil_status_after_all": run_command(["xdputil", "status"]),
    }
    build_final_summary(root, summaries, final_meta)

    # Integridade basica final.
    validation = {
        "configurations_expected": len(args.scenarios) * len(args.threads),
        "configurations_completed": len(summaries),
        "all_samples_10000": all(s["samples_total"] == 10000 for s in summaries),
        "all_latencies_count_10000": all(s["latency_ms"]["count"] == 10000 for s in summaries),
        "all_positive_fps": all(s["fps_global_weighted"] > 0 for s in summaries),
        "all_power_bracketed": all(bool(s["power_energy_overall"]["bracketed"]) for s in summaries),
        "accuracy_configs": [
            {
                "scenario": s["scenario"],
                "threads": s["threads"],
                "correct": s["accuracy"]["correct"],
                "agreement": s["accuracy"]["host_int8_argmax_agreement"],
                "divergences": s["accuracy"]["divergence_count"],
            }
            for s in summaries if s["accuracy"] is not None
        ],
    }
    validation["passed"] = bool(
        validation["configurations_completed"] == validation["configurations_expected"]
        and validation["all_samples_10000"]
        and validation["all_latencies_count_10000"]
        and validation["all_positive_fps"]
    )
    json_dump(root / "validacao_integridade.json", validation)

    print()
    print("===== CAMPANHA CONCLUIDA =====")
    print("output_dir:", root)
    print("FINAL_SUMMARY.csv:", root / "FINAL_SUMMARY.csv")
    print("validacao_integridade.json:", root / "validacao_integridade.json")
    print("passed:", validation["passed"])
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
