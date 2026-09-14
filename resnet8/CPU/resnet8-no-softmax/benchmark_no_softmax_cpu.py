#!/usr/bin/env python3
"""Reproducible batch-1 CPU benchmark for the logits ResNet-8 model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import psutil
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "resnet8_cifar10_keras3_no_softmax.h5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "resultados"
MODEL_VARIANT = "no_softmax"
EXPECTED_ACTIVATION = "linear"
REPORT_TITLE = "ResNet-8 sem softmax na CPU"


@dataclass
class Sample:
    epoch_s: float
    system_percent: float
    process_percent: float
    frequency_mhz: float
    package_temp_c: float
    rss_mib: float


class CpuSampler:
    def __init__(self, interval_ms: int) -> None:
        self.interval_s = interval_ms / 1000.0
        self.samples: list[Sample] = []
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error_queue: queue.Queue[str] = queue.Queue()

    @staticmethod
    def package_temp() -> float:
        try:
            for item in psutil.sensors_temperatures().get("coretemp", []):
                if item.label == "Package id 0":
                    return float(item.current)
        except Exception:
            pass
        return math.nan

    def take(self) -> None:
        try:
            freq = psutil.cpu_freq(percpu=False)
            self.samples.append(
                Sample(
                    epoch_s=time.time(),
                    system_percent=float(psutil.cpu_percent(interval=None)),
                    process_percent=float(self.process.cpu_percent(interval=None)),
                    frequency_mhz=(float(freq.current) if freq and float(freq.current) >= 100.0 else math.nan),
                    package_temp_c=self.package_temp(),
                    rss_mib=self.process.memory_info().rss / 1048576.0,
                )
            )
        except Exception as exc:
            self.error_queue.put(repr(exc))

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self.take()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.take()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.take()

    def between(self, start: float, end: float) -> list[Sample]:
        return [x for x in self.samples if start <= x.epoch_s <= end]

    def errors(self) -> list[str]:
        values: list[str] = []
        while not self.error_queue.empty():
            values.append(self.error_queue.get_nowait())
        return values


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 1000, 10000])
    p.add_argument("--repetitions", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--baseline-seconds", type=float, default=5.0)
    p.add_argument("--telemetry-ms", type=int, default=100)
    p.add_argument("--intra-op-threads", type=int, default=0)
    p.add_argument("--inter-op-threads", type=int, default=0)
    p.add_argument("--restart", action="store_true")
    return p.parse_args()


def setup_cpu(args: argparse.Namespace) -> tuple[object, object]:
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if min(args.sample_sizes) < 1 or max(args.sample_sizes) > 10000:
        raise ValueError("sample sizes must be in [1, 10000]")
    if min(args.repetitions) < 1:
        raise ValueError("repetitions must be positive")
    tf.config.set_visible_devices([], "GPU")
    if args.intra_op_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_op_threads)
    if args.inter_op_threads:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_op_threads)
    physical = tf.config.list_physical_devices("CPU")
    if not physical:
        raise RuntimeError("TensorFlow CPU unavailable")
    tf.config.set_visible_devices(physical[0], "CPU")
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_op_determinism()
    logical = tf.config.list_logical_devices("CPU")
    if len(logical) != 1 or tf.config.get_visible_devices("GPU"):
        raise RuntimeError("CPU-only placement could not be guaranteed")
    return physical[0], logical[0]


def compile_graph(model: tf.keras.Model) -> Callable[[tf.Tensor], tf.Tensor]:
    @tf.function(
        input_signature=[tf.TensorSpec((1, 32, 32, 3), tf.float32)],
        autograph=False,
        jit_compile=False,
    )
    def infer(batch: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):
            return model(batch, training=False)

    return infer


def stratified_order(labels: np.ndarray, seed: int) -> np.ndarray:
    labels = labels.reshape(-1)
    rng = np.random.default_rng(seed)
    groups = []
    for class_id in range(10):
        idx = np.flatnonzero(labels == class_id)
        rng.shuffle(idx)
        if len(idx) != 1000:
            raise ValueError("Not the standard CIFAR-10 test split")
        groups.append(idx)
    return np.stack(groups, axis=1).reshape(-1)


def finite_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


def stats(a: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise RuntimeError("Invalid latency")
    return {
        "latency_mean_ms": float(x.mean()),
        "latency_median_ms": float(np.median(x)),
        "latency_std_ms": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "latency_p95_ms": float(np.percentile(x, 95)),
        "latency_min_ms": float(x.min()),
        "latency_max_ms": float(x.max()),
    }


def wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return c - m, c + m


def mean_ci(a: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, np.float64)
    mean = float(x.mean())
    if x.size < 2:
        return mean, mean
    margin = 1.959963984540054 * float(x.std(ddof=1)) / math.sqrt(x.size)
    return mean - margin, mean + margin


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latency_store(path: Path, shape: tuple[int, int], restart: bool) -> np.memmap:
    if restart and path.exists():
        path.unlink()
    if path.exists():
        a = np.load(path, mmap_mode="r+")
        if a.shape != shape or a.dtype != np.float32:
            raise ValueError(f"Bad checkpoint {path}")
        return a
    a = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    a[:] = np.nan
    a.flush()
    return a


def telemetry(samples: list[Sample], logical_cpus: int) -> dict[str, float | int]:
    process = finite_mean(x.process_percent for x in samples)
    return {
        "telemetry_samples": len(samples),
        "cpu_system_utilization_mean_percent": finite_mean(x.system_percent for x in samples),
        "cpu_process_utilization_mean_percent": process,
        "cpu_process_utilization_normalized_percent": process / logical_cpus if math.isfinite(process) else math.nan,
        "cpu_frequency_psutil_mean_mhz": finite_mean(x.frequency_mhz for x in samples),
        "cpu_package_temperature_psutil_mean_c": finite_mean(x.package_temp_c for x in samples),
        "process_rss_mean_mib": finite_mean(x.rss_mib for x in samples),
    }


def run_size(
    infer: Callable[[tf.Tensor], tf.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    n: int,
    repetitions: int,
    seed: int,
    output: Path,
    sampler: CpuSampler,
    logical_cpus: int,
    restart: bool,
) -> tuple[list[dict[str, object]], np.memmap, int]:
    npy = output / f"latencias_n{n}.npy"
    csv_path = output / f"passagens_n{n}.csv"
    if restart and csv_path.exists():
        csv_path.unlink()
    latencies = latency_store(npy, (repetitions, n), restart)
    rows: list[dict[str, object]] = [dict(x) for x in read_csv(csv_path)]
    completed = len(rows)
    if completed and not np.isfinite(latencies[:completed]).all():
        raise RuntimeError("CSV/checkpoint disagreement")
    reference = int(rows[0]["correct"]) if rows else None
    rng = np.random.default_rng(seed + n * 1009)
    orders = [rng.permutation(n) for _ in range(repetitions)]

    for rep in range(completed, repetitions):
        prediction = np.empty(n, np.int64)
        start_epoch = time.time()
        start_perf = time.perf_counter()
        start_cpu = time.process_time()
        started_utc = datetime.now(timezone.utc).isoformat()
        for pos, image_idx in enumerate(orders[rep]):
            with tf.device("/CPU:0"):
                batch = tf.convert_to_tensor(images[image_idx : image_idx + 1])
                t0 = time.perf_counter_ns()
                result = infer(batch)
                host = result.numpy()
                t1 = time.perf_counter_ns()
            if "CPU:0" not in result.device.upper():
                raise RuntimeError(f"Output not on CPU: {result.device}")
            latencies[rep, pos] = (t1 - t0) / 1e6
            prediction[image_idx] = int(np.argmax(host[0]))
        ended_utc = datetime.now(timezone.utc).isoformat()
        end_cpu = time.process_time()
        end_perf = time.perf_counter()
        end_epoch = time.time()
        correct = int(np.count_nonzero(prediction == labels))
        if reference is None:
            reference = correct
        elif correct != reference:
            raise RuntimeError("Non-deterministic predictions")
        duration = end_perf - start_perf
        cpu_seconds = end_cpu - start_cpu
        row: dict[str, object] = {
            "sample_size": n,
            "repetition": rep + 1,
            "batch_size": 1,
            "correct": correct,
            "accuracy_percent": 100 * correct / n,
            "start_epoch_s": start_epoch,
            "end_epoch_s": end_epoch,
            "wall_time_s": duration,
            "process_cpu_time_s": cpu_seconds,
            "cpu_time_per_inference_ms": cpu_seconds * 1000 / n,
            "throughput_fps": n / duration,
            **stats(latencies[rep]),
            **telemetry(sampler.between(start_epoch, end_epoch), logical_cpus),
            "rapl_samples": 0,
            "cpu_busy_percent_turbostat": math.nan,
            "cpu_busy_frequency_mean_mhz": math.nan,
            "cpu_package_temperature_mean_c": math.nan,
            "power_package_mean_w": math.nan,
            "power_cores_mean_w": math.nan,
            "power_dram_mean_w": math.nan,
            "power_dynamic_mean_w": math.nan,
            "energy_package_j": math.nan,
            "energy_per_inference_mj": math.nan,
            "dynamic_energy_per_inference_mj": math.nan,
            "started_utc": started_utc,
            "ended_utc": ended_utc,
        }
        rows.append(row)
        latencies.flush()
        write_csv(csv_path, rows)
        print(
            f"n={n:5d} pass={rep + 1:3d}/{repetitions} "
            f"acc={row['accuracy_percent']:.2f}% "
            f"mean={row['latency_mean_ms']:.3f} ms fps={row['throughput_fps']:.1f}",
            flush=True,
        )
    assert reference is not None
    return rows, latencies, reference


def aggregate(n: int, prefixes: list[int], rows: list[dict[str, object]], lat: np.ndarray, correct: int) -> list[dict[str, object]]:
    output = []
    acc_low, acc_high = wilson(correct, n)
    for k in prefixes:
        selected = rows[:k]
        values = np.asarray(lat[:k], np.float64)
        pass_means = values.mean(axis=1)
        low, high = mean_ci(pass_means)
        fps = np.asarray([float(x["throughput_fps"]) for x in selected])
        fps_low, fps_high = mean_ci(fps)
        wall = sum(float(x["wall_time_s"]) for x in selected)
        cpu_time = sum(float(x["process_cpu_time_s"]) for x in selected)
        item: dict[str, object] = {
            "model_variant": MODEL_VARIANT,
            "device": "CPU",
            "batch_size": 1,
            "unique_images": n,
            "repetitions": k,
            "collected_inferences": n * k,
            "correct_unique_images": correct,
            "accuracy_percent": 100 * correct / n,
            "accuracy_wilson95_low_percent": 100 * acc_low,
            "accuracy_wilson95_high_percent": 100 * acc_high,
            **stats(values),
            "latency_mean_pass_ci95_low_ms": low,
            "latency_mean_pass_ci95_high_ms": high,
            "throughput_mean_fps": float(fps.mean()),
            "throughput_std_fps": float(fps.std(ddof=1)),
            "throughput_mean_ci95_low_fps": fps_low,
            "throughput_mean_ci95_high_fps": fps_high,
            "throughput_effective_fps": n * k / wall,
            "wall_time_total_s": wall,
            "process_cpu_time_total_s": cpu_time,
            "cpu_time_per_inference_mean_ms": finite_mean(float(x["cpu_time_per_inference_ms"]) for x in selected),
        }
        for name in [
            "cpu_system_utilization_mean_percent",
            "cpu_process_utilization_mean_percent",
            "cpu_process_utilization_normalized_percent",
            "cpu_frequency_psutil_mean_mhz",
            "cpu_package_temperature_psutil_mean_c",
            "process_rss_mean_mib",
            "cpu_busy_percent_turbostat",
            "cpu_busy_frequency_mean_mhz",
            "cpu_package_temperature_mean_c",
            "power_package_mean_w",
            "power_cores_mean_w",
            "power_dram_mean_w",
            "power_dynamic_mean_w",
            "energy_per_inference_mj",
            "dynamic_energy_per_inference_mj",
        ]:
            item[name] = finite_mean(float(x[name]) for x in selected)
        item["telemetry_samples_total"] = sum(int(float(x["telemetry_samples"])) for x in selected)
        item["rapl_samples_total"] = sum(int(float(x["rapl_samples"])) for x in selected)
        output.append(item)
    return output


def model_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def rebuild_results(output: Path, sizes: list[int], prefixes: list[int]) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    for n in sizes:
        rows: list[dict[str, object]] = [dict(x) for x in read_csv(output / f"passagens_n{n}.csv")]
        lat = np.load(output / f"latencias_n{n}.npy", mmap_mode="r")
        correct = int(rows[0]["correct"])
        all_rows.extend(aggregate(n, prefixes, rows, lat, correct))
    write_csv(output / "resultados.csv", all_rows)
    return all_rows


def compact_report(path: Path, results: list[dict[str, object]], metadata: dict[str, object]) -> None:
    lines = [
        f"# Resultados — {REPORT_TITLE}",
        "",
        f"CPU: {metadata['cpu_model_name']}",
        "",
        "| Imagens | Ciclos | Medições | Acurácia (%) | Média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | FPS | PkgWatt | Energia/inf. (mJ) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in results:
        def f(name: str, d: int = 4) -> str:
            v = float(x[name])
            return f"{v:.{d}f}" if math.isfinite(v) else "N/A"
        lines.append(
            f"| {x['unique_images']} | {x['repetitions']} | {x['collected_inferences']} | "
            f"{f('accuracy_percent', 3)} | {f('latency_mean_ms')} | {f('latency_median_ms')} | "
            f"{f('latency_std_ms')} | {f('latency_p95_ms')} | {f('throughput_mean_fps', 2)} | "
            f"{f('power_package_mean_w', 3)} | {f('energy_per_inference_mj')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physical, logical = setup_cpu(args)
    logical_cpus = int(psutil.cpu_count(logical=True) or 1)

    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(args.model, compile=False)
    activation = tf.keras.activations.serialize(model.layers[-1].activation)
    if activation != EXPECTED_ACTIVATION:
        raise ValueError(f"Expected {EXPECTED_ACTIVATION}, found {activation}")
    infer = compile_graph(model)

    (_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
    x = x.astype(np.float32) / 255.0
    y = y.reshape(-1).astype(np.int64)
    order = stratified_order(y, args.seed)
    for idx in order[: args.warmup]:
        out = infer(tf.convert_to_tensor(x[idx : idx + 1]))
        out.numpy()
    if "CPU:0" not in out.device.upper():
        raise RuntimeError("Warmup output not on CPU")

    sampler = CpuSampler(args.telemetry_ms)
    sampler.start()
    baseline_start = time.time()
    time.sleep(args.baseline_seconds)
    baseline_end = time.time()
    baseline = sampler.between(baseline_start, baseline_end)
    sizes = sorted(set(args.sample_sizes))
    prefixes = sorted(set(args.repetitions))
    maximum = max(prefixes)
    started = datetime.now(timezone.utc).isoformat()
    try:
        for n in sizes:
            chosen = order[:n]
            run_size(
                infer,
                np.ascontiguousarray(x[chosen]),
                np.ascontiguousarray(y[chosen]),
                n,
                maximum,
                args.seed,
                args.output_dir,
                sampler,
                logical_cpus,
                args.restart,
            )
    finally:
        sampler.stop()

    results = rebuild_results(args.output_dir, sizes, prefixes)
    freq = psutil.cpu_freq(percpu=False)
    metadata = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "model_parameter_count": int(model.count_params()),
        "model_final_activation": activation,
        "model_variant": MODEL_VARIANT,
        "input_normalization": "float32 / 255.0",
        "dataset": "CIFAR-10 official test split",
        "sampling": "nested stratified fixed-seed order",
        "seed": args.seed,
        "sample_sizes": sizes,
        "repetition_prefixes": prefixes,
        "maximum_repetitions_executed": maximum,
        "batch_size": 1,
        "execution_mode": "tf.function graph on /CPU:0, jit_compile=False",
        "warmup_inferences": args.warmup,
        "telemetry_interval_ms": args.telemetry_ms,
        "baseline_start_epoch_s": baseline_start,
        "baseline_end_epoch_s": baseline_end,
        "baseline_telemetry_samples": len(baseline),
        "baseline_system_utilization_percent": finite_mean(z.system_percent for z in baseline),
        "baseline_process_utilization_percent": finite_mean(z.process_percent for z in baseline),
        "baseline_frequency_mhz": finite_mean(z.frequency_mhz for z in baseline),
        "baseline_package_temperature_c": finite_mean(z.package_temp_c for z in baseline),
        "rapl_enriched": False,
        "turbostat_log": None,
        "intra_op_threads": args.intra_op_threads,
        "inter_op_threads": args.inter_op_threads,
        "thread_setting_meaning": "0 means TensorFlow automatic selection",
        "cpu_model_name": model_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": logical_cpus,
        "frequency_min_mhz": freq.min if freq else None,
        "frequency_max_mhz": freq.max if freq else None,
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "physical_cpu": str(physical),
        "logical_cpu": str(logical),
        "visible_tensorflow_gpus": [str(z) for z in tf.config.get_visible_devices("GPU")],
        "inference_output_device": str(out.device),
        "telemetry_errors": sampler.errors(),
    }
    (args.output_dir / "metadados.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    compact_report(args.output_dir / "RESULTADOS.md", results, metadata)
    print(f"Results: {args.output_dir / 'RESULTADOS.md'}")


if __name__ == "__main__":
    main()
