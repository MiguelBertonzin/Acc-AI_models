#!/usr/bin/env python3
"""Benchmark CPU batch 1 da LeNet/MNIST com energia RAPL via turbostat.

O turbostat deve estar ativo como root antes deste processo. O TensorFlow roda
como usuario comum e exclusivamente em /CPU:0. O script associa as amostras
RAPL aos intervalos de cada ciclo por sobreposicao temporal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import queue
import shutil
import time
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import psutil
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "lenet_mnist_final.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "CPU" / "resultados"
DEFAULT_TURBOSTAT_LOG = Path("/tmp/lenet_cpu_turbostat.log")
Z95 = 1.959963984540054


@dataclass
class CpuSample:
    epoch_s: float
    monotonic_s: float
    system_percent: float
    process_percent: float
    frequency_mhz: float
    package_temp_c: float
    rss_mib: float


@dataclass
class RaplSample:
    epoch_s: float
    busy_percent: float
    busy_frequency_mhz: float
    package_temperature_c: float
    package_power_w: float
    cores_power_w: float


class CpuSampler:
    def __init__(self, interval_ms: int) -> None:
        self.interval_s = interval_ms / 1000.0
        self.samples: list[CpuSample] = []
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.errors_queue: queue.Queue[str] = queue.Queue()

    @staticmethod
    def package_temperature() -> float:
        try:
            for item in psutil.sensors_temperatures().get("coretemp", []):
                if item.label == "Package id 0":
                    return float(item.current)
        except Exception:
            pass
        return math.nan

    def take(self) -> None:
        try:
            frequency = psutil.cpu_freq(percpu=False)
            self.samples.append(
                CpuSample(
                    epoch_s=time.time(),
                    monotonic_s=time.perf_counter(),
                    system_percent=float(psutil.cpu_percent(interval=None)),
                    process_percent=float(self.process.cpu_percent(interval=None)),
                    frequency_mhz=float(frequency.current) if frequency else math.nan,
                    package_temp_c=self.package_temperature(),
                    rss_mib=self.process.memory_info().rss / 1048576.0,
                )
            )
        except Exception as exc:
            self.errors_queue.put(repr(exc))

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

    def between(self, start_s: float, end_s: float) -> list[CpuSample]:
        return [sample for sample in self.samples if start_s <= sample.epoch_s <= end_s]

    def errors(self) -> list[str]:
        values: list[str] = []
        while not self.errors_queue.empty():
            values.append(self.errors_queue.get_nowait())
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turbostat-log", type=Path, default=DEFAULT_TURBOSTAT_LOG)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--repetitions", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--telemetry-ms", type=int, default=100)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.sample_sizes or min(args.sample_sizes) < 1 or max(args.sample_sizes) > 10000:
        raise ValueError("sample-sizes deve estar entre 1 e 10000")
    if not args.repetitions or min(args.repetitions) < 1:
        raise ValueError("repetitions deve conter inteiros positivos")
    if args.warmup < 1 or args.baseline_seconds < 1 or args.telemetry_ms < 50:
        raise ValueError("warmup >= 1, baseline >= 1 e telemetry-ms >= 50")
    if not args.turbostat_log.is_file() or not os.access(args.turbostat_log, os.R_OK):
        raise RuntimeError(
            f"log turbostat ausente/ilegivel: {args.turbostat_log}. "
            "Inicie CPU/run_benchmark_rapl.sh em um terminal."
        )


def configure_cpu(args: argparse.Namespace) -> tuple[tf.config.PhysicalDevice, tf.config.LogicalDevice]:
    tf.config.set_visible_devices([], "GPU")
    if args.intra_op_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_op_threads)
    if args.inter_op_threads:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_op_threads)
    physical = tf.config.list_physical_devices("CPU")
    if not physical:
        raise RuntimeError("TensorFlow nao encontrou CPU")
    tf.config.set_visible_devices(physical[0], "CPU")
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_op_determinism()
    logical = tf.config.list_logical_devices("CPU")
    if len(logical) != 1 or tf.config.get_visible_devices("GPU"):
        raise RuntimeError("nao foi possivel garantir execucao exclusiva na CPU")
    return physical[0], logical[0]


def compile_inference_graph(model: tf.keras.Model) -> Callable[[tf.Tensor], tf.Tensor]:
    @tf.function(
        input_signature=[tf.TensorSpec((1, 28, 28, 1), tf.float32)],
        autograph=False,
        jit_compile=False,
    )
    def infer(batch: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):
            return model(batch, training=False)

    return infer


def validate_model(model: tf.keras.Model) -> dict[str, object]:
    names = [layer.name for layer in model.layers]
    expected = ["conv1", "pool1", "conv2", "pool2", "flatten", "dense1", "dense2", "output"]
    activation = tf.keras.activations.serialize(model.layers[-1].activation)
    if (
        model.input_shape != (None, 28, 28, 1)
        or model.output_shape != (None, 10)
        or names != expected
        or model.count_params() != 44426
        or activation != "linear"
    ):
        raise RuntimeError(
            f"modelo inesperado: input={model.input_shape}, output={model.output_shape}, "
            f"camadas={names}, params={model.count_params()}, ativacao={activation}"
        )
    return {
        "name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameters": int(model.count_params()),
        "macs_per_inference": 281640,
        "final_activation": activation,
        "layers": [
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "parameters": int(layer.count_params()),
                "output_shape": list(layer.output.shape),
                "activation": getattr(getattr(layer, "activation", None), "__name__", None),
            }
            for layer in model.layers
        ],
    }


def make_nested_stratified_order(labels: np.ndarray, seed: int) -> np.ndarray:
    labels = labels.reshape(-1)
    if labels.size != 10000 or not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError("esperada a divisao oficial de teste MNIST")
    rng = np.random.default_rng(seed)
    groups = []
    for class_id in range(10):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        groups.append(indices)
    balanced_thousand = np.stack([indices[:100] for indices in groups], axis=1).reshape(-1)
    remainder = np.concatenate([indices[100:] for indices in groups])
    rng.shuffle(remainder)
    order = np.concatenate([balanced_thousand, remainder])
    if np.unique(order).size != 10000:
        raise RuntimeError("ordem nao cobre o teste uma unica vez")
    return order


def finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def finite_mean(values: Iterable[float]) -> float:
    array = finite_values(values)
    return float(np.mean(array)) if array.size else math.nan


def finite_std(values: Iterable[float]) -> float:
    array = finite_values(values)
    return float(np.std(array, ddof=1)) if array.size > 1 else (0.0 if array.size else math.nan)


def latency_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("latencia nao finita")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "latency_mean_ms": mean,
        "latency_median_ms": float(np.median(values)),
        "latency_std_ms": std,
        "latency_cv_percent": 100.0 * std / mean,
        "latency_p90_ms": float(np.percentile(values, 90)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
        "latency_min_ms": float(np.min(values)),
        "latency_max_ms": float(np.max(values)),
    }


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    proportion = correct / total
    denominator = 1.0 + Z95 * Z95 / total
    centre = (proportion + Z95 * Z95 / (2.0 * total)) / denominator
    margin = Z95 * math.sqrt(
        proportion * (1.0 - proportion) / total + Z95 * Z95 / (4.0 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    margin = Z95 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    return mean - margin, mean + margin


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def create_latencies(path: Path, shape: tuple[int, int], restart: bool) -> np.memmap:
    if restart and path.exists():
        path.unlink()
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.float32:
            raise RuntimeError(f"checkpoint incompativel: {path}")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    array[:] = np.nan
    array.flush()
    return array


def cpu_telemetry(samples: list[CpuSample], logical_cpus: int) -> dict[str, float | int]:
    process = finite_mean(sample.process_percent for sample in samples)
    return {
        "telemetry_samples": len(samples),
        "cpu_system_utilization_mean_percent": finite_mean(s.system_percent for s in samples),
        "cpu_process_utilization_mean_percent": process,
        "cpu_process_utilization_normalized_percent": process / logical_cpus if math.isfinite(process) else math.nan,
        "cpu_frequency_psutil_mean_mhz": finite_mean(s.frequency_mhz for s in samples),
        "cpu_package_temperature_psutil_mean_c": finite_mean(s.package_temp_c for s in samples),
        "process_rss_mean_mib": finite_mean(s.rss_mib for s in samples),
    }


def write_predictions(
    path: Path,
    dataset_indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
) -> None:
    rows: list[dict[str, object]] = []
    for local_index in range(labels.size):
        row: dict[str, object] = {
            "local_index": local_index,
            "mnist_test_index": int(dataset_indices[local_index]),
            "label": int(labels[local_index]),
            "prediction": int(predictions[local_index]),
            "correct": int(predictions[local_index] == labels[local_index]),
        }
        for class_id in range(10):
            row[f"logit_{class_id}"] = float(logits[local_index, class_id])
        rows.append(row)
    write_rows(path, rows)


def run_sample_size(
    *,
    infer: Callable[[tf.Tensor], tf.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    dataset_indices: np.ndarray,
    sample_size: int,
    repetitions: int,
    seed: int,
    output_dir: Path,
    sampler: CpuSampler,
    logical_cpus: int,
    restart: bool,
) -> tuple[list[dict[str, object]], np.memmap, int]:
    latency_path = output_dir / f"latencias_n{sample_size}.npy"
    pass_path = output_dir / f"passagens_n{sample_size}.csv"
    prediction_path = output_dir / f"predicoes_n{sample_size}.csv"
    if restart:
        for path in (pass_path, prediction_path):
            if path.exists():
                path.unlink()
    latencies = create_latencies(latency_path, (repetitions, sample_size), restart)
    existing = read_rows(pass_path)
    completed = len(existing)
    if completed and not np.all(np.isfinite(latencies[:completed])):
        raise RuntimeError("CSV e checkpoint discordam")
    rows: list[dict[str, object]] = [dict(row) for row in existing]
    correct_reference = int(existing[0]["correct"]) if existing else None
    digest_reference = existing[0].get("prediction_sha256") if existing else None
    rng = np.random.default_rng(seed + sample_size * 1009)
    permutations = [rng.permutation(sample_size) for _ in range(repetitions)]

    for repetition in range(completed, repetitions):
        predictions = np.empty(sample_size, dtype=np.int64)
        logits = np.empty((sample_size, 10), dtype=np.float32) if repetition == 0 else None
        start_epoch_s = time.time()
        start_monotonic_s = time.perf_counter()
        start_cpu_s = time.process_time()
        started_utc = datetime.now(timezone.utc).isoformat()
        for position, image_index in enumerate(permutations[repetition]):
            with tf.device("/CPU:0"):
                batch = tf.convert_to_tensor(images[image_index : image_index + 1])
                item_start_ns = time.perf_counter_ns()
                output = infer(batch)
                host_output = output.numpy()
                item_end_ns = time.perf_counter_ns()
            if "CPU:0" not in output.device.upper():
                raise RuntimeError(f"saida fora da CPU: {output.device}")
            if not np.all(np.isfinite(host_output)):
                raise RuntimeError("saida nao finita")
            latencies[repetition, position] = (item_end_ns - item_start_ns) / 1_000_000.0
            predictions[image_index] = int(np.argmax(host_output[0]))
            if logits is not None:
                logits[image_index] = host_output[0]
        ended_utc = datetime.now(timezone.utc).isoformat()
        end_cpu_s = time.process_time()
        end_monotonic_s = time.perf_counter()
        end_epoch_s = time.time()
        duration_s = end_monotonic_s - start_monotonic_s
        process_cpu_s = end_cpu_s - start_cpu_s
        correct = int(np.count_nonzero(predictions == labels))
        digest = hashlib.sha256(predictions.tobytes()).hexdigest()
        if correct_reference is None:
            correct_reference, digest_reference = correct, digest
        elif correct != correct_reference or digest != digest_reference:
            raise RuntimeError(f"predicao nao deterministica no ciclo {repetition + 1}")
        if repetition == 0 and logits is not None:
            write_predictions(prediction_path, dataset_indices, labels, predictions, logits)

        row: dict[str, object] = {
            "sample_size": sample_size,
            "repetition": repetition + 1,
            "batch_size": 1,
            "correct": correct,
            "accuracy_percent": 100.0 * correct / sample_size,
            "prediction_sha256": digest,
            "start_epoch_s": start_epoch_s,
            "end_epoch_s": end_epoch_s,
            "start_monotonic_s": start_monotonic_s,
            "end_monotonic_s": end_monotonic_s,
            "wall_time_s": duration_s,
            "process_cpu_time_s": process_cpu_s,
            "cpu_time_per_inference_ms": process_cpu_s * 1000.0 / sample_size,
            "throughput_fps": sample_size / duration_s,
            **latency_stats(np.asarray(latencies[repetition], dtype=np.float64)),
            **cpu_telemetry(sampler.between(start_epoch_s, end_epoch_s), logical_cpus),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
        }
        rows.append(row)
        latencies.flush()
        write_rows(pass_path, rows)
        print(
            f"n={sample_size:5d} ciclo={repetition + 1:3d}/{repetitions} "
            f"acc={row['accuracy_percent']:.2f}% media={row['latency_mean_ms']:.4f} ms "
            f"p95={row['latency_p95_ms']:.4f} ms fps={row['throughput_fps']:.1f}",
            flush=True,
        )
    assert correct_reference is not None
    return rows, latencies, correct_reference


def parse_turbostat(path: Path) -> tuple[list[RaplSample], list[str]]:
    samples: list[RaplSample] = []
    rejected: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if len(fields) != 6:
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        if values[0] < 1_000_000_000:
            rejected.append(line)
            continue
        samples.append(RaplSample(*values))
    samples.sort(key=lambda sample: sample.epoch_s)
    return samples, rejected


def verify_turbostat_active(path: Path, timeout_s: float = 3.0) -> None:
    initial, _ = parse_turbostat(path)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(0.25)
        current, _ = parse_turbostat(path)
        if len(current) > len(initial) and time.time() - current[-1].epoch_s < 2.0:
            return
    raise RuntimeError("o log turbostat nao esta recebendo novas amostras")


def overlap_weighted_rapl(
    samples: list[RaplSample], start_s: float, end_s: float, interval_s: float
) -> dict[str, float | int]:
    duration_s = end_s - start_s
    weighted: list[tuple[RaplSample, float]] = []
    for sample in samples:
        sample_start = sample.epoch_s - interval_s
        overlap = max(0.0, min(end_s, sample.epoch_s) - max(start_s, sample_start))
        if overlap > 0:
            weighted.append((sample, overlap))
    coverage_s = sum(weight for _, weight in weighted)
    if coverage_s <= 0:
        raise RuntimeError(f"janela [{start_s}, {end_s}] sem cobertura RAPL")
    coverage_fraction = min(coverage_s / duration_s, 1.0)
    if coverage_fraction < 0.90:
        raise RuntimeError(f"cobertura RAPL insuficiente: {coverage_fraction:.3%}")

    def average(getter: Callable[[RaplSample], float]) -> float:
        return sum(getter(sample) * weight for sample, weight in weighted) / coverage_s

    package_power_w = average(lambda sample: sample.package_power_w)
    cores_power_w = average(lambda sample: sample.cores_power_w)
    package_energy_j = package_power_w * duration_s
    cores_energy_j = cores_power_w * duration_s
    temperatures = [sample.package_temperature_c for sample, _ in weighted]
    return {
        "rapl_samples": len(weighted),
        "rapl_coverage_s": coverage_s,
        "rapl_coverage_percent": 100.0 * coverage_fraction,
        "cpu_busy_percent_turbostat": average(lambda sample: sample.busy_percent),
        "cpu_busy_frequency_mhz": average(lambda sample: sample.busy_frequency_mhz),
        "cpu_package_temperature_mean_c": average(lambda sample: sample.package_temperature_c),
        "cpu_package_temperature_max_c": float(max(temperatures)),
        "power_package_mean_w": package_power_w,
        "power_cores_mean_w": cores_power_w,
        "energy_package_j": package_energy_j,
        "energy_cores_j": cores_energy_j,
    }


def enrich_passes_with_rapl(
    output_dir: Path,
    sizes: list[int],
    samples: list[RaplSample],
    interval_s: float,
    baseline_start_s: float,
    baseline_end_s: float,
) -> tuple[dict[str, float | int], dict[int, list[dict[str, object]]]]:
    baseline = overlap_weighted_rapl(samples, baseline_start_s, baseline_end_s, interval_s)
    idle_power_w = float(baseline["power_package_mean_w"])
    enriched: dict[int, list[dict[str, object]]] = {}
    for sample_size in sizes:
        rows: list[dict[str, object]] = []
        for original in read_rows(output_dir / f"passagens_n{sample_size}.csv"):
            row: dict[str, object] = dict(original)
            start_s, end_s = float(original["start_epoch_s"]), float(original["end_epoch_s"])
            rapl = overlap_weighted_rapl(samples, start_s, end_s, interval_s)
            duration_s = float(original["wall_time_s"])
            package_energy_j = float(rapl["energy_package_j"])
            dynamic_energy_j = max(package_energy_j - idle_power_w * duration_s, 0.0)
            row.update(rapl)
            row.update(
                {
                    "power_dynamic_mean_w": dynamic_energy_j / duration_s,
                    "energy_dynamic_j": dynamic_energy_j,
                    "energy_per_inference_mj": package_energy_j * 1000.0 / sample_size,
                    "dynamic_energy_per_inference_mj": dynamic_energy_j * 1000.0 / sample_size,
                }
            )
            rows.append(row)
        write_rows(output_dir / f"passagens_n{sample_size}.csv", rows)
        enriched[sample_size] = rows
    baseline["idle_power_package_w"] = idle_power_w
    return baseline, enriched


def aggregate_results(
    sample_size: int,
    prefixes: list[int],
    rows: list[dict[str, object]],
    latencies: np.ndarray,
    correct: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    accuracy_low, accuracy_high = wilson_interval(correct, sample_size)
    for prefix in prefixes:
        chosen = rows[:prefix]
        selected_latency = np.asarray(latencies[:prefix], dtype=np.float64)
        latency_low, latency_high = mean_ci95(selected_latency.mean(axis=1))
        fps = np.asarray([float(row["throughput_fps"]) for row in chosen])
        fps_low, fps_high = mean_ci95(fps)
        wall_time_s = sum(float(row["wall_time_s"]) for row in chosen)
        process_cpu_s = sum(float(row["process_cpu_time_s"]) for row in chosen)
        item: dict[str, object] = {
            "model_variant": "no_softmax_logits",
            "device": "CPU",
            "precision": "float32",
            "batch_size": 1,
            "unique_images": sample_size,
            "repetitions": prefix,
            "collected_inferences": sample_size * prefix,
            "correct_unique_images": correct,
            "accuracy_percent": 100.0 * correct / sample_size,
            "accuracy_wilson95_low_percent": 100.0 * accuracy_low,
            "accuracy_wilson95_high_percent": 100.0 * accuracy_high,
            **latency_stats(selected_latency),
            "latency_mean_pass_ci95_low_ms": latency_low,
            "latency_mean_pass_ci95_high_ms": latency_high,
            "throughput_mean_fps": float(np.mean(fps)),
            "throughput_median_fps": float(np.median(fps)),
            "throughput_std_fps": float(np.std(fps, ddof=1)) if fps.size > 1 else 0.0,
            "throughput_mean_ci95_low_fps": fps_low,
            "throughput_mean_ci95_high_fps": fps_high,
            "throughput_effective_fps": sample_size * prefix / wall_time_s,
            "wall_time_total_s": wall_time_s,
            "process_cpu_time_total_s": process_cpu_s,
            "cpu_time_per_inference_mean_ms": process_cpu_s * 1000.0 / (sample_size * prefix),
        }
        mean_columns = [
            "cpu_system_utilization_mean_percent",
            "cpu_process_utilization_mean_percent",
            "cpu_process_utilization_normalized_percent",
            "cpu_frequency_psutil_mean_mhz",
            "cpu_package_temperature_psutil_mean_c",
            "process_rss_mean_mib",
            "rapl_coverage_percent",
            "cpu_busy_percent_turbostat",
            "cpu_busy_frequency_mhz",
            "cpu_package_temperature_mean_c",
            "cpu_package_temperature_max_c",
            "power_package_mean_w",
            "power_cores_mean_w",
            "power_dynamic_mean_w",
            "energy_per_inference_mj",
            "dynamic_energy_per_inference_mj",
        ]
        for column in mean_columns:
            item[column] = finite_mean(float(row[column]) for row in chosen)
        item["energy_package_j"] = sum(float(row["energy_package_j"]) for row in chosen)
        item["energy_cores_j"] = sum(float(row["energy_cores_j"]) for row in chosen)
        item["energy_dynamic_j"] = sum(float(row["energy_dynamic_j"]) for row in chosen)
        item["telemetry_samples_total"] = sum(int(float(row["telemetry_samples"])) for row in chosen)
        item["rapl_samples_total"] = sum(int(float(row["rapl_samples"])) for row in chosen)
        item["prediction_sha256"] = str(chosen[0]["prediction_sha256"])
        results.append(item)
    return results


def cpu_model_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def write_report(path: Path, results: list[dict[str, object]], metadata: dict[str, object]) -> None:
    primary = next(
        row for row in results
        if row["unique_images"] == max(metadata["sample_sizes"])
        and row["repetitions"] == metadata["maximum_repetitions_executed"]
    )
    lines = [
        "# Resultados — LeNet/MNIST sem softmax na CPU",
        "",
        "## Resultado principal",
        "",
        "| Metrica | Resultado |",
        "|---|---:|",
        f"| CPU | {metadata['cpu_model_name']} |",
        f"| Precisao / batch | float32 / 1 |",
        f"| Imagens / ciclos / inferencias | {primary['unique_images']} / {primary['repetitions']} / {primary['collected_inferences']} |",
        f"| Acuracia | {primary['accuracy_percent']:.4f}% |",
        f"| IC95 Wilson | [{primary['accuracy_wilson95_low_percent']:.4f}%; {primary['accuracy_wilson95_high_percent']:.4f}%] |",
        f"| Latencia media | {primary['latency_mean_ms']:.4f} ms |",
        f"| IC95 da media por ciclo | [{primary['latency_mean_pass_ci95_low_ms']:.4f}; {primary['latency_mean_pass_ci95_high_ms']:.4f}] ms |",
        f"| Mediana / desvio | {primary['latency_median_ms']:.4f} / {primary['latency_std_ms']:.4f} ms |",
        f"| p90 / p95 / p99 | {primary['latency_p90_ms']:.4f} / {primary['latency_p95_ms']:.4f} / {primary['latency_p99_ms']:.4f} ms |",
        f"| Minimo / maximo | {primary['latency_min_ms']:.4f} / {primary['latency_max_ms']:.4f} ms |",
        f"| Vazao media / efetiva | {primary['throughput_mean_fps']:.3f} / {primary['throughput_effective_fps']:.3f} inf/s |",
        f"| Tempo CPU por inferencia | {primary['cpu_time_per_inference_mean_ms']:.4f} ms |",
        f"| Ocupacao / Bzy_MHz turbostat | {primary['cpu_busy_percent_turbostat']:.3f}% / {primary['cpu_busy_frequency_mhz']:.1f} MHz |",
        f"| Temperatura package media | {primary['cpu_package_temperature_mean_c']:.2f} C |",
        f"| Potencia package / cores | {primary['power_package_mean_w']:.3f} / {primary['power_cores_mean_w']:.3f} W |",
        f"| Potencia dinamica package | {primary['power_dynamic_mean_w']:.3f} W |",
        f"| Energia package/inferencia | {primary['energy_per_inference_mj']:.4f} mJ |",
        f"| Energia dinamica/inferencia | {primary['dynamic_energy_per_inference_mj']:.4f} mJ |",
        f"| Cobertura RAPL media | {primary['rapl_coverage_percent']:.3f}% |",
        f"| Amostras RAPL | {primary['rapl_samples_total']} |",
        f"| Potencia package ociosa | {metadata['rapl_baseline']['idle_power_package_w']:.3f} W |",
        "",
        "## Convergencia",
        "",
        "| Imagens | Ciclos | Inferencias | Acuracia | Media (ms) | Mediana | p95 | FPS efetivo | Pkg W | Energia mJ/inf. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['unique_images']} | {row['repetitions']} | {row['collected_inferences']} "
            f"| {row['accuracy_percent']:.3f}% | {row['latency_mean_ms']:.4f} "
            f"| {row['latency_median_ms']:.4f} | {row['latency_p95_ms']:.4f} "
            f"| {row['throughput_effective_fps']:.2f} | {row['power_package_mean_w']:.3f} "
            f"| {row['energy_per_inference_mj']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metodologia",
            "",
            "Inferencia float32, batch 1, serial e sincrona, com entrada normalizada antes "
            "do cronometro. TensorFlow foi restrito a `/CPU:0`; GPU e soft placement foram "
            "desabilitados. O grafo usa `tf.function`, assinatura fixa e `jit_compile=False`.",
            "",
            "O turbostat root amostrou a cada 100 ms `Busy%`, `Bzy_MHz`, `PkgTmp`, `PkgWatt` "
            "e `CorWatt`. Cada leitura representa o intervalo anterior de 100 ms. A energia "
            "foi integrada pela sobreposicao exata entre esse intervalo e cada ciclo. "
            "`PkgWatt` ja inclui `CorWatt`; ambos nao foram somados.",
            "",
            "Potencia e energia RAPL representam o pacote do processador, nao a tomada. "
            "O escopo fisico difere da potencia da placa GPU. Nenhum outlier foi removido.",
            "",
            "Arquivos brutos, ciclos, latencias, predicoes, telemetria e metadados estao "
            "preservados neste diretorio.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    verify_turbostat_active(args.turbostat_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    physical_cpu, logical_cpu = configure_cpu(args)
    logical_cpus = int(psutil.cpu_count(logical=True) or 1)
    physical_cpus = int(psutil.cpu_count(logical=False) or 1)
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(args.model, compile=False)
    architecture = validate_model(model)
    infer = compile_inference_graph(model)

    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = (test_images.astype(np.float32) / 255.0)[..., np.newaxis]
    test_labels = test_labels.reshape(-1).astype(np.int64)
    order = make_nested_stratified_order(test_labels, args.seed)

    for index in order[: args.warmup]:
        with tf.device("/CPU:0"):
            warmup_output = infer(tf.convert_to_tensor(test_images[index : index + 1]))
            warmup_host = warmup_output.numpy()
    if "CPU:0" not in warmup_output.device.upper() or not np.all(np.isfinite(warmup_host)):
        raise RuntimeError(f"aquecimento invalido: {warmup_output.device}")

    sampler = CpuSampler(args.telemetry_ms)
    sampler.start()
    baseline_start_s = time.time()
    time.sleep(args.baseline_seconds)
    baseline_end_s = time.time()
    sizes = sorted(set(args.sample_sizes))
    prefixes = sorted(set(args.repetitions))
    maximum = max(prefixes)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        for sample_size in sizes:
            selected = order[:sample_size]
            run_sample_size(
                infer=infer,
                images=np.ascontiguousarray(test_images[selected]),
                labels=np.ascontiguousarray(test_labels[selected]),
                dataset_indices=selected,
                sample_size=sample_size,
                repetitions=maximum,
                seed=args.seed,
                output_dir=args.output_dir,
                sampler=sampler,
                logical_cpus=logical_cpus,
                restart=args.restart,
            )
    finally:
        sampler.stop()
        write_rows(args.output_dir / "telemetria_cpu_psutil.csv", [asdict(s) for s in sampler.samples])

    time.sleep(max(0.3, args.telemetry_ms / 1000.0 * 2.0))
    rapl_samples, rejected = parse_turbostat(args.turbostat_log)
    if not rapl_samples:
        raise RuntimeError("nenhuma amostra RAPL encontrada")
    shutil.copy2(args.turbostat_log, args.output_dir / "turbostat_bruto.log")
    baseline, enriched = enrich_passes_with_rapl(
        args.output_dir,
        sizes,
        rapl_samples,
        args.telemetry_ms / 1000.0,
        baseline_start_s,
        baseline_end_s,
    )

    results: list[dict[str, object]] = []
    for sample_size in sizes:
        latencies = np.load(args.output_dir / f"latencias_n{sample_size}.npy", mmap_mode="r")
        correct = int(enriched[sample_size][0]["correct"])
        results.extend(
            aggregate_results(sample_size, prefixes, enriched[sample_size], latencies, correct)
        )
    write_rows(args.output_dir / "resultados.csv", results)

    metadata: dict[str, object] = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "model_sha256": sha256_file(args.model),
        "architecture": architecture,
        "dataset": "MNIST official test split",
        "dataset_class_counts": {
            str(class_id): int(np.count_nonzero(test_labels == class_id)) for class_id in range(10)
        },
        "input_normalization": "uint8 -> float32 / 255.0; canal adicionado ao final",
        "sampling": "100 e 1000 balanceados/aninhados; 10000 = teste oficial completo",
        "seed": args.seed,
        "sample_sizes": sizes,
        "repetition_prefixes": prefixes,
        "maximum_repetitions_executed": maximum,
        "batch_size": 1,
        "precision": "float32",
        "execution_mode": "tf.function em /CPU:0, autograph=False, jit_compile=False",
        "synchronization": "output.numpy() em cada inferencia",
        "warmup_inferences": args.warmup,
        "telemetry_interval_ms": args.telemetry_ms,
        "baseline_duration_s": args.baseline_seconds,
        "baseline_start_epoch_s": baseline_start_s,
        "baseline_end_epoch_s": baseline_end_s,
        "rapl_baseline": baseline,
        "turbostat_source": str(args.turbostat_log),
        "turbostat_snapshot_sha256": sha256_file(args.output_dir / "turbostat_bruto.log"),
        "turbostat_samples_parsed": len(rapl_samples),
        "turbostat_rejected_numeric_lines": rejected,
        "cpu_model_name": cpu_model_name(),
        "physical_cpu_count": physical_cpus,
        "logical_cpu_count": logical_cpus,
        "process_cpu_affinity": psutil.Process().cpu_affinity(),
        "tensorflow_intra_op_threads": args.intra_op_threads,
        "tensorflow_inter_op_threads": args.inter_op_threads,
        "tensorflow_visible_gpus": [str(item) for item in tf.config.get_visible_devices("GPU")],
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "physical_cpu": str(physical_cpu),
        "logical_cpu": str(logical_cpu),
        "cpu_sampler_errors": sampler.errors(),
        "limitations": [
            "RAPL estima energia do pacote; nao mede a tomada",
            "ciclos sequenciais podem apresentar autocorrelacao temporal",
            "CPU permaneceu com escalonamento e frequencia dinamicos",
            "threads TensorFlow/oneDNN em politica automatica quando configuradas como zero",
        ],
    }
    (args.output_dir / "metadados.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "RESULTADOS.md", results, metadata)
    write_report(PROJECT_ROOT / "CPU" / "README.md", results, metadata)
    print(f"Resultados: {args.output_dir / 'RESULTADOS.md'}", flush=True)


if __name__ == "__main__":
    main()
