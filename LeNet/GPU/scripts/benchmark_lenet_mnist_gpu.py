#!/usr/bin/env python3
"""Benchmark reproduzivel de inferencia batch 1 da LeNet/MNIST na GPU.

Executa uma unica serie ate o maior numero de ciclos. Os resultados de 10, 20,
50 e 100 ciclos sao prefixos cumulativos da mesma coleta. A saida do modelo sao
10 logits (sem softmax), e a sincronizacao e forcada por Tensor.numpy().
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
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "lenet_mnist_final.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "GPU" / "resultados"
Z95 = 1.959963984540054


@dataclass
class TelemetrySample:
    monotonic_s: float
    timestamp: str
    pstate: str
    power_instant_w: float
    power_average_w: float
    utilization_gpu_percent: float
    utilization_memory_percent: float
    memory_used_mib: float
    clock_sm_mhz: float
    clock_memory_mhz: float
    temperature_c: float
    fan_speed_percent: float


class NvidiaSmiSampler:
    """Le continuamente a telemetria de um processo nvidia-smi persistente."""

    FIELDS = (
        "timestamp,pstate,power.draw.instant,power.draw.average,"
        "utilization.gpu,utilization.memory,memory.used,clocks.current.sm,"
        "clocks.current.memory,temperature.gpu,fan.speed"
    )

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = interval_ms
        self.samples: list[TelemetrySample] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._errors: queue.Queue[str] = queue.Queue()

    @staticmethod
    def _number(value: str) -> float:
        try:
            return float(value.strip())
        except ValueError:
            return math.nan

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi nao foi encontrado")
        command = [
            "nvidia-smi",
            f"--query-gpu={self.FIELDS}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                fields = [part.strip() for part in line.strip().split(",")]
                if len(fields) != 11:
                    self._errors.put(f"linha de telemetria ignorada: {line.strip()}")
                    continue
                self.samples.append(
                    TelemetrySample(
                        monotonic_s=time.perf_counter(),
                        timestamp=fields[0],
                        pstate=fields[1],
                        power_instant_w=self._number(fields[2]),
                        power_average_w=self._number(fields[3]),
                        utilization_gpu_percent=self._number(fields[4]),
                        utilization_memory_percent=self._number(fields[5]),
                        memory_used_mib=self._number(fields[6]),
                        clock_sm_mhz=self._number(fields[7]),
                        clock_memory_mhz=self._number(fields[8]),
                        temperature_c=self._number(fields[9]),
                        fan_speed_percent=self._number(fields[10]),
                    )
                )
        except Exception as exc:
            self._errors.put(f"falha no leitor de telemetria: {exc!r}")

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)

    def between(self, start_s: float, end_s: float) -> list[TelemetrySample]:
        return [sample for sample in self.samples if start_s <= sample.monotonic_s <= end_s]

    def errors(self) -> list[str]:
        values: list[str] = []
        while not self._errors.empty():
            values.append(self._errors.get_nowait())
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--repetitions", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--telemetry-ms", type=int, default=100)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.sample_sizes or min(args.sample_sizes) <= 0 or max(args.sample_sizes) > 10000:
        raise ValueError("sample-sizes deve estar entre 1 e 10000")
    if not args.repetitions or min(args.repetitions) <= 0:
        raise ValueError("repetitions deve conter inteiros positivos")
    if args.warmup < 1 or args.baseline_seconds < 0 or args.telemetry_ms < 50:
        raise ValueError("warmup >= 1, baseline >= 0 e telemetry-ms >= 50")


def configure_gpu() -> tuple[tf.config.PhysicalDevice, tf.config.LogicalDevice]:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("TensorFlow nao encontrou GPU; fallback para CPU e proibido")
    tf.config.set_visible_devices(physical[0], "GPU")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_op_determinism()
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError(f"esperada uma GPU logica, encontradas: {logical!r}")
    return physical[0], logical[0]


def compile_inference_graph(model: tf.keras.Model) -> Callable[[tf.Tensor], tf.Tensor]:
    @tf.function(
        input_signature=[tf.TensorSpec((1, 28, 28, 1), tf.float32)],
        autograph=False,
        jit_compile=False,
    )
    def infer(batch: tf.Tensor) -> tf.Tensor:
        with tf.device("/GPU:0"):
            return model(batch, training=False)

    return infer


def validate_model(model: tf.keras.Model) -> dict[str, object]:
    expected_names = ["conv1", "pool1", "conv2", "pool2", "flatten", "dense1", "dense2", "output"]
    observed_names = [layer.name for layer in model.layers]
    activation = tf.keras.activations.serialize(model.layers[-1].activation)
    if model.input_shape != (None, 28, 28, 1):
        raise RuntimeError(f"entrada inesperada: {model.input_shape}")
    if model.output_shape != (None, 10) or activation != "linear":
        raise RuntimeError(f"saida inesperada: {model.output_shape}, ativacao={activation}")
    if observed_names != expected_names or model.count_params() != 44426:
        raise RuntimeError(
            f"arquitetura inesperada: camadas={observed_names}, parametros={model.count_params()}"
        )
    layers = []
    for layer in model.layers:
        layers.append(
            {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "output_shape": list(layer.output.shape),
                "parameters": int(layer.count_params()),
                "activation": getattr(getattr(layer, "activation", None), "__name__", None),
            }
        )
    return {
        "name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "final_activation": activation,
        "parameters": int(model.count_params()),
        "macs_per_inference": 281640,
        "layers": layers,
    }


def make_nested_stratified_order(labels: np.ndarray, seed: int) -> np.ndarray:
    """100/1000 balanceados e aninhados; 10000 usa todo o teste oficial."""
    labels = labels.reshape(-1)
    classes = np.unique(labels)
    if labels.size != 10000 or not np.array_equal(classes, np.arange(10)):
        raise ValueError("esperada a divisao oficial de teste MNIST com 10 classes")
    rng = np.random.default_rng(seed)
    per_class: list[np.ndarray] = []
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        if indices.size < 100:
            raise ValueError(f"classe {class_id} tem menos de 100 exemplos")
        per_class.append(indices)
    balanced_thousand = np.stack([indices[:100] for indices in per_class], axis=1).reshape(-1)
    remainder = np.concatenate([indices[100:] for indices in per_class])
    rng.shuffle(remainder)
    order = np.concatenate([balanced_thousand, remainder])
    if np.unique(order).size != 10000:
        raise RuntimeError("ordem estratificada nao cobre o teste exatamente uma vez")
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


def metric_stats(values_ms: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values_ms, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise RuntimeError("latencia nao finita encontrada")
    return {
        "latency_mean_ms": float(np.mean(flat)),
        "latency_median_ms": float(np.median(flat)),
        "latency_std_ms": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
        "latency_cv_percent": 100.0 * float(np.std(flat, ddof=1)) / float(np.mean(flat)) if flat.size > 1 else 0.0,
        "latency_p90_ms": float(np.percentile(flat, 90)),
        "latency_p95_ms": float(np.percentile(flat, 95)),
        "latency_p99_ms": float(np.percentile(flat, 99)),
        "latency_min_ms": float(np.min(flat)),
        "latency_max_ms": float(np.max(flat)),
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


def run_capture(command: list[str]) -> tuple[int, str]:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    output = process.stdout
    if process.stderr:
        output += ("\n" if output else "") + process.stderr
    return process.returncode, output.strip()


def query_gpu_metadata() -> dict[str, str]:
    fields = (
        "index,name,uuid,pci.bus_id,driver_version,pstate,memory.total,memory.used,"
        "power.limit,clocks.max.sm,clocks.max.memory,compute_cap"
    )
    code, output = run_capture(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    if code != 0:
        raise RuntimeError(f"consulta nvidia-smi falhou: {output}")
    values = next(csv.reader([output.splitlines()[0]], skipinitialspace=True))
    return dict(zip(fields.split(","), [value.strip() for value in values], strict=True))


def write_snapshot(output_dir: Path, name: str) -> None:
    code, output = run_capture(["nvidia-smi", "-q"])
    (output_dir / f"nvidia_smi_{name}.txt").write_text(
        f"exit_code={code}\n{output}\n", encoding="utf-8"
    )


def load_rows(path: Path) -> list[dict[str, str]]:
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


def create_or_resume_latencies(path: Path, shape: tuple[int, int], restart: bool) -> np.memmap:
    if path.exists() and restart:
        path.unlink()
    if path.exists():
        stored = np.load(path, mmap_mode="r+")
        if stored.shape != shape or stored.dtype != np.float32:
            raise ValueError(
                f"checkpoint {path}: {stored.shape}/{stored.dtype}; esperado {shape}/float32"
            )
        return stored
    stored = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    stored[:] = np.nan
    stored.flush()
    return stored


def distribution_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = finite_values(values)
    if not array.size:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def summarize_telemetry(
    samples: list[TelemetrySample], duration_s: float, baseline_w: float, image_count: int
) -> dict[str, float | int | str]:
    power = finite_values(sample.power_instant_w for sample in samples)
    total_power_w = float(np.mean(power)) if power.size else math.nan
    dynamic_power_w = max(total_power_w - baseline_w, 0.0) if math.isfinite(total_power_w) else math.nan
    total_energy_j = total_power_w * duration_s if math.isfinite(total_power_w) else math.nan
    dynamic_energy_j = dynamic_power_w * duration_s if math.isfinite(dynamic_power_w) else math.nan
    states = sorted({sample.pstate for sample in samples if sample.pstate})
    result: dict[str, float | int | str] = {
        "power_samples": len(samples),
        "pstates": ";".join(states),
        "power_total_mean_w": total_power_w,
        "power_total_std_w": float(np.std(power, ddof=1)) if power.size > 1 else (0.0 if power.size else math.nan),
        "power_total_min_w": float(np.min(power)) if power.size else math.nan,
        "power_total_max_w": float(np.max(power)) if power.size else math.nan,
        "power_sensor_1s_mean_w": finite_mean(sample.power_average_w for sample in samples),
        "power_dynamic_mean_w": dynamic_power_w,
        "energy_total_j": total_energy_j,
        "energy_dynamic_j": dynamic_energy_j,
        "energy_per_inference_mj": total_energy_j * 1000.0 / image_count,
        "dynamic_energy_per_inference_mj": dynamic_energy_j * 1000.0 / image_count,
    }
    result.update(distribution_stats((s.utilization_gpu_percent for s in samples), "gpu_utilization_percent"))
    result.update(distribution_stats((s.utilization_memory_percent for s in samples), "memory_utilization_percent"))
    result.update(distribution_stats((s.memory_used_mib for s in samples), "memory_used_mib"))
    result.update(distribution_stats((s.clock_sm_mhz for s in samples), "gpu_clock_sm_mhz"))
    result.update(distribution_stats((s.clock_memory_mhz for s in samples), "gpu_clock_memory_mhz"))
    result.update(distribution_stats((s.temperature_c for s in samples), "gpu_temperature_c"))
    result.update(distribution_stats((s.fan_speed_percent for s in samples), "fan_speed_percent"))
    return result


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


def benchmark_sample_size(
    *,
    infer: Callable[[tf.Tensor], tf.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    dataset_indices: np.ndarray,
    sample_size: int,
    repetitions: int,
    seed: int,
    output_dir: Path,
    baseline_w: float,
    sampler: NvidiaSmiSampler,
    restart: bool,
) -> tuple[list[dict[str, object]], np.memmap, int]:
    latency_path = output_dir / f"latencias_n{sample_size}.npy"
    pass_path = output_dir / f"passagens_n{sample_size}.csv"
    prediction_path = output_dir / f"predicoes_n{sample_size}.csv"
    if restart:
        for path in (pass_path, prediction_path):
            if path.exists():
                path.unlink()
    latencies = create_or_resume_latencies(latency_path, (repetitions, sample_size), restart)
    existing_rows = load_rows(pass_path)
    completed = len(existing_rows)
    if completed > repetitions:
        raise ValueError(f"{pass_path} tem mais ciclos que o solicitado")
    if completed and not np.all(np.isfinite(latencies[:completed])):
        raise RuntimeError("CSV de ciclos e checkpoint de latencias discordam")

    rows: list[dict[str, object]] = [dict(row) for row in existing_rows]
    correct_reference = int(existing_rows[0]["correct"]) if existing_rows else None
    digest_reference = existing_rows[0].get("prediction_sha256") if existing_rows else None
    rng = np.random.default_rng(seed + sample_size * 1009)
    permutations = [rng.permutation(sample_size) for _ in range(repetitions)]

    for repetition in range(completed, repetitions):
        order = permutations[repetition]
        predictions = np.empty(sample_size, dtype=np.int64)
        logits = np.empty((sample_size, 10), dtype=np.float32) if repetition == 0 else None
        started_utc = datetime.now(timezone.utc).isoformat()
        start_s = time.perf_counter()
        for position, image_index in enumerate(order):
            with tf.device("/GPU:0"):
                batch = tf.convert_to_tensor(images[image_index : image_index + 1])
                item_start_ns = time.perf_counter_ns()
                output = infer(batch)
                host_output = output.numpy()
                item_end_ns = time.perf_counter_ns()
            if "GPU:0" not in output.device.upper():
                raise RuntimeError(f"saida fora da GPU: {output.device}")
            if not np.all(np.isfinite(host_output)):
                raise RuntimeError("saida nao finita encontrada")
            latencies[repetition, position] = (item_end_ns - item_start_ns) / 1_000_000.0
            predictions[image_index] = int(np.argmax(host_output[0]))
            if logits is not None:
                logits[image_index] = host_output[0]
        end_s = time.perf_counter()
        ended_utc = datetime.now(timezone.utc).isoformat()
        duration_s = end_s - start_s
        correct = int(np.count_nonzero(predictions == labels))
        prediction_digest = hashlib.sha256(predictions.tobytes()).hexdigest()
        if correct_reference is None:
            correct_reference = correct
            digest_reference = prediction_digest
        elif correct != correct_reference or prediction_digest != digest_reference:
            raise RuntimeError(
                f"predicoes nao deterministicas no ciclo {repetition + 1}: "
                f"acertos={correct}, hash={prediction_digest}"
            )
        if repetition == 0 and logits is not None:
            write_predictions(prediction_path, dataset_indices, labels, predictions, logits)

        row: dict[str, object] = {
            "sample_size": sample_size,
            "repetition": repetition + 1,
            "batch_size": 1,
            "correct": correct,
            "accuracy_percent": 100.0 * correct / sample_size,
            "prediction_sha256": prediction_digest,
            "wall_time_s": duration_s,
            "throughput_fps": sample_size / duration_s,
            **metric_stats(np.asarray(latencies[repetition], dtype=np.float64)),
            **summarize_telemetry(
                sampler.between(start_s, end_s), duration_s, baseline_w, sample_size
            ),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "monotonic_start_s": start_s,
            "monotonic_end_s": end_s,
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


def aggregate_results(
    sample_size: int,
    prefixes: list[int],
    rows: list[dict[str, object]],
    latencies: np.ndarray,
    correct: int,
) -> list[dict[str, object]]:
    aggregates: list[dict[str, object]] = []
    accuracy_low, accuracy_high = wilson_interval(correct, sample_size)
    for prefix in prefixes:
        selected_rows = rows[:prefix]
        selected_latency = np.asarray(latencies[:prefix], dtype=np.float64)
        pass_means = selected_latency.mean(axis=1)
        latency_ci_low, latency_ci_high = mean_ci95(pass_means)
        fps_values = np.asarray([float(row["throughput_fps"]) for row in selected_rows])
        fps_ci_low, fps_ci_high = mean_ci95(fps_values)
        wall_time = sum(float(row["wall_time_s"]) for row in selected_rows)
        result: dict[str, object] = {
            "model_variant": "no_softmax_logits",
            "device": "GPU",
            "precision": "float32",
            "batch_size": 1,
            "unique_images": sample_size,
            "repetitions": prefix,
            "collected_inferences": sample_size * prefix,
            "correct_unique_images": correct,
            "accuracy_percent": 100.0 * correct / sample_size,
            "accuracy_wilson95_low_percent": 100.0 * accuracy_low,
            "accuracy_wilson95_high_percent": 100.0 * accuracy_high,
            **metric_stats(selected_latency),
            "latency_mean_pass_ci95_low_ms": latency_ci_low,
            "latency_mean_pass_ci95_high_ms": latency_ci_high,
            "throughput_mean_fps": float(np.mean(fps_values)),
            "throughput_median_fps": float(np.median(fps_values)),
            "throughput_std_fps": float(np.std(fps_values, ddof=1)) if fps_values.size > 1 else 0.0,
            "throughput_mean_ci95_low_fps": fps_ci_low,
            "throughput_mean_ci95_high_fps": fps_ci_high,
            "throughput_effective_fps": sample_size * prefix / wall_time,
            "wall_time_total_s": wall_time,
        }
        mean_columns = [
            "power_total_mean_w",
            "power_total_std_w",
            "power_total_min_w",
            "power_total_max_w",
            "power_sensor_1s_mean_w",
            "power_dynamic_mean_w",
            "energy_per_inference_mj",
            "dynamic_energy_per_inference_mj",
            "gpu_utilization_percent_mean",
            "gpu_utilization_percent_std",
            "memory_utilization_percent_mean",
            "memory_used_mib_mean",
            "gpu_clock_sm_mhz_mean",
            "gpu_clock_sm_mhz_std",
            "gpu_clock_memory_mhz_mean",
            "gpu_temperature_c_mean",
            "gpu_temperature_c_max",
            "fan_speed_percent_mean",
        ]
        for column in mean_columns:
            result[column] = finite_mean(float(row[column]) for row in selected_rows)
        result["energy_total_j"] = sum(float(row["energy_total_j"]) for row in selected_rows)
        result["energy_dynamic_j"] = sum(float(row["energy_dynamic_j"]) for row in selected_rows)
        result["power_samples_total"] = sum(int(float(row["power_samples"])) for row in selected_rows)
        result["prediction_sha256"] = str(selected_rows[0]["prediction_sha256"])
        aggregates.append(result)
    return aggregates


def write_telemetry(path: Path, samples: list[TelemetrySample]) -> None:
    rows = [asdict(sample) for sample in samples]
    write_rows(path, rows)


def write_report(path: Path, results: list[dict[str, object]], metadata: dict[str, object]) -> None:
    primary = next(
        row for row in results
        if row["unique_images"] == max(metadata["sample_sizes"])
        and row["repetitions"] == metadata["maximum_repetitions_executed"]
    )
    lines = [
        "# Resultados — LeNet/MNIST sem softmax na GPU",
        "",
        "## Resultado principal",
        "",
        "O resultado principal usa todas as 10.000 imagens oficiais de teste e 100 ciclos. "
        "Os pontos de 10, 20 e 50 ciclos sao prefixos da mesma coleta.",
        "",
        "| Metrica | Resultado |",
        "|---|---:|",
        f"| GPU | {metadata['gpu']['name']} |",
        f"| Precisao | float32 |",
        f"| Batch | 1 |",
        f"| Imagens unicas | {primary['unique_images']} |",
        f"| Ciclos | {primary['repetitions']} |",
        f"| Inferencias cronometradas | {primary['collected_inferences']} |",
        f"| Acertos | {primary['correct_unique_images']} |",
        f"| Acuracia | {primary['accuracy_percent']:.4f}% |",
        f"| IC95 Wilson | [{primary['accuracy_wilson95_low_percent']:.4f}%; {primary['accuracy_wilson95_high_percent']:.4f}%] |",
        f"| Latencia media | {primary['latency_mean_ms']:.4f} ms |",
        f"| IC95 da media por ciclo | [{primary['latency_mean_pass_ci95_low_ms']:.4f}; {primary['latency_mean_pass_ci95_high_ms']:.4f}] ms |",
        f"| Mediana | {primary['latency_median_ms']:.4f} ms |",
        f"| Desvio padrao | {primary['latency_std_ms']:.4f} ms |",
        f"| Coeficiente de variacao | {primary['latency_cv_percent']:.2f}% |",
        f"| p90 / p95 / p99 | {primary['latency_p90_ms']:.4f} / {primary['latency_p95_ms']:.4f} / {primary['latency_p99_ms']:.4f} ms |",
        f"| Minimo / maximo | {primary['latency_min_ms']:.4f} / {primary['latency_max_ms']:.4f} ms |",
        f"| Vazao media | {primary['throughput_mean_fps']:.3f} inferencias/s |",
        f"| Vazao efetiva global | {primary['throughput_effective_fps']:.3f} inferencias/s |",
        f"| Potencia total media | {primary['power_total_mean_w']:.3f} W |",
        f"| Potencia dinamica media | {primary['power_dynamic_mean_w']:.3f} W |",
        f"| Energia total/inferencia | {primary['energy_per_inference_mj']:.4f} mJ |",
        f"| Energia dinamica/inferencia | {primary['dynamic_energy_per_inference_mj']:.4f} mJ |",
        f"| Utilizacao GPU media | {primary['gpu_utilization_percent_mean']:.3f}% |",
        f"| Memoria GPU usada | {primary['memory_used_mib_mean']:.3f} MiB |",
        f"| Clock SM medio | {primary['gpu_clock_sm_mhz_mean']:.3f} MHz |",
        f"| Temperatura media / maxima | {primary['gpu_temperature_c_mean']:.3f} / {primary['gpu_temperature_c_max']:.3f} C |",
        f"| Amostras nvidia-smi | {primary['power_samples_total']} |",
        f"| Potencia ociosa | {metadata['idle_power_mean_w']:.3f} W |",
        "",
        "## Convergencia de todas as series",
        "",
        "| Imagens | Ciclos | Inferencias | Acuracia (%) | Media (ms) | Mediana (ms) | p95 (ms) | FPS medio | FPS efetivo | Potencia (W) | Energia/inf. (mJ) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['unique_images']} | {row['repetitions']} | {row['collected_inferences']} "
            f"| {row['accuracy_percent']:.3f} | {row['latency_mean_ms']:.4f} "
            f"| {row['latency_median_ms']:.4f} | {row['latency_p95_ms']:.4f} "
            f"| {row['throughput_mean_fps']:.2f} | {row['throughput_effective_fps']:.2f} "
            f"| {row['power_total_mean_w']:.3f} | {row['energy_per_inference_mj']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Escopo metodologico",
            "",
            "Inferencia serial, sincrona e batch 1. A entrada ja esta normalizada em float32/[0,1]. "
            "A latencia comeca depois de `tf.convert_to_tensor` e termina depois de `.numpy()`, "
            "que sincroniza a GPU e materializa os 10 logits no host. O throughput inclui laco Python, "
            "criacao do tensor, inferencia, sincronizacao, argmax e armazenamento da predicao.",
            "",
            "Nao foram usados XLA, TensorRT, mixed precision, quantizacao, batching maior que 1 ou "
            "inferencias concorrentes. Nenhum outlier foi removido. A potencia e da placa GPU, medida "
            "pelo sensor NVIDIA a cada 100 ms; energia e uma estimativa integrada, e nao uma medicao na tomada.",
            "",
            "Os conjuntos de 100 e 1.000 imagens sao balanceados e aninhados. O conjunto de 10.000 "
            "usa toda a divisao oficial do MNIST e, portanto, mantem sua distribuicao real por classe.",
            "",
            "LUT, FF, DSP, BRAM, URAM e frequencia atingida da implementacao nao se aplicam a GPU; "
            "essas metricas pertencem aos fluxos hls4ml/Vitis AI.",
            "",
            "## Evidencias",
            "",
            "- `resultados.csv`: agregados completos.",
            "- `passagens_n*.csv`: uma linha por ciclo.",
            "- `latencias_n*.npy`: cada latencia individual.",
            "- `predicoes_n*.csv`: rotulo, predicao e logits da primeira passagem.",
            "- `telemetria_nvidia_smi.csv`: telemetria bruta.",
            "- `metadados.json`: ambiente, arquitetura, hashes e configuracao.",
            "- `nvidia_smi_antes.txt` e `nvidia_smi_depois.txt`: snapshots integrais.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    physical_gpu, logical_gpu = configure_gpu()
    gpu_metadata = query_gpu_metadata()
    write_snapshot(args.output_dir, "antes")

    with tf.device("/GPU:0"):
        model = tf.keras.models.load_model(args.model, compile=False)
    architecture = validate_model(model)
    infer = compile_inference_graph(model)

    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = (test_images.astype(np.float32) / 255.0)[..., np.newaxis]
    test_labels = test_labels.reshape(-1).astype(np.int64)
    order = make_nested_stratified_order(test_labels, args.seed)

    for index in order[: args.warmup]:
        with tf.device("/GPU:0"):
            warmup_output = infer(tf.convert_to_tensor(test_images[index : index + 1]))
            warmup_host = warmup_output.numpy()
    if "GPU:0" not in warmup_output.device.upper() or not np.all(np.isfinite(warmup_host)):
        raise RuntimeError(f"aquecimento invalido, dispositivo={warmup_output.device}")

    sampler = NvidiaSmiSampler(args.telemetry_ms)
    sampler.start()
    baseline_start_s = time.perf_counter()
    if args.baseline_seconds:
        time.sleep(args.baseline_seconds)
    baseline_end_s = time.perf_counter()
    baseline_samples = sampler.between(baseline_start_s, baseline_end_s)
    baseline_w = finite_mean(sample.power_instant_w for sample in baseline_samples)
    if not math.isfinite(baseline_w):
        print("AVISO: telemetria de potencia indisponivel", file=sys.stderr)

    prefixes = sorted(set(args.repetitions))
    max_repetitions = max(prefixes)
    all_results: list[dict[str, object]] = []
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        for sample_size in sorted(set(args.sample_sizes)):
            selected = order[:sample_size]
            images = np.ascontiguousarray(test_images[selected])
            labels = np.ascontiguousarray(test_labels[selected])
            rows, latencies, correct = benchmark_sample_size(
                infer=infer,
                images=images,
                labels=labels,
                dataset_indices=selected,
                sample_size=sample_size,
                repetitions=max_repetitions,
                seed=args.seed,
                output_dir=args.output_dir,
                baseline_w=baseline_w,
                sampler=sampler,
                restart=args.restart,
            )
            all_results.extend(
                aggregate_results(sample_size, prefixes, rows, latencies, correct)
            )
    finally:
        sampler.stop()
        write_telemetry(args.output_dir / "telemetria_nvidia_smi.csv", sampler.samples)
        write_snapshot(args.output_dir, "depois")

    class_counts = {
        str(class_id): int(np.count_nonzero(test_labels == class_id)) for class_id in range(10)
    }
    build_info = tf.sysconfig.get_build_info()
    metadata: dict[str, object] = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "model_sha256": sha256_file(args.model),
        "architecture": architecture,
        "input_normalization": "uint8 -> float32 / 255.0; canal adicionado ao final",
        "dataset": "MNIST official test split",
        "dataset_test_images": int(test_images.shape[0]),
        "dataset_class_counts": class_counts,
        "sampling": "100 e 1000 balanceados/aninhados; 10000 = teste oficial completo",
        "seed": args.seed,
        "sample_sizes": sorted(set(args.sample_sizes)),
        "repetition_prefixes": prefixes,
        "maximum_repetitions_executed": max_repetitions,
        "batch_size": 1,
        "precision": "float32",
        "execution_mode": "tf.function graph, autograph=False, jit_compile=False",
        "synchronization": "output.numpy() em cada inferencia",
        "warmup_inferences": args.warmup,
        "telemetry_interval_ms": args.telemetry_ms,
        "baseline_duration_s": args.baseline_seconds,
        "baseline_monotonic_start_s": baseline_start_s,
        "baseline_monotonic_end_s": baseline_end_s,
        "idle_power_mean_w": baseline_w,
        "idle_power_std_w": finite_std(s.power_instant_w for s in baseline_samples),
        "idle_power_samples": len(baseline_samples),
        "telemetry_errors": sampler.errors(),
        "gpu": gpu_metadata,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_build_info": build_info,
        "physical_gpu": str(physical_gpu),
        "logical_gpu": str(logical_gpu),
        "limitations": [
            "GPU tambem dirige a interface grafica do sistema",
            "nvidia-smi amostra em 100 ms, muito mais lento que uma inferencia",
            "potencia representa a placa GPU, nao o sistema completo",
            "ciclos sequenciais podem apresentar autocorrelacao temporal",
        ],
    }
    (args.output_dir / "metadados.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_rows(args.output_dir / "resultados.csv", all_results)
    write_report(args.output_dir / "RESULTADOS.md", all_results, metadata)
    print(f"Resultados: {args.output_dir / 'RESULTADOS.md'}", flush=True)


if __name__ == "__main__":
    main()
