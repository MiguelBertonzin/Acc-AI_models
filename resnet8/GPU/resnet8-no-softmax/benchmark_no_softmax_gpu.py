#!/usr/bin/env python3
"""Reproducible batch-1 GPU benchmark for the logits ResNet-8 model.

The maximum number of repetitions is executed once for each sample size. Results
for smaller repetition counts are cumulative prefixes of the same measurements.
This makes the 10/20/50/100-run comparisons paired and reproducible.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_MODEL = PROJECT_ROOT / "resnet8_cifar10_keras3_no_softmax.h5"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "resultados"


@dataclass
class TelemetrySample:
    monotonic_s: float
    timestamp: str
    power_instant_w: float
    power_average_w: float
    utilization_percent: float
    clock_sm_mhz: float
    temperature_c: float


class NvidiaSmiSampler:
    """Collect board telemetry using one persistent nvidia-smi process."""

    FIELDS = (
        "timestamp,power.draw.instant,power.draw.average,utilization.gpu,"
        "clocks.current.sm,temperature.gpu"
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
            self._errors.put("nvidia-smi was not found")
            return
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
                if len(fields) != 6:
                    continue
                self.samples.append(
                    TelemetrySample(
                        monotonic_s=time.perf_counter(),
                        timestamp=fields[0],
                        power_instant_w=self._number(fields[1]),
                        power_average_w=self._number(fields[2]),
                        utilization_percent=self._number(fields[3]),
                        clock_sm_mhz=self._number(fields[4]),
                        temperature_c=self._number(fields[5]),
                    )
                )
        except Exception as exc:  # telemetry must not invalidate inference data
            self._errors.put(f"telemetry reader failed: {exc!r}")

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
        return [s for s in self.samples if start_s <= s.monotonic_s <= end_s]

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
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Overwrite checkpoints for the requested sample sizes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.sample_sizes or min(args.sample_sizes) <= 0 or max(args.sample_sizes) > 10000:
        raise ValueError("sample sizes must be between 1 and 10000")
    if not args.repetitions or min(args.repetitions) <= 0:
        raise ValueError("repetitions must be positive")
    if args.warmup < 1:
        raise ValueError("warmup must be positive")
    if args.baseline_seconds < 0:
        raise ValueError("baseline-seconds cannot be negative")
    if args.telemetry_ms < 50:
        raise ValueError("telemetry-ms must be at least 50")


def configure_gpu() -> tuple[tf.config.PhysicalDevice, tf.config.LogicalDevice]:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("No TensorFlow GPU is available; CPU fallback is forbidden.")
    tf.config.set_visible_devices(physical[0], "GPU")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_op_determinism()
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError(f"Expected exactly one visible logical GPU, found {logical!r}")
    return physical[0], logical[0]


def compile_inference_graph(model: tf.keras.Model) -> Callable[[tf.Tensor], tf.Tensor]:
    """Trace the unchanged Keras model once, without XLA, for production-like calls."""

    @tf.function(
        input_signature=[tf.TensorSpec((1, 32, 32, 3), tf.float32)],
        autograph=False,
        jit_compile=False,
    )
    def infer(batch: tf.Tensor) -> tf.Tensor:
        with tf.device("/GPU:0"):
            return model(batch, training=False)

    return infer


def make_nested_stratified_order(labels: np.ndarray, seed: int) -> np.ndarray:
    """Return a balanced, nested CIFAR-10 order: 10/100/1000 per class."""
    labels = labels.reshape(-1)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    per_class: list[np.ndarray] = []
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        per_class.append(indices)
    counts = {len(indices) for indices in per_class}
    if classes.size != 10 or counts != {1000}:
        raise ValueError("Expected the standard CIFAR-10 test set (10 classes x 1000 images).")
    return np.stack(per_class, axis=1).reshape(-1)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else math.nan


def metric_stats(values_ms: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values_ms, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise RuntimeError("Non-finite latency found in completed measurements.")
    return {
        "latency_mean_ms": float(np.mean(flat)),
        "latency_median_ms": float(np.median(flat)),
        "latency_std_ms": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
        "latency_p95_ms": float(np.percentile(flat, 95)),
        "latency_min_ms": float(np.min(flat)),
        "latency_max_ms": float(np.max(flat)),
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return centre - margin, centre + margin


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Normal 95% CI over independent pass-level values (n >= 10 here)."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    margin = 1.959963984540054 * float(np.std(values, ddof=1)) / math.sqrt(values.size)
    return mean - margin, mean + margin


def query_gpu_metadata() -> dict[str, str]:
    fields = "name,uuid,driver_version,memory.total,power.limit"
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"]
    output = subprocess.check_output(command, text=True).strip().splitlines()[0]
    values = [item.strip() for item in output.split(",")]
    return dict(zip(fields.split(","), values, strict=True))


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
            raise ValueError(f"Checkpoint {path} has shape/dtype {stored.shape}/{stored.dtype}, expected {shape}/float32")
        return stored
    stored = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    stored[:] = np.nan
    stored.flush()
    return stored


def summarize_telemetry(
    samples: list[TelemetrySample], duration_s: float, baseline_w: float, image_count: int
) -> dict[str, float | int]:
    total_power_w = finite_mean(s.power_instant_w for s in samples)
    average_sensor_w = finite_mean(s.power_average_w for s in samples)
    dynamic_power_w = max(total_power_w - baseline_w, 0.0) if math.isfinite(total_power_w) else math.nan
    total_energy_j = total_power_w * duration_s if math.isfinite(total_power_w) else math.nan
    dynamic_energy_j = dynamic_power_w * duration_s if math.isfinite(dynamic_power_w) else math.nan
    return {
        "power_samples": len(samples),
        "power_total_mean_w": total_power_w,
        "power_sensor_1s_mean_w": average_sensor_w,
        "power_dynamic_mean_w": dynamic_power_w,
        "energy_total_j": total_energy_j,
        "energy_dynamic_j": dynamic_energy_j,
        "energy_per_inference_mj": total_energy_j * 1000.0 / image_count,
        "dynamic_energy_per_inference_mj": dynamic_energy_j * 1000.0 / image_count,
        "gpu_utilization_mean_percent": finite_mean(s.utilization_percent for s in samples),
        "gpu_clock_sm_mean_mhz": finite_mean(s.clock_sm_mhz for s in samples),
        "gpu_temperature_mean_c": finite_mean(s.temperature_c for s in samples),
    }


def benchmark_sample_size(
    *,
    infer: Callable[[tf.Tensor], tf.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
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
    if restart and pass_path.exists():
        pass_path.unlink()
    latencies = create_or_resume_latencies(latency_path, (repetitions, sample_size), restart)
    existing_rows = load_rows(pass_path)
    completed = len(existing_rows)
    if completed > repetitions:
        raise ValueError(f"{pass_path} has {completed} rows but only {repetitions} were requested")
    if completed and not np.all(np.isfinite(latencies[:completed])):
        raise RuntimeError("Pass CSV and latency checkpoint disagree; use --restart to overwrite them.")

    rows: list[dict[str, object]] = [dict(row) for row in existing_rows]
    correct_reference: int | None = int(existing_rows[0]["correct"]) if existing_rows else None
    rng = np.random.default_rng(seed + sample_size * 1009)
    permutations = [rng.permutation(sample_size) for _ in range(repetitions)]

    for repetition in range(completed, repetitions):
        order = permutations[repetition]
        predictions = np.empty(sample_size, dtype=np.int64)
        started_utc = datetime.now(timezone.utc).isoformat()
        start_s = time.perf_counter()
        for position, image_index in enumerate(order):
            # Normalization/data selection is outside the timed interval. Tensor creation
            # and device transfer are queued before the model invocation and synchronized
            # by .numpy(), so latency represents the batch-1 accelerator call end-to-end.
            with tf.device("/GPU:0"):
                batch = tf.convert_to_tensor(images[image_index : image_index + 1])
                item_start_ns = time.perf_counter_ns()
                output = infer(batch)
                host_output = output.numpy()
                item_end_ns = time.perf_counter_ns()
            if "GPU:0" not in output.device.upper():
                raise RuntimeError(f"Model output was not placed on GPU: {output.device}")
            latencies[repetition, position] = (item_end_ns - item_start_ns) / 1_000_000.0
            predictions[image_index] = int(np.argmax(host_output[0]))
        end_s = time.perf_counter()
        ended_utc = datetime.now(timezone.utc).isoformat()
        duration_s = end_s - start_s
        correct = int(np.count_nonzero(predictions == labels))
        if correct_reference is None:
            correct_reference = correct
        elif correct != correct_reference:
            raise RuntimeError(
                f"Non-deterministic predictions: pass 1 had {correct_reference} correct, "
                f"pass {repetition + 1} had {correct}."
            )
        pass_latency = np.asarray(latencies[repetition], dtype=np.float64)
        row: dict[str, object] = {
            "sample_size": sample_size,
            "repetition": repetition + 1,
            "batch_size": 1,
            "correct": correct,
            "accuracy_percent": 100.0 * correct / sample_size,
            "wall_time_s": duration_s,
            "throughput_fps": sample_size / duration_s,
            **metric_stats(pass_latency),
            **summarize_telemetry(
                sampler.between(start_s, end_s), duration_s, baseline_w, sample_size
            ),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
        }
        rows.append(row)
        latencies.flush()
        write_rows(pass_path, rows)
        print(
            f"n={sample_size:5d} pass={repetition + 1:3d}/{repetitions} "
            f"acc={row['accuracy_percent']:.2f}% mean={row['latency_mean_ms']:.3f} ms "
            f"fps={row['throughput_fps']:.1f}",
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
            "model_variant": "no_softmax",
            "device": "GPU",
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
            "throughput_std_fps": float(np.std(fps_values, ddof=1)) if fps_values.size > 1 else 0.0,
            "throughput_mean_ci95_low_fps": fps_ci_low,
            "throughput_mean_ci95_high_fps": fps_ci_high,
            "throughput_effective_fps": sample_size * prefix / wall_time,
            "wall_time_total_s": wall_time,
        }
        telemetry_columns = [
            "power_total_mean_w",
            "power_sensor_1s_mean_w",
            "power_dynamic_mean_w",
            "energy_per_inference_mj",
            "dynamic_energy_per_inference_mj",
            "gpu_utilization_mean_percent",
            "gpu_clock_sm_mean_mhz",
            "gpu_temperature_mean_c",
        ]
        for column in telemetry_columns:
            result[column] = finite_mean(float(row[column]) for row in selected_rows)
        result["power_samples_total"] = sum(int(float(row["power_samples"])) for row in selected_rows)
        aggregates.append(result)
    return aggregates


def write_markdown(path: Path, results: list[dict[str, object]], metadata: dict[str, object]) -> None:
    lines = [
        "# Resultados — ResNet-8 sem softmax na GPU",
        "",
        f"Modelo: `{metadata['model_path']}`  ",
        f"GPU: {metadata['gpu']['name']}  ",
        f"Seed: {metadata['seed']}  ",
        f"Potência ociosa média após aquecimento: {metadata['idle_power_mean_w']:.3f} W",
        "",
        "Cada linha resume os primeiros K ciclos da mesma execução de 100 ciclos. "
        "A acurácia e seu IC de Wilson usam somente imagens únicas; repetições não são "
        "tratadas como novas amostras de acurácia.",
        "",
        "| Imagens | Ciclos | Medições | Acurácia (%) | Latência média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | Vazão média (FPS) | Potência total (W) | Potência dinâmica (W) | Energia/inf. (mJ) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['unique_images']} | {row['repetitions']} | {row['collected_inferences']} "
            f"| {row['accuracy_percent']:.3f} | {row['latency_mean_ms']:.4f} "
            f"| {row['latency_median_ms']:.4f} | {row['latency_std_ms']:.4f} "
            f"| {row['latency_p95_ms']:.4f} | {row['throughput_mean_fps']:.3f} "
            f"| {row['power_total_mean_w']:.3f} | {row['power_dynamic_mean_w']:.3f} "
            f"| {row['energy_per_inference_mj']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Definições",
            "",
            "- Latência: tempo do grafo TensorFlow (`tf.function`, sem XLA) até a saída estar disponível "
            "no host (`.numpy()`), batch 1. Normalização e seleção da imagem ficam fora do cronômetro.",
            "- Vazão média: média aritmética da vazão de cada ciclo completo.",
            "- Potência total: potência da placa (`power.draw.instant`) amostrada pelo `nvidia-smi`.",
            "- Potência dinâmica: potência total menos a potência ociosa após o aquecimento, limitada a zero.",
            "- Energia por inferência: potência total média vezes o tempo do ciclo, dividida pelo número de imagens.",
            "- A telemetria tem resolução de 100 ms e precisão própria do sensor NVIDIA (aproximadamente ±5 W). "
            "Nos ciclos curtos de 100 imagens, potência e energia têm maior incerteza; consulte `power_samples_total` no CSV.",
            "- LUT, FF, DSP, BRAM, URAM e frequência atingida não se aplicam a esta execução em GPU; "
            "serão coletadas na implementação FPGA.",
            "",
            "Os dados completos estão em `resultados.csv`, as estatísticas de cada ciclo em "
            "`passagens_n*.csv` e todas as latências individuais em `latencias_n*.npy`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physical_gpu, logical_gpu = configure_gpu()
    gpu_metadata = query_gpu_metadata()

    with tf.device("/GPU:0"):
        model = tf.keras.models.load_model(args.model, compile=False)
    if len(model.outputs) != 1 or model.output_shape[-1] != 10:
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")
    final_layer = model.layers[-1]
    activation = tf.keras.activations.serialize(final_layer.activation)
    if activation != "linear":
        raise ValueError(f"Expected the logits model, found final activation {activation!r}")
    infer = compile_inference_graph(model)

    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_images = test_images.astype(np.float32) / 255.0
    test_labels = test_labels.reshape(-1).astype(np.int64)
    stratified_order = make_nested_stratified_order(test_labels, args.seed)

    # Confirm GPU placement and remove graph/kernel initialization from measurements.
    for index in stratified_order[: args.warmup]:
        with tf.device("/GPU:0"):
            warmup_output = infer(tf.convert_to_tensor(test_images[index : index + 1]))
            _ = warmup_output.numpy()
    if "GPU:0" not in warmup_output.device.upper():
        raise RuntimeError(f"Warm-up output is not on GPU: {warmup_output.device}")

    sampler = NvidiaSmiSampler(args.telemetry_ms)
    sampler.start()
    baseline_start = time.perf_counter()
    if args.baseline_seconds:
        time.sleep(args.baseline_seconds)
    baseline_end = time.perf_counter()
    baseline_samples = sampler.between(baseline_start, baseline_end)
    baseline_w = finite_mean(s.power_instant_w for s in baseline_samples)
    if not math.isfinite(baseline_w):
        baseline_w = math.nan
        print("WARNING: GPU power telemetry is unavailable; power/energy will be NaN.", file=sys.stderr)

    prefixes = sorted(set(args.repetitions))
    max_repetitions = max(prefixes)
    all_results: list[dict[str, object]] = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for sample_size in sorted(set(args.sample_sizes)):
            selected = stratified_order[:sample_size]
            images = np.ascontiguousarray(test_images[selected])
            labels = np.ascontiguousarray(test_labels[selected])
            rows, latencies, correct = benchmark_sample_size(
                infer=infer,
                images=images,
                labels=labels,
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

    metadata: dict[str, object] = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "model_final_activation": activation,
        "input_normalization": "float32 / 255.0",
        "dataset": "CIFAR-10 official test split",
        "sampling": "nested stratified fixed-seed order",
        "seed": args.seed,
        "sample_sizes": sorted(set(args.sample_sizes)),
        "repetition_prefixes": prefixes,
        "maximum_repetitions_executed": max_repetitions,
        "batch_size": 1,
        "execution_mode": "tf.function graph, jit_compile=False",
        "warmup_inferences": args.warmup,
        "telemetry_interval_ms": args.telemetry_ms,
        "baseline_duration_s": args.baseline_seconds,
        "idle_power_mean_w": baseline_w,
        "idle_power_samples": len(baseline_samples),
        "telemetry_errors": sampler.errors(),
        "gpu": gpu_metadata,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "physical_gpu": str(physical_gpu),
        "logical_gpu": str(logical_gpu),
    }
    (args.output_dir / "metadados.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_rows(args.output_dir / "resultados.csv", all_results)
    write_markdown(args.output_dir / "RESULTADOS.md", all_results, metadata)
    print(f"Results: {args.output_dir / 'RESULTADOS.md'}")


if __name__ == "__main__":
    main()
