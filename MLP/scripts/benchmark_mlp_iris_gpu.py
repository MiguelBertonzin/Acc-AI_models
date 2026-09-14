#!/usr/bin/env python3
"""Validação funcional, estatística e energética da MLP Iris em uma GPU NVIDIA.

O protocolo mede inferência batch 1, serial e síncrona. A entrada já está
padronizada antes do cronômetro individual e ``Tensor.numpy()`` força a
sincronização e materialização da saída no host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TELEMETRY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw.instant",
    "power.draw.average",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "fan.speed",
]
NUMERIC_TELEMETRY_FIELDS = set(TELEMETRY_FIELDS) - {
    "timestamp",
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "pstate",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "iris_mlp_clean.h5")
    parser.add_argument("--scaler", type=Path, default=root / "iris_scaler.joblib")
    parser.add_argument("--output-dir", type=Path, default=root / "GPU" / "resultados")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--telemetry-ms", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.cycles < 2 or args.warmup < 1 or args.bootstrap < 100:
        parser.error("use cycles >= 2, warmup >= 1 e bootstrap >= 100")
    if args.telemetry_ms < 20 or args.idle_seconds < 1:
        parser.error("use telemetry-ms >= 20 e idle-seconds >= 1")
    return args


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(to_builtin(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout


def write_snapshot(output_dir: Path, stem: str) -> None:
    for suffix, command in [
        ("", ["nvidia-smi"]),
        ("_detalhado", ["nvidia-smi", "-q"]),
    ]:
        code, output = run_capture(command)
        (output_dir / f"nvidia_smi_{stem}{suffix}.txt").write_text(
            f"exit_code={code}\n{output}", encoding="utf-8"
        )


class NvidiaSmiSampler:
    def __init__(self, gpu_index: int, interval_ms: int) -> None:
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.records: list[dict[str, Any]] = []
        self.stderr = ""
        self._ready = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        query = ",".join(TELEMETRY_FIELDS)
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self._process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15):
            self.stop()
            raise RuntimeError("nvidia-smi não produziu telemetria em até 15 s")

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            received_ns = time.perf_counter_ns()
            values = next(csv.reader([line.strip()], skipinitialspace=True))
            if len(values) != len(TELEMETRY_FIELDS):
                continue
            row: dict[str, Any] = {"received_perf_counter_ns": received_ns}
            for field, raw in zip(TELEMETRY_FIELDS, values):
                raw = raw.strip()
                if field in NUMERIC_TELEMETRY_FIELDS:
                    try:
                        row[field] = float(raw)
                    except ValueError:
                        row[field] = math.nan
                else:
                    row[field] = raw
            self.records.append(row)
            self._ready.set()

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._process.stderr is not None:
            self.stderr = self._process.stderr.read()


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    from scipy.stats import norm

    if total == 0:
        return math.nan, math.nan
    z = float(norm.ppf(0.5 + confidence / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def expected_calibration_error(y_true: Any, probabilities: Any, bins: int = 10) -> tuple[float, float]:
    import numpy as np

    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(selected):
            continue
        gap = abs(float(np.mean(correct[selected])) - float(np.mean(confidence[selected])))
        ece += float(np.mean(selected)) * gap
        mce = max(mce, gap)
    return ece, mce


def bootstrap_intervals(y_true: Any, probabilities: Any, iterations: int, seed: int) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == class_id) for class_id in range(len(CLASS_NAMES))]
    values = {"accuracy": [], "macro_f1": [], "log_loss": [], "brier_multiclass": [], "roc_auc_ovr_macro": []}
    for _ in range(iterations):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        ys = y_true[sampled]
        ps = probabilities[sampled]
        predictions = np.argmax(ps, axis=1)
        one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[ys]
        values["accuracy"].append(accuracy_score(ys, predictions))
        values["macro_f1"].append(f1_score(ys, predictions, average="macro", zero_division=0))
        values["log_loss"].append(log_loss(ys, ps, labels=list(range(len(CLASS_NAMES)))))
        values["brier_multiclass"].append(float(np.mean(np.sum((ps - one_hot) ** 2, axis=1))))
        values["roc_auc_ovr_macro"].append(roc_auc_score(ys, ps, multi_class="ovr", average="macro"))
    result: dict[str, Any] = {"method": "bootstrap estratificado por classe, percentil", "iterations": iterations}
    for name, metric_values in values.items():
        lower, upper = np.percentile(metric_values, [2.5, 97.5])
        result[name] = {"lower_95": float(lower), "upper_95": float(upper)}
    return result


def statistical_validation(y_true: Any, probabilities: Any, bootstrap: int, seed: int) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import binomtest
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
        top_k_accuracy_score,
    )

    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    prediction = np.argmax(probabilities, axis=1)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[y_true]
    confusion = confusion_matrix(y_true, prediction, labels=list(range(len(CLASS_NAMES))))
    correct = int(np.sum(prediction == y_true))
    accuracy_ci = wilson_interval(correct, len(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    per_class: dict[str, Any] = {}
    total = int(np.sum(confusion))
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fn = int(np.sum(confusion[class_id, :]) - tp)
        fp = int(np.sum(confusion[:, class_id]) - tp)
        tn = total - tp - fn - fp
        recall_ci = wilson_interval(tp, tp + fn)
        specificity_ci = wilson_interval(tn, tn + fp)
        per_class[class_name] = {
            "support": int(support[class_id]),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": float(precision[class_id]),
            "recall_sensitivity": float(recall[class_id]),
            "recall_ci95_wilson": recall_ci,
            "specificity": tn / (tn + fp) if tn + fp else math.nan,
            "specificity_ci95_wilson": specificity_ci,
            "f1": float(f1[class_id]),
        }
    ece, mce = expected_calibration_error(y_true, probabilities)
    probability_sums = np.sum(probabilities, axis=1)
    binomial = binomtest(correct, len(y_true), p=1 / len(CLASS_NAMES), alternative="greater")
    return {
        "holdout_samples": int(len(y_true)),
        "correct": correct,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "accuracy_ci95_wilson": accuracy_ci,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, prediction)),
        "cohen_kappa": float(cohen_kappa_score(y_true, prediction)),
        "top2_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=2, labels=[0, 1, 2])),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")),
        "roc_auc_ovr_weighted": float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")),
        "average_precision_macro": float(average_precision_score(one_hot, probabilities, average="macro")),
        "ece_10_bins": ece,
        "mce_10_bins": mce,
        "probabilities_finite": bool(np.all(np.isfinite(probabilities))),
        "probability_sum_max_abs_error": float(np.max(np.abs(probability_sums - 1.0))),
        "confusion_matrix_rows_true_cols_pred": confusion,
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, prediction, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "exact_binomial_test_vs_random_1_over_3": {
            "alternative": "accuracy > 1/3",
            "p_value": float(binomial.pvalue),
        },
        "bootstrap_ci95": bootstrap_intervals(y_true, probabilities, bootstrap, seed),
    }


def infer_all(infer: Callable[[Any], Any], x_data: Any, expected_device: str) -> tuple[Any, str]:
    import numpy as np
    import tensorflow as tf

    outputs = []
    actual_device = ""
    for row in x_data:
        tensor = tf.convert_to_tensor(row[np.newaxis, :], dtype=tf.float32)
        output = infer(tensor)
        actual_device = output.device
        if expected_device.upper() not in actual_device.upper():
            raise RuntimeError(f"saída em {actual_device}; esperado {expected_device}")
        outputs.append(output.numpy()[0])
    return np.asarray(outputs, dtype=np.float64), actual_device


def paired_cpu_gpu_validation(y_true: Any, gpu_probabilities: Any, cpu_probabilities: Any) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import binomtest

    gpu_prediction = np.argmax(gpu_probabilities, axis=1)
    cpu_prediction = np.argmax(cpu_probabilities, axis=1)
    gpu_correct = gpu_prediction == y_true
    cpu_correct = cpu_prediction == y_true
    cpu_only = int(np.sum(cpu_correct & ~gpu_correct))
    gpu_only = int(np.sum(gpu_correct & ~cpu_correct))
    discordant = cpu_only + gpu_only
    mcnemar_p = 1.0 if discordant == 0 else float(binomtest(min(cpu_only, gpu_only), discordant, 0.5).pvalue)
    absolute = np.abs(gpu_probabilities - cpu_probabilities)
    return {
        "prediction_agreement": float(np.mean(gpu_prediction == cpu_prediction)),
        "all_predictions_equal": bool(np.array_equal(gpu_prediction, cpu_prediction)),
        "maximum_absolute_probability_difference": float(np.max(absolute)),
        "mean_absolute_probability_difference": float(np.mean(absolute)),
        "allclose_rtol_1e_5_atol_1e_6": bool(np.allclose(gpu_probabilities, cpu_probabilities, rtol=1e-5, atol=1e-6)),
        "mcnemar_exact": {
            "cpu_correct_gpu_wrong": cpu_only,
            "gpu_correct_cpu_wrong": gpu_only,
            "discordant_pairs": discordant,
            "p_value_two_sided": mcnemar_p,
        },
    }


def telemetry_frame(records: list[dict[str, Any]]) -> Any:
    import pandas as pd

    frame = pd.DataFrame(records)
    if not frame.empty:
        origin = int(frame["received_perf_counter_ns"].iloc[0])
        frame.insert(1, "seconds_since_first_sample", (frame["received_perf_counter_ns"] - origin) / 1e9)
    return frame


def interval_frame(frame: Any, start_ns: int, end_ns: int) -> Any:
    if frame.empty:
        return frame
    return frame[(frame["received_perf_counter_ns"] >= start_ns) & (frame["received_perf_counter_ns"] <= end_ns)]


def integrate_power(frame: Any, start_ns: int, end_ns: int, field: str = "power.draw.instant") -> float:
    import numpy as np

    valid = frame[["received_perf_counter_ns", field]].dropna().sort_values("received_perf_counter_ns")
    if len(valid) < 2:
        return math.nan
    x = valid["received_perf_counter_ns"].to_numpy(dtype=np.float64) / 1e9
    y = valid[field].to_numpy(dtype=np.float64)
    start_s, end_s = start_ns / 1e9, end_ns / 1e9
    inside = (x > start_s) & (x < end_s)
    integration_x = np.concatenate(([start_s], x[inside], [end_s]))
    integration_y = np.interp(integration_x, x, y)
    return float(np.trapz(integration_y, integration_x))


def summarize_telemetry(frame: Any, start_ns: int, end_ns: int, baseline_power_w: float) -> dict[str, Any]:
    import numpy as np

    selected = interval_frame(frame, start_ns, end_ns)
    duration_s = (end_ns - start_ns) / 1e9
    energy_j = integrate_power(frame, start_ns, end_ns)
    result: dict[str, Any] = {
        "duration_s": duration_s,
        "samples": int(len(selected)),
        "sampling_interval_nominal_ms": None,
        "energy_total_j": energy_j,
        "power_baseline_w": baseline_power_w,
        "energy_dynamic_j": max(energy_j - baseline_power_w * duration_s, 0.0) if math.isfinite(energy_j) else math.nan,
    }
    for field in [
        "power.draw.instant",
        "power.draw.average",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "temperature.gpu",
        "clocks.current.sm",
        "clocks.current.memory",
        "fan.speed",
    ]:
        values = selected[field].dropna().to_numpy(dtype=np.float64) if field in selected else np.asarray([])
        if len(values):
            result[field] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    if len(selected) >= 2:
        diffs_ms = np.diff(selected["received_perf_counter_ns"].to_numpy(dtype=np.float64)) / 1e6
        result["sampling_interval_observed_ms"] = {
            "mean": float(np.mean(diffs_ms)), "median": float(np.median(diffs_ms)), "maximum": float(np.max(diffs_ms))
        }
    if "pstate" in selected:
        result["pstate_counts"] = selected["pstate"].value_counts().to_dict()
    return result


def performance_rows(latencies_ms: Any, cycle_durations_s: Any, cycle_end_ns: list[int], telemetry: Any,
                     benchmark_start_ns: int, baseline_power_w: float) -> list[dict[str, Any]]:
    import numpy as np
    from scipy.stats import t

    cycles, samples_per_cycle = latencies_ms.shape
    prefixes = sorted(set([p for p in [10, 20, 50, 100, 200, 500, 1000] if p <= cycles] + [cycles]))
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        values = latencies_ms[:prefix].reshape(-1)
        cycle_means = np.mean(latencies_ms[:prefix], axis=1)
        mean_of_cycles = float(np.mean(cycle_means))
        sem = float(np.std(cycle_means, ddof=1) / math.sqrt(prefix))
        critical = float(t.ppf(0.975, df=prefix - 1))
        total_inferences = prefix * samples_per_cycle
        total_time = float(np.sum(cycle_durations_s[:prefix]))
        interval_end_ns = cycle_end_ns[prefix - 1]
        energy_j = integrate_power(telemetry, benchmark_start_ns, interval_end_ns)
        duration_wall_s = (interval_end_ns - benchmark_start_ns) / 1e9
        dynamic_j = max(energy_j - baseline_power_w * duration_wall_s, 0.0) if math.isfinite(energy_j) else math.nan
        throughputs = samples_per_cycle / cycle_durations_s[:prefix]
        rows.append({
            "cycles": prefix,
            "samples_per_cycle": samples_per_cycle,
            "inferences": total_inferences,
            "latency_mean_ms": float(np.mean(values)),
            "latency_mean_ci95_lower_ms": mean_of_cycles - critical * sem,
            "latency_mean_ci95_upper_ms": mean_of_cycles + critical * sem,
            "latency_median_ms": float(np.median(values)),
            "latency_std_sample_ms": float(np.std(values, ddof=1)),
            "latency_cv_percent": float(100 * np.std(values, ddof=1) / np.mean(values)),
            "latency_p90_ms": float(np.percentile(values, 90)),
            "latency_p95_ms": float(np.percentile(values, 95)),
            "latency_p99_ms": float(np.percentile(values, 99)),
            "latency_min_ms": float(np.min(values)),
            "latency_max_ms": float(np.max(values)),
            "throughput_cycle_mean_fps": float(np.mean(throughputs)),
            "throughput_cycle_std_fps": float(np.std(throughputs, ddof=1)),
            "throughput_effective_fps": total_inferences / total_time,
            "cycle_time_total_s": total_time,
            "power_energy_window_s": duration_wall_s,
            "energy_total_j": energy_j,
            "energy_total_per_inference_j": energy_j / total_inferences,
            "energy_dynamic_per_inference_j": dynamic_j / total_inferences,
        })
    return rows


def write_predictions(path: Path, indices: Any, x_raw: Any, x_scaled: Any, y_true: Any,
                      gpu_probabilities: Any, cpu_probabilities: Any) -> None:
    prediction = gpu_probabilities.argmax(axis=1)
    columns = ["dataset_index", "true_id", "true_class", "predicted_id", "predicted_class", "correct"]
    columns += [f"raw_{name}" for name in FEATURE_NAMES]
    columns += [f"scaled_{name}" for name in FEATURE_NAMES]
    columns += [f"gpu_probability_{name}" for name in CLASS_NAMES]
    columns += [f"cpu_probability_{name}" for name in CLASS_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row_id in range(len(y_true)):
            row: dict[str, Any] = {
                "dataset_index": int(indices[row_id]),
                "true_id": int(y_true[row_id]),
                "true_class": CLASS_NAMES[int(y_true[row_id])],
                "predicted_id": int(prediction[row_id]),
                "predicted_class": CLASS_NAMES[int(prediction[row_id])],
                "correct": bool(prediction[row_id] == y_true[row_id]),
            }
            row.update({f"raw_{name}": float(x_raw[row_id, i]) for i, name in enumerate(FEATURE_NAMES)})
            row.update({f"scaled_{name}": float(x_scaled[row_id, i]) for i, name in enumerate(FEATURE_NAMES)})
            row.update({f"gpu_probability_{name}": float(gpu_probabilities[row_id, i]) for i, name in enumerate(CLASS_NAMES)})
            row.update({f"cpu_probability_{name}": float(cpu_probabilities[row_id, i]) for i, name in enumerate(CLASS_NAMES)})
            writer.writerow(row)


def generate_report(output_dir: Path, stats: dict[str, Any], parity: dict[str, Any], determinism: dict[str, Any],
                    performance: list[dict[str, Any]], telemetry: dict[str, Any], metadata: dict[str, Any]) -> None:
    final = performance[-1]
    ci = stats["accuracy_ci95_wilson"]
    power = telemetry.get("power.draw.instant", {})
    gpu_util = telemetry.get("utilization.gpu", {})
    clock = telemetry.get("clocks.current.sm", {})
    temperature = telemetry.get("temperature.gpu", {})
    memory_used = telemetry.get("memory.used", {})
    memory_util = telemetry.get("utilization.memory", {})
    fan = telemetry.get("fan.speed", {})
    dynamic_power_w = telemetry.get("energy_dynamic_j", math.nan) / telemetry.get("duration_s", math.nan)
    pstates = ", ".join(f"{name}: {count}" for name, count in telemetry.get("pstate_counts", {}).items())
    report = f"""# Resultados — MLP Iris em GPU

## Protocolo

- Modelo: MLP 4–8–8–3 com softmax, 139 parâmetros, `float32`.
- Holdout: 30 amostras balanceadas, reconstruído com `test_size=0.2`, `random_state=42` e `stratify=y`.
- Desempenho: batch 1, serial e síncrono; entrada padronizada antes do cronômetro e saída materializada no host.
- Repetições: {final['cycles']} ciclos × {final['samples_per_cycle']} amostras = {final['inferences']} inferências.
- GPU: {metadata['gpu']['name']}; driver {metadata['gpu']['driver_version']}.

## Validação estatística

| Métrica | Resultado |
|---|---:|
| Acurácia | {stats['correct']}/{stats['holdout_samples']} = {100 * stats['accuracy']:.2f}% |
| IC95% da acurácia (Wilson) | [{100 * ci[0]:.2f}%, {100 * ci[1]:.2f}%] |
| Acurácia balanceada | {stats['balanced_accuracy']:.6f} |
| F1 macro | {stats['macro_f1']:.6f} |
| MCC | {stats['matthews_correlation_coefficient']:.6f} |
| Cohen kappa | {stats['cohen_kappa']:.6f} |
| Log loss | {stats['log_loss']:.8f} |
| Brier multiclasse | {stats['brier_multiclass']:.8f} |
| ROC AUC OvR macro | {stats['roc_auc_ovr_macro']:.6f} |
| ECE (10 bins) | {stats['ece_10_bins']:.8f} |
| Teste binomial exato contra 1/3 | p = {stats['exact_binomial_test_vs_random_1_over_3']['p_value']:.3e} |

Matriz de confusão (linhas = classe real; colunas = classe predita):

```text
{stats['confusion_matrix_rows_true_cols_pred']}
```

O holdout é pequeno. O IC95% de Wilson e os intervalos bootstrap devem acompanhar a estimativa pontual; repetir uma mesma amostra melhora a caracterização temporal, mas não aumenta o tamanho estatístico da avaliação de acurácia.

## Paridade NumPy/CPU × TensorFlow/GPU e determinismo

- Concordância de classes NumPy/CPU × TensorFlow/GPU: {100 * parity['prediction_agreement']:.2f}%.
- Maior diferença absoluta de probabilidade NumPy/CPU × TensorFlow/GPU: {parity['maximum_absolute_probability_difference']:.3e}.
- `allclose(rtol=1e-5, atol=1e-6)`: {parity['allclose_rtol_1e_5_atol_1e_6']}.
- Maior diferença entre execuções repetidas na GPU: {determinism['maximum_absolute_probability_difference']:.3e}.
- Concordância do `argmax` nas repetições: {100 * determinism['argmax_agreement']:.2f}%.

## Desempenho e energia

| Métrica | Resultado |
|---|---:|
| Latência média | {final['latency_mean_ms']:.6f} ms |
| IC95% da latência média | [{final['latency_mean_ci95_lower_ms']:.6f}, {final['latency_mean_ci95_upper_ms']:.6f}] ms |
| Mediana | {final['latency_median_ms']:.6f} ms |
| Desvio-padrão | {final['latency_std_sample_ms']:.6f} ms |
| p95 / p99 | {final['latency_p95_ms']:.6f} / {final['latency_p99_ms']:.6f} ms |
| Mínimo / máximo | {final['latency_min_ms']:.6f} / {final['latency_max_ms']:.6f} ms |
| Vazão média por ciclo | {final['throughput_cycle_mean_fps']:.2f} inferências/s |
| Vazão efetiva global | {final['throughput_effective_fps']:.2f} inferências/s |
| Potência média da placa | {power.get('mean', math.nan):.3f} W |
| Potência ociosa após aquecimento | {telemetry['power_baseline_w']:.3f} W |
| Potência dinâmica média estimada | {dynamic_power_w:.3f} W |
| Energia total da campanha | {telemetry['energy_total_j']:.3f} J |
| Energia dinâmica da campanha | {telemetry['energy_dynamic_j']:.3f} J |
| Energia total por inferência | {1000 * final['energy_total_per_inference_j']:.6f} mJ |
| Energia dinâmica por inferência | {1000 * final['energy_dynamic_per_inference_j']:.6f} mJ |
| Utilização GPU média / máxima | {gpu_util.get('mean', math.nan):.2f}% / {gpu_util.get('maximum', math.nan):.2f}% |
| Utilização de memória média / máxima | {memory_util.get('mean', math.nan):.2f}% / {memory_util.get('maximum', math.nan):.2f}% |
| Memória usada média / máxima | {memory_used.get('mean', math.nan):.1f} / {memory_used.get('maximum', math.nan):.1f} MiB |
| Clock SM médio / máximo | {clock.get('mean', math.nan):.1f} / {clock.get('maximum', math.nan):.1f} MHz |
| Temperatura média / máxima | {temperature.get('mean', math.nan):.1f} / {temperature.get('maximum', math.nan):.1f} °C |
| Ventoinha média / máxima | {fan.get('mean', math.nan):.1f}% / {fan.get('maximum', math.nan):.1f}% |
| P-state observado | {pstates} |
| Telemetria | {telemetry['samples']} amostras em {telemetry['duration_s']:.3f} s |

A energia integra `power.draw.instant` do sensor da placa. A energia dinâmica subtrai a potência ociosa medida depois do aquecimento. O intervalo nominal da telemetria ({metadata['benchmark']['telemetry_ms']} ms) é muito maior que uma única inferência; a integração é interpretável sobre a campanha inteira, não por amostra isolada. O consumo do host não está incluído. A diferença entre potência ativa e ociosa é muito pequena frente à incerteza aproximada de ±5 W registrada na metodologia; portanto, a energia dinâmica deve ser tratada como estimativa de baixa confiança.

## Evidências

- `validacao_estatistica.json`: métricas, intervalos e métricas por classe.
- `validacao_cpu_gpu.json`: paridade numérica e teste pareado.
- `metricas_desempenho.csv`: prefixos de ciclos e convergência.
- `passagens_gpu.csv`: uma linha por ciclo.
- `latencias_gpu.npy`: todas as latências individuais.
- `predicoes_holdout.csv`: entradas, probabilidades CPU/GPU e classes.
- `telemetria_nvidia_smi.csv`: amostras brutas do sensor.
- `metadados.json`: ambiente, hashes, índices da divisão e configuração.
- `nvidia_smi_*.txt`: snapshots completos antes e depois.
"""
    (output_dir / "RESULTADOS.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.scaler = args.scaler.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"diretório não vazio: {args.output_dir}; use outro caminho ou --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import joblib
    import keras
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import tensorflow as tf
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.set_soft_device_placement(False)
    physical_gpus = tf.config.list_physical_devices("GPU")
    if len(physical_gpus) != 1:
        raise RuntimeError(f"esperada exatamente uma GPU visível; encontradas: {physical_gpus}")
    try:
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
    except RuntimeError:
        pass

    X, y = load_iris(return_X_y=True)
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = joblib.load(args.scaler)
    reference_scaler = StandardScaler().fit(X_train.astype(np.float32))
    scaler_validation = {
        "class": scaler.__class__.__name__,
        "n_samples_seen": int(scaler.n_samples_seen_),
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "matches_reconstructed_train_mean_atol_2e_7": bool(np.allclose(scaler.mean_, reference_scaler.mean_, rtol=0, atol=2e-7)),
        "matches_reconstructed_train_scale_atol_2e_7": bool(np.allclose(scaler.scale_, reference_scaler.scale_, rtol=0, atol=2e-7)),
        "load_warnings": [str(item.message) for item in caught],
    }
    if scaler.n_samples_seen_ != 120 or not scaler_validation["matches_reconstructed_train_mean_atol_2e_7"]:
        raise RuntimeError("o scaler não corresponde à divisão treino/teste reconstruída")
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    with tf.device("/GPU:0"):
        gpu_model = tf.keras.models.load_model(args.model, compile=False)
    expected_layers = [(8, "relu"), (8, "relu"), (3, "softmax")]
    observed_layers = [(layer.units, layer.activation.__name__) for layer in gpu_model.layers]
    if gpu_model.input_shape != (None, 4) or observed_layers != expected_layers or gpu_model.count_params() != 139:
        raise RuntimeError(f"arquitetura inesperada: input={gpu_model.input_shape}, layers={observed_layers}, params={gpu_model.count_params()}")

    with tf.device("/GPU:0"):
        @tf.function(input_signature=[tf.TensorSpec((1, 4), tf.float32)], autograph=False, jit_compile=False)
        def gpu_infer(batch: Any) -> Any:
            with tf.device("/GPU:0"):
                return gpu_model(batch, training=False)

    for index in range(args.warmup):
        row = X_test_scaled[index % len(X_test_scaled)]
        output = gpu_infer(tf.convert_to_tensor(row[np.newaxis, :], dtype=tf.float32))
        _ = output.numpy()
    if "GPU:0" not in output.device.upper():
        raise RuntimeError(f"aquecimento terminou em {output.device}, não em GPU:0")

    gpu_probabilities, gpu_device = infer_all(gpu_infer, X_test_scaled, "GPU:0")
    repeated = []
    for _ in range(5):
        values, _ = infer_all(gpu_infer, X_test_scaled, "GPU:0")
        repeated.append(values)
    repeated_array = np.asarray(repeated)
    determinism = {
        "passes": 5,
        "maximum_absolute_probability_difference": float(np.max(np.abs(repeated_array - gpu_probabilities))),
        "argmax_agreement": float(np.mean(np.argmax(repeated_array, axis=2) == np.argmax(gpu_probabilities, axis=1))),
        "bitwise_equal": bool(np.array_equal(repeated_array, np.broadcast_to(gpu_probabilities, repeated_array.shape))),
    }

    weights = [np.asarray(value, dtype=np.float32) for value in gpu_model.get_weights()]
    cpu_hidden1 = np.maximum(np.matmul(X_test_scaled, weights[0]) + weights[1], 0.0)
    cpu_hidden2 = np.maximum(np.matmul(cpu_hidden1, weights[2]) + weights[3], 0.0)
    cpu_logits = np.matmul(cpu_hidden2, weights[4]) + weights[5]
    cpu_shifted = cpu_logits - np.max(cpu_logits, axis=1, keepdims=True)
    cpu_exponentials = np.exp(cpu_shifted)
    cpu_probabilities = (cpu_exponentials / np.sum(cpu_exponentials, axis=1, keepdims=True)).astype(np.float64)
    cpu_device = "NumPy CPU float32"
    stats = statistical_validation(y_test, gpu_probabilities, args.bootstrap, args.seed)
    parity = paired_cpu_gpu_validation(y_test, gpu_probabilities, cpu_probabilities)
    write_predictions(
        args.output_dir / "predicoes_holdout.csv", test_indices, X_test, X_test_scaled, y_test,
        gpu_probabilities, cpu_probabilities
    )

    static_query = [
        "nvidia-smi", f"--id={args.gpu_index}",
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,memory.total,power.limit,clocks.max.sm,clocks.max.memory,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    static_code, static_output = run_capture(static_query)
    static_values = next(csv.reader([static_output.strip()], skipinitialspace=True)) if static_code == 0 else []
    gpu_static_fields = ["index", "name", "uuid", "pci_bus_id", "driver_version", "pstate_before", "memory_total_mib", "power_limit_w", "clock_max_sm_mhz", "clock_max_memory_mhz", "compute_capability"]
    gpu_static = dict(zip(gpu_static_fields, [value.strip() for value in static_values]))
    if static_code != 0:
        raise RuntimeError(f"consulta estática nvidia-smi falhou: {static_output}")

    write_snapshot(args.output_dir, "antes")
    sampler = NvidiaSmiSampler(args.gpu_index, args.telemetry_ms)
    sampler.start()
    idle_start_ns = time.perf_counter_ns()
    time.sleep(args.idle_seconds)
    idle_end_ns = time.perf_counter_ns()

    latencies_ms = np.empty((args.cycles, len(X_test_scaled)), dtype=np.float64)
    cycle_durations_s = np.empty(args.cycles, dtype=np.float64)
    cycle_end_ns: list[int] = []
    passage_rows: list[dict[str, Any]] = []
    order_rng = np.random.default_rng(args.seed)
    expected_predictions = np.argmax(gpu_probabilities, axis=1)
    benchmark_start_ns = time.perf_counter_ns()
    progress_every = max(1, args.cycles // 10)
    for cycle in range(args.cycles):
        order = order_rng.permutation(len(X_test_scaled))
        observed_predictions = np.empty(len(order), dtype=np.int64)
        cycle_start_ns = time.perf_counter_ns()
        for position, sample_id in enumerate(order):
            batch = tf.convert_to_tensor(X_test_scaled[sample_id][np.newaxis, :], dtype=tf.float32)
            start_ns = time.perf_counter_ns()
            value = gpu_infer(batch).numpy()[0]
            end_ns = time.perf_counter_ns()
            latencies_ms[cycle, position] = (end_ns - start_ns) / 1e6
            observed_predictions[position] = int(np.argmax(value))
        cycle_finish_ns = time.perf_counter_ns()
        cycle_end_ns.append(cycle_finish_ns)
        duration_s = (cycle_finish_ns - cycle_start_ns) / 1e9
        cycle_durations_s[cycle] = duration_s
        expected_in_order = expected_predictions[order]
        if not np.array_equal(observed_predictions, expected_in_order):
            raise RuntimeError(f"não determinismo de classe detectado no ciclo {cycle + 1}")
        values = latencies_ms[cycle]
        passage_rows.append({
            "cycle": cycle + 1,
            "inferences": len(order),
            "correct_unique": int(np.sum(expected_predictions == y_test)),
            "latency_mean_ms": float(np.mean(values)),
            "latency_median_ms": float(np.median(values)),
            "latency_std_sample_ms": float(np.std(values, ddof=1)),
            "latency_p95_ms": float(np.percentile(values, 95)),
            "latency_min_ms": float(np.min(values)),
            "latency_max_ms": float(np.max(values)),
            "cycle_duration_s": duration_s,
            "throughput_fps": len(order) / duration_s,
        })
        if (cycle + 1) % progress_every == 0 or cycle + 1 == args.cycles:
            print(f"progresso: {cycle + 1}/{args.cycles} ciclos", flush=True)
    benchmark_end_ns = time.perf_counter_ns()
    time.sleep(max(0.5, args.telemetry_ms / 1000 * 2))
    sampler.stop()
    write_snapshot(args.output_dir, "depois")
    if sampler.stderr.strip():
        (args.output_dir / "telemetria_nvidia_smi_stderr.txt").write_text(sampler.stderr, encoding="utf-8")

    telemetry = telemetry_frame(sampler.records)
    telemetry.to_csv(args.output_dir / "telemetria_nvidia_smi.csv", index=False)
    idle_selected = interval_frame(telemetry, idle_start_ns, idle_end_ns)
    if len(idle_selected) < 2:
        raise RuntimeError("telemetria insuficiente na janela ociosa")
    baseline_power_w = float(idle_selected["power.draw.instant"].dropna().mean())
    telemetry_summary = summarize_telemetry(telemetry, benchmark_start_ns, benchmark_end_ns, baseline_power_w)
    telemetry_summary["sampling_interval_nominal_ms"] = args.telemetry_ms
    performance = performance_rows(
        latencies_ms, cycle_durations_s, cycle_end_ns, telemetry,
        benchmark_start_ns, baseline_power_w
    )

    np.save(args.output_dir / "latencias_gpu.npy", latencies_ms)
    pd.DataFrame(passage_rows).to_csv(args.output_dir / "passagens_gpu.csv", index=False)
    pd.DataFrame(performance).to_csv(args.output_dir / "metricas_desempenho.csv", index=False)
    write_json(args.output_dir / "validacao_estatistica.json", stats)
    write_json(args.output_dir / "validacao_cpu_gpu.json", parity)
    write_json(args.output_dir / "determinismo_gpu.json", determinism)
    write_json(args.output_dir / "resumo_telemetria.json", telemetry_summary)

    tf_build = tf.sysconfig.get_build_info()
    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "protocol": "batch 1, serial, síncrono, entrada pré-padronizada, saída no host",
        "benchmark": {
            "cycles": args.cycles,
            "samples_per_cycle": len(X_test_scaled),
            "warmup_inferences": args.warmup,
            "idle_seconds": args.idle_seconds,
            "telemetry_ms": args.telemetry_ms,
            "bootstrap_iterations": args.bootstrap,
            "seed_order_and_bootstrap": args.seed,
            "jit_compile": False,
            "mixed_precision": False,
            "batch_size": 1,
        },
        "dataset": {
            "name": "sklearn Iris",
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
            "train_indices": train_indices,
            "test_indices": test_indices,
            "test_class_counts": np.bincount(y_test, minlength=3),
            "feature_names": FEATURE_NAMES,
            "class_map": {index: name for index, name in enumerate(CLASS_NAMES)},
        },
        "model": {
            "path": str(args.model),
            "sha256": sha256(args.model),
            "input_shape": gpu_model.input_shape,
            "layers": observed_layers,
            "parameters": gpu_model.count_params(),
            "dtype": "float32",
        },
        "scaler": {"path": str(args.scaler), "sha256": sha256(args.scaler), **scaler_validation},
        "placement": {"gpu_output_device": gpu_device, "cpu_output_device": cpu_device, "soft_device_placement": False},
        "gpu": gpu_static,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow_built_with_cuda": tf.test.is_built_with_cuda(),
            "tensorflow_build_info": tf_build,
        },
    }
    write_json(args.output_dir / "metadados.json", metadata)
    generate_report(args.output_dir, stats, parity, determinism, performance, telemetry_summary, metadata)

    print(f"resultados gravados em: {args.output_dir}")
    print(f"acurácia: {100 * stats['accuracy']:.2f}%")
    print(f"latência média: {performance[-1]['latency_mean_ms']:.6f} ms")
    print(f"vazão efetiva: {performance[-1]['throughput_effective_fps']:.2f} inferências/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
