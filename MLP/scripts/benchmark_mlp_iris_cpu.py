#!/usr/bin/env python3
"""Benchmark batch 1 da MLP Iris em CPU, com inference-only e end-to-end."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import benchmark_mlp_iris_gpu as core
import benchmark_mlp_iris_gpu_100k as common


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "iris_mlp_clean.h5")
    parser.add_argument("--scaler", type=Path, default=root / "iris_scaler.joblib")
    parser.add_argument("--gpu-reference-dir", type=Path, default=root / "GPU" / "resultados_100000")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inferences", type=int, default=100_000)
    parser.add_argument("--samples-per-passage", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--telemetry-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--require-rapl", action="store_true",
        help="aborta se o contador RAPL do pacote não estiver legível",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.inferences < 60 or args.samples_per_passage < 2 or args.samples_per_passage > 30:
        parser.error("use inferences >= 60 e samples-per-passage entre 2 e 30")
    if args.warmup < 1 or args.idle_seconds < 1 or args.telemetry_ms < 20:
        parser.error("use warmup >= 1, idle-seconds >= 1 e telemetry-ms >= 20")
    if args.output_dir is None:
        args.output_dir = root / "CPU" / f"resultados_{args.inferences}"
    return args


def read_float(path: Path) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return math.nan


def discover_rapl_domains() -> dict[str, dict[str, Any]]:
    """Descobre um pacote RAPL MSR e seus subdomínios, sem duplicar MMIO."""
    domains: dict[str, dict[str, Any]] = {}
    for package_path in sorted(Path("/sys/class/powercap").glob("intel-rapl:[0-9]*")):
        try:
            package_name = (package_path / "name").read_text(encoding="utf-8").strip()
            energy_path = package_path / "energy_uj"
            current = int(energy_path.read_text(encoding="utf-8").strip())
            maximum = int((package_path / "max_energy_range_uj").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        key = "package" if package_name.startswith("package-") else package_name.replace("-", "_")
        domains[key] = {
            "name": package_name,
            "energy_path": energy_path,
            "max_energy_range_uj": maximum,
            "initial_energy_uj": current,
        }
        for child in sorted(package_path.glob("intel-rapl:*:*")):
            try:
                child_name = (child / "name").read_text(encoding="utf-8").strip()
                child_energy_path = child / "energy_uj"
                child_current = int(child_energy_path.read_text(encoding="utf-8").strip())
                child_maximum = int((child / "max_energy_range_uj").read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            child_key = child_name.replace("-", "_")
            domains[child_key] = {
                "name": child_name,
                "energy_path": child_energy_path,
                "max_energy_range_uj": child_maximum,
                "initial_energy_uj": child_current,
            }
        break
    return domains


class RaplMeter:
    def __init__(self) -> None:
        self.domains = discover_rapl_domains()

    @property
    def available(self) -> bool:
        return "package" in self.domains

    def capture(self) -> dict[str, Any]:
        before_ns = time.perf_counter_ns()
        energies: dict[str, int] = {}
        for key, domain in self.domains.items():
            energies[key] = int(domain["energy_path"].read_text(encoding="utf-8").strip())
        after_ns = time.perf_counter_ns()
        return {"perf_counter_ns": (before_ns + after_ns) // 2, "energy_uj": energies}

    def delta(self, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
        duration_s = (end["perf_counter_ns"] - start["perf_counter_ns"]) / 1e9
        result: dict[str, Any] = {"duration_s": duration_s, "domains": {}}
        for key, domain in self.domains.items():
            delta_uj = end["energy_uj"][key] - start["energy_uj"][key]
            if delta_uj < 0:
                delta_uj += int(domain["max_energy_range_uj"])
            energy_j = delta_uj / 1e6
            result["domains"][key] = {
                "name": domain["name"],
                "energy_j": energy_j,
                "average_power_w": energy_j / duration_s,
            }
        return result


def discover_hwmon() -> dict[str, Any]:
    package_temperature: Path | None = None
    core_temperatures: list[Path] = []
    fan_input: Path | None = None
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (hwmon / "name").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        for label_path in sorted(hwmon.glob("temp*_label")):
            try:
                label = label_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            input_path = label_path.with_name(label_path.name.replace("_label", "_input"))
            if name == "coretemp" and label == "Package id 0":
                package_temperature = input_path
            elif name == "coretemp" and label.startswith("Core "):
                core_temperatures.append(input_path)
        if fan_input is None and name in {"dell_smm", "dell_ddv"}:
            candidates = sorted(hwmon.glob("fan*_input"))
            if candidates:
                fan_input = candidates[0]
    frequency_paths = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"))
    return {
        "package_temperature": package_temperature,
        "core_temperatures": core_temperatures,
        "fan_input": fan_input,
        "frequency_paths": frequency_paths,
    }


class CpuTelemetrySampler:
    def __init__(self, interval_ms: int, rapl: RaplMeter) -> None:
        import psutil

        self.interval_s = interval_ms / 1000.0
        self.process = psutil.Process()
        self.hwmon = discover_hwmon()
        self.rapl = rapl
        self.records: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("a telemetria CPU não iniciou")

    def _sample(self) -> dict[str, Any]:
        import psutil

        frequencies = [read_float(path) / 1000.0 for path in self.hwmon["frequency_paths"]]
        frequencies = [value for value in frequencies if math.isfinite(value)]
        core_temperatures = [read_float(path) / 1000.0 for path in self.hwmon["core_temperatures"]]
        core_temperatures = [value for value in core_temperatures if math.isfinite(value)]
        package_path = self.hwmon["package_temperature"]
        fan_path = self.hwmon["fan_input"]
        memory = self.process.memory_info()
        load1, load5, load15 = os.getloadavg()
        record = {
            "received_perf_counter_ns": time.perf_counter_ns(),
            "timestamp_local": datetime.now().astimezone().isoformat(),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "process_cpu_percent": self.process.cpu_percent(interval=None),
            "process_rss_mib": memory.rss / (1024 * 1024),
            "process_vms_mib": memory.vms / (1024 * 1024),
            "process_threads": self.process.num_threads(),
            "system_memory_percent": psutil.virtual_memory().percent,
            "load1": load1,
            "load5": load5,
            "load15": load15,
            "frequency_mean_mhz": sum(frequencies) / len(frequencies) if frequencies else math.nan,
            "frequency_min_mhz": min(frequencies) if frequencies else math.nan,
            "frequency_max_mhz": max(frequencies) if frequencies else math.nan,
            "package_temperature_c": read_float(package_path) / 1000.0 if package_path else math.nan,
            "core_temperature_mean_c": sum(core_temperatures) / len(core_temperatures) if core_temperatures else math.nan,
            "core_temperature_max_c": max(core_temperatures) if core_temperatures else math.nan,
            "fan_rpm": read_float(fan_path) if fan_path else math.nan,
        }
        if self.rapl.available:
            captured = self.rapl.capture()
            record["rapl_capture_perf_counter_ns"] = captured["perf_counter_ns"]
            for key, energy_uj in captured["energy_uj"].items():
                record[f"rapl_{key}_energy_uj"] = energy_uj
        return record

    def _loop(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            self.records.append(self._sample())
            self._ready.set()
            next_sample += self.interval_s
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def summarize_telemetry(frame: Any, start_ns: int, end_ns: int, nominal_ms: int) -> dict[str, Any]:
    import numpy as np

    selected = frame[
        (frame["received_perf_counter_ns"] >= start_ns)
        & (frame["received_perf_counter_ns"] <= end_ns)
    ]
    result: dict[str, Any] = {
        "duration_s": (end_ns - start_ns) / 1e9,
        "samples": int(len(selected)),
        "sampling_interval_nominal_ms": nominal_ms,
    }
    for field in [
        "system_cpu_percent", "process_cpu_percent", "process_rss_mib", "process_vms_mib",
        "process_threads", "system_memory_percent", "load1", "frequency_mean_mhz",
        "frequency_min_mhz", "frequency_max_mhz", "package_temperature_c",
        "core_temperature_mean_c", "core_temperature_max_c", "fan_rpm",
    ]:
        values = selected[field].dropna().to_numpy(dtype=np.float64)
        if len(values):
            result[field] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    if len(selected) >= 2:
        differences = np.diff(selected["received_perf_counter_ns"].to_numpy(dtype=np.float64)) / 1e6
        result["sampling_interval_observed_ms"] = {
            "mean": float(np.mean(differences)),
            "median": float(np.median(differences)),
            "maximum": float(np.max(differences)),
        }
    return result


def save_diagnostic(output_dir: Path, name: str, command: list[str]) -> dict[str, Any]:
    code, output = core.run_capture(command)
    (output_dir / f"{name}.txt").write_text(f"exit_code={code}\n{output}", encoding="utf-8")
    return {"command": command, "exit_code": code, "output": output}


def generate_readme(output_dir: Path, summary: dict[str, Any], telemetry: dict[str, Any],
                    validation: dict[str, Any], metadata: dict[str, Any]) -> None:
    inf = summary["inference_only"]
    e2e = summary["end_to_end"]
    unavailable = {"mean": math.nan, "median": math.nan, "p95": math.nan, "minimum": math.nan, "maximum": math.nan}
    process_cpu = telemetry.get("process_cpu_percent", unavailable)
    system_cpu = telemetry.get("system_cpu_percent", unavailable)
    frequency = telemetry.get("frequency_mean_mhz", unavailable)
    package_temp = telemetry.get("package_temperature_c", unavailable)
    rss = telemetry.get("process_rss_mib", unavailable)
    fan = telemetry.get("fan_rpm", unavailable)
    formatted = f"{summary['inferences']:,}".replace(",", ".")
    rapl = summary["rapl"]
    if rapl["available"]:
        package = rapl["benchmark"]["domains"]["package"]
        core_domain = rapl["benchmark"]["domains"].get("core")
        core_line = (
            f"| Energia dos núcleos | {core_domain['energy_j']:.6f} J |\n"
            f"| Potência média dos núcleos | {core_domain['average_power_w']:.6f} W |\n"
            if core_domain else ""
        )
        rapl_report = f"""## Energia RAPL da CPU

| Métrica | Resultado |
|---|---:|
| Fonte | Linux powercap intel-rapl / package-0 |
| Duração exata da janela RAPL | {rapl['benchmark']['duration_s']:.9f} s |
| Energia total do pacote | {package['energy_j']:.6f} J |
| Potência média do pacote | {package['average_power_w']:.6f} W |
| Potência ociosa do pacote | {rapl['idle_package_power_w']:.6f} W |
| Energia total por inferência | {1000 * rapl['total_energy_per_inference_j']:.6f} mJ |
| Potência dinâmica média | {rapl['dynamic_package_power_w']:.6f} W |
| Energia dinâmica por inferência | {1000 * rapl['dynamic_energy_per_inference_j']:.6f} mJ |
{core_line}
A energia principal é a do pacote completo; core e uncore são subdomínios e não são somados novamente. A energia dinâmica subtrai a potência ociosa medida durante {metadata['benchmark']['idle_seconds']:.1f} s. RAPL estima energia no pacote e não equivale a consumo na tomada.

"""
    else:
        rapl_report = """## Energia RAPL da CPU

Indisponível: o contador energy_uj do domínio package-0 não estava legível. Nenhum valor foi estimado.

"""
    report = f"""# MLP Iris em CPU — {formatted} inferências batch 1

## Resultados

| Janela / métrica | Resultado |
|---|---:|
| Inference-only média | {inf['mean_ms']:.6f} ms |
| Inference-only IC95% | [{inf['mean_ci95_lower_ms']:.6f}, {inf['mean_ci95_upper_ms']:.6f}] ms |
| Inference-only mediana | {inf['median_ms']:.6f} ms |
| Inference-only p95 / p99 | {inf['p95_ms']:.6f} / {inf['p99_ms']:.6f} ms |
| Inference-only mínimo / máximo | {inf['minimum_ms']:.6f} / {inf['maximum_ms']:.6f} ms |
| End-to-end média individual | {e2e['mean_ms']:.6f} ms |
| End-to-end IC95% | [{e2e['mean_ci95_lower_ms']:.6f}, {e2e['mean_ci95_upper_ms']:.6f}] ms |
| End-to-end mediana | {e2e['median_ms']:.6f} ms |
| End-to-end p95 / p99 | {e2e['p95_ms']:.6f} / {e2e['p99_ms']:.6f} ms |
| End-to-end média efetiva | {summary['end_to_end_effective_mean_ms']:.6f} ms |
| Vazão efetiva | {summary['throughput_effective_fps']:.2f} inferências/s |

## Telemetria CPU

| Métrica | Resultado |
|---|---:|
| Amostras durante benchmark | {telemetry['samples']} |
| Utilização do processo média / máxima | {process_cpu['mean']:.2f}% / {process_cpu['maximum']:.2f}% |
| Núcleos lógicos equivalentes médios | {process_cpu['mean'] / 100:.3f} |
| Utilização global média / máxima | {system_cpu['mean']:.2f}% / {system_cpu['maximum']:.2f}% |
| Frequência média / p95 / máxima | {frequency['mean']:.1f} / {frequency['p95']:.1f} / {frequency['maximum']:.1f} MHz |
| Temperatura do pacote média / máxima | {package_temp['mean']:.1f} / {package_temp['maximum']:.1f} °C |
| RSS do processo média / máxima | {rss['mean']:.1f} / {rss['maximum']:.1f} MiB |
| Fan média / máxima | {fan['mean']:.0f} / {fan['maximum']:.0f} RPM |

{rapl_report}## Validação funcional

- Arquitetura: 4–8–8–3, ReLU/ReLU/softmax, 139 parâmetros, `float32`.
- Execução confirmada em `{validation['cpu_output_device']}`, com nenhuma GPU visível e soft placement desabilitado.
- Holdout: 120 treino / 30 teste, estratificado, `random_state=42`.
- Acurácia: {validation['correct']}/30 = {100 * validation['accuracy']:.2f}%.
- Concordância com a GPU: {100 * validation['cpu_gpu_prediction_agreement']:.2f}%.
- Diferença máxima de probabilidade CPU×GPU: {validation['cpu_gpu_max_abs_probability_difference']:.3e}.
- Todas as {formatted} repetições mantiveram as classes da validação inicial.

## Metodologia

1. Definiu `CUDA_VISIBLE_DEVICES` vazio antes de importar TensorFlow e removeu todas as GPUs visíveis.
2. Desabilitou soft placement e fixou modelo e grafo em `/CPU:0`; uma saída fora da CPU abortaria o teste.
3. Usou TensorFlow {metadata['software']['tensorflow']} com oneDNN, `tf.function`, batch 1, `float32`, `jit_compile=False`, sem XLA, TensorRT, mixed precision ou quantização.
4. Reutilizou o mesmo modelo, scaler, holdout, mapa de classes e seed das campanhas GPU.
5. Executou {metadata['benchmark']['warmup_inferences']} inferências de aquecimento, excluídas da medição.
6. Mediu {metadata['benchmark']['idle_seconds']:.1f} s de baseline após aquecimento e amostrou telemetria, inclusive os acumuladores RAPL, a cada {metadata['benchmark']['telemetry_ms']} ms.
7. Executou exatamente {formatted} inferências em {summary['full_passages']} passagens completas de 30 e uma passagem final de {summary['final_passage_samples']}.
8. Permutou a ordem deterministicamente a cada passagem com seed {metadata['benchmark']['seed']}.
9. Não removeu outliers.
10. `Inference-only`: cronômetro depois de `tf.convert_to_tensor` e até o retorno de `.numpy()`.
11. `End-to-end` individual: antes da criação do tensor até depois de `.numpy()` e `argmax`.
12. Vazão efetiva: total de inferências dividido pela soma dos tempos completos das passagens, incluindo laço e armazenamento.
13. IC95%: distribuição t aplicada às médias das passagens completas; o ciclo parcial não participa do IC.
14. Telemetria: thread auxiliar registrando CPU global/processo, memória, frequência, temperatura, carga, fan e acumuladores RAPL.
15. Energia: diferença dos contadores cumulativos energy_uj lidos nas fronteiras exatas da campanha, com correção de rollover pelo max_energy_range_uj; potência média é energia dividida pela duração da própria janela RAPL.
16. Energia dinâmica: energia total menos potência média ociosa multiplicada pela duração ativa; resultados negativos são truncados em zero.

O TensorFlow pode usar múltiplas threads oneDNN mesmo em batch 1. Portanto, batch 1 descreve uma amostra por chamada, não execução limitada a um único núcleo.

## Ambiente e arquivos

- CPU: {metadata['cpu']['model_name']}; {metadata['cpu']['physical_cores']} núcleos físicos / {metadata['cpu']['logical_cpus']} CPUs lógicas.
- Governador: {metadata['cpu']['governors']}.
- Modelo SHA-256: `{metadata['model']['sha256']}`.
- Scaler SHA-256: `{metadata['scaler']['sha256']}`.
- `latencias_inference_only_cpu.npy` e `latencias_end_to_end_cpu.npy`: tempos individuais.
- `passagens_cpu.csv`: métricas por passagem.
- `telemetria_cpu.csv`: telemetria bruta.
- `metricas_resumo.json`, `resumo_telemetria.json`, `resumo_rapl.json`, `validacao_execucao.json` e `metadados.json`: resultados estruturados.
- `diagnostico_*.txt`: evidência da disponibilidade dos sensores.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.scaler = args.scaler.resolve()
    args.gpu_reference_dir = args.gpu_reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"diretório não vazio: {args.output_dir}; use --overwrite ou outro caminho")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    import joblib
    import keras
    import numpy as np
    import pandas as pd
    import psutil
    import sklearn
    import tensorflow as tf
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.set_visible_devices([], "GPU")
    tf.config.set_soft_device_placement(False)
    if tf.config.list_physical_devices("GPU") or tf.config.get_visible_devices("GPU"):
        raise RuntimeError("uma GPU permaneceu visível no ensaio CPU")

    X, y = load_iris(return_X_y=True)
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = joblib.load(args.scaler)
    fitted = StandardScaler().fit(X_train.astype(np.float32))
    scaler_matches = bool(
        scaler.n_samples_seen_ == 120
        and np.allclose(scaler.mean_, fitted.mean_, rtol=0, atol=2e-7)
        and np.allclose(scaler.scale_, fitted.scale_, rtol=0, atol=2e-7)
    )
    if not scaler_matches:
        raise RuntimeError("o scaler não coincide com a divisão reconstruída")
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(args.model, compile=False)
    layers = [(layer.units, layer.activation.__name__) for layer in model.layers]
    if model.input_shape != (None, 4) or layers != [(8, "relu"), (8, "relu"), (3, "softmax")] or model.count_params() != 139:
        raise RuntimeError("arquitetura inesperada")

    @tf.function(input_signature=[tf.TensorSpec((1, 4), tf.float32)], autograph=False, jit_compile=False)
    def cpu_infer(batch: Any) -> Any:
        with tf.device("/CPU:0"):
            return model(batch, training=False)

    for index in range(args.warmup):
        row = X_test_scaled[index % len(X_test_scaled)]
        warmed = cpu_infer(tf.convert_to_tensor(row[np.newaxis, :], dtype=tf.float32))
        _ = warmed.numpy()
    if "CPU:0" not in warmed.device.upper():
        raise RuntimeError(f"saída fora da CPU: {warmed.device}")

    cpu_probabilities, cpu_device = core.infer_all(cpu_infer, X_test_scaled, "CPU:0")
    gpu_predictions_frame = __import__("pandas").read_csv(args.gpu_reference_dir / ".." / "resultados" / "predicoes_holdout.csv")
    gpu_predictions = gpu_predictions_frame["predicted_id"].to_numpy(dtype=np.int64)
    gpu_probability_columns = [f"gpu_probability_{name}" for name in core.CLASS_NAMES]
    gpu_probabilities = gpu_predictions_frame[gpu_probability_columns].to_numpy(dtype=np.float64)
    cpu_predictions = np.argmax(cpu_probabilities, axis=1)
    validation = {
        "samples": 30,
        "correct": int(np.sum(cpu_predictions == y_test)),
        "accuracy": float(np.mean(cpu_predictions == y_test)),
        "cpu_gpu_prediction_agreement": float(np.mean(cpu_predictions == gpu_predictions)),
        "cpu_gpu_max_abs_probability_difference": float(np.max(np.abs(cpu_probabilities - gpu_probabilities))),
        "cpu_output_device": cpu_device,
        "probabilities_finite": bool(np.isfinite(cpu_probabilities).all()),
    }
    if validation["cpu_gpu_prediction_agreement"] != 1.0:
        raise RuntimeError("predições CPU divergiram da referência GPU")

    diagnostics = {
        "lscpu": save_diagnostic(args.output_dir, "diagnostico_lscpu", ["lscpu"]),
        "sensors_before": save_diagnostic(args.output_dir, "diagnostico_sensors_antes", ["sensors", "-j"]),
        "turbostat_access": save_diagnostic(
            args.output_dir, "diagnostico_turbostat_acesso",
            ["turbostat", "--Summary", "--quiet", "--no-perf", "--interval", "0.1", "--num_iterations", "1"],
        ),
        "perf_energy": save_diagnostic(args.output_dir, "diagnostico_perf_energia", ["perf", "stat", "-e", "power/energy-pkg/", "true"]),
    }

    rapl = RaplMeter()
    if args.require_rapl and not rapl.available:
        raise RuntimeError(
            "RAPL obrigatório, mas /sys/class/powercap/intel-rapl:0/energy_uj não está legível"
        )
    sampler = CpuTelemetrySampler(args.telemetry_ms, rapl)
    sampler.start()
    rapl_idle_start = rapl.capture() if rapl.available else None
    idle_start_ns = time.perf_counter_ns()
    time.sleep(args.idle_seconds)
    idle_end_ns = time.perf_counter_ns()
    rapl_idle_end = rapl.capture() if rapl.available else None

    process = psutil.Process()
    cpu_times_before = process.cpu_times()
    context_before = process.num_ctx_switches()
    inference_only_ms = np.empty(args.inferences, dtype=np.float64)
    end_to_end_ms = np.empty(args.inferences, dtype=np.float64)
    passage_rows: list[dict[str, Any]] = []
    passage_durations_s: list[float] = []
    passage_means_inference: list[float] = []
    passage_means_end_to_end: list[float] = []
    rng = np.random.default_rng(args.seed)
    completed = 0
    passage = 0
    total_passages = math.ceil(args.inferences / args.samples_per_passage)
    rapl_benchmark_start = rapl.capture() if rapl.available else None
    benchmark_start_ns = time.perf_counter_ns()
    progress_every = max(1, total_passages // 10)
    while completed < args.inferences:
        count = min(args.samples_per_passage, args.inferences - completed)
        order = rng.permutation(len(X_test_scaled))[:count]
        observed = np.empty(count, dtype=np.int64)
        passage_start_ns = time.perf_counter_ns()
        for position, sample_id in enumerate(order):
            global_index = completed + position
            end_to_end_start_ns = time.perf_counter_ns()
            batch = tf.convert_to_tensor(X_test_scaled[sample_id][np.newaxis, :], dtype=tf.float32)
            inference_start_ns = time.perf_counter_ns()
            host_output = cpu_infer(batch).numpy()[0]
            inference_end_ns = time.perf_counter_ns()
            observed[position] = int(np.argmax(host_output))
            end_to_end_end_ns = time.perf_counter_ns()
            inference_only_ms[global_index] = (inference_end_ns - inference_start_ns) / 1e6
            end_to_end_ms[global_index] = (end_to_end_end_ns - end_to_end_start_ns) / 1e6
        passage_end_ns = time.perf_counter_ns()
        if not np.array_equal(observed, cpu_predictions[order]):
            raise RuntimeError(f"não determinismo na passagem {passage + 1}")
        duration_s = (passage_end_ns - passage_start_ns) / 1e9
        current_inference = inference_only_ms[completed : completed + count]
        current_end = end_to_end_ms[completed : completed + count]
        passage_durations_s.append(duration_s)
        if count == args.samples_per_passage:
            passage_means_inference.append(float(np.mean(current_inference)))
            passage_means_end_to_end.append(float(np.mean(current_end)))
        passage_rows.append({
            "passage": passage + 1,
            "inferences": count,
            "inference_only_mean_ms": float(np.mean(current_inference)),
            "inference_only_median_ms": float(np.median(current_inference)),
            "inference_only_p95_ms": float(np.percentile(current_inference, 95)),
            "end_to_end_mean_ms": float(np.mean(current_end)),
            "end_to_end_median_ms": float(np.median(current_end)),
            "end_to_end_p95_ms": float(np.percentile(current_end, 95)),
            "passage_duration_s": duration_s,
            "throughput_effective_fps": count / duration_s,
        })
        completed += count
        passage += 1
        if passage % progress_every == 0 or completed == args.inferences:
            print(f"progresso: {completed}/{args.inferences} inferências", flush=True)
    benchmark_end_ns = time.perf_counter_ns()
    rapl_benchmark_end = rapl.capture() if rapl.available else None
    cpu_times_after = process.cpu_times()
    context_after = process.num_ctx_switches()
    time.sleep(max(0.5, 2 * args.telemetry_ms / 1000))
    sampler.stop()
    diagnostics["sensors_after"] = save_diagnostic(args.output_dir, "diagnostico_sensors_depois", ["sensors", "-j"])

    telemetry_frame = __import__("pandas").DataFrame(sampler.records)
    if not telemetry_frame.empty:
        origin = int(telemetry_frame["received_perf_counter_ns"].iloc[0])
        telemetry_frame.insert(1, "seconds_since_first_sample", (telemetry_frame["received_perf_counter_ns"] - origin) / 1e9)
        if rapl.available and "rapl_capture_perf_counter_ns" in telemetry_frame:
            capture_times = telemetry_frame["rapl_capture_perf_counter_ns"].to_numpy(dtype=np.float64)
            elapsed_ns = np.diff(capture_times)
            for key, domain in rapl.domains.items():
                column = f"rapl_{key}_energy_uj"
                energies = telemetry_frame[column].to_numpy(dtype=np.float64)
                delta_uj = np.diff(energies)
                delta_uj[delta_uj < 0] += float(domain["max_energy_range_uj"])
                power = np.full(len(energies), np.nan, dtype=np.float64)
                power[1:] = delta_uj * 1e3 / elapsed_ns
                telemetry_frame[f"rapl_{key}_power_w"] = power
    telemetry_frame.to_csv(args.output_dir / "telemetria_cpu.csv", index=False)
    idle_summary = summarize_telemetry(telemetry_frame, idle_start_ns, idle_end_ns, args.telemetry_ms)
    telemetry_summary = summarize_telemetry(telemetry_frame, benchmark_start_ns, benchmark_end_ns, args.telemetry_ms)
    if rapl.available:
        assert rapl_idle_start is not None and rapl_idle_end is not None
        assert rapl_benchmark_start is not None and rapl_benchmark_end is not None
        rapl_idle = rapl.delta(rapl_idle_start, rapl_idle_end)
        rapl_benchmark = rapl.delta(rapl_benchmark_start, rapl_benchmark_end)
        idle_package_power_w = rapl_idle["domains"]["package"]["average_power_w"]
        package_energy_j = rapl_benchmark["domains"]["package"]["energy_j"]
        duration_rapl_s = rapl_benchmark["duration_s"]
        dynamic_package_power_w = max(
            rapl_benchmark["domains"]["package"]["average_power_w"] - idle_package_power_w, 0.0
        )
        dynamic_energy_j = max(package_energy_j - idle_package_power_w * duration_rapl_s, 0.0)
        rapl_summary = {
            "available": True,
            "source": "Linux powercap intel-rapl (MSR), cumulative energy_uj",
            "primary_domain": "package",
            "idle": rapl_idle,
            "benchmark": rapl_benchmark,
            "idle_package_power_w": idle_package_power_w,
            "dynamic_package_power_w": dynamic_package_power_w,
            "total_energy_per_inference_j": package_energy_j / args.inferences,
            "dynamic_energy_j": dynamic_energy_j,
            "dynamic_energy_per_inference_j": dynamic_energy_j / args.inferences,
            "counter_sampling_interval_ms": args.telemetry_ms,
            "boundary_reads_used_for_energy": True,
        }
    else:
        rapl_summary = {
            "available": False,
            "reason": "domínio package RAPL ausente ou energy_uj sem permissão de leitura",
        }
    total_passage_time_s = float(np.sum(passage_durations_s))
    summary = {
        "inferences": args.inferences,
        "passages": passage,
        "full_passages": args.inferences // args.samples_per_passage,
        "final_passage_samples": args.inferences % args.samples_per_passage,
        "inference_only": common.latency_summary(inference_only_ms, passage_means_inference),
        "end_to_end": common.latency_summary(end_to_end_ms, passage_means_end_to_end),
        "passage_time_total_s": total_passage_time_s,
        "benchmark_telemetry_window_s": (benchmark_end_ns - benchmark_start_ns) / 1e9,
        "throughput_effective_fps": args.inferences / total_passage_time_s,
        "end_to_end_effective_mean_ms": 1000.0 * total_passage_time_s / args.inferences,
        "process_cpu_time_s": (cpu_times_after.user + cpu_times_after.system) - (cpu_times_before.user + cpu_times_before.system),
        "process_cpu_time_per_inference_us": 1e6 * ((cpu_times_after.user + cpu_times_after.system) - (cpu_times_before.user + cpu_times_before.system)) / args.inferences,
        "process_context_switches_voluntary": context_after.voluntary - context_before.voluntary,
        "process_context_switches_involuntary": context_after.involuntary - context_before.involuntary,
        "power_w": (
            rapl_summary["benchmark"]["domains"]["package"]["average_power_w"]
            if rapl_summary["available"] else None
        ),
        "energy_per_inference_j": (
            rapl_summary["total_energy_per_inference_j"] if rapl_summary["available"] else None
        ),
        "power_energy_status": "disponível via RAPL sysfs" if rapl_summary["available"] else "indisponível",
        "rapl": rapl_summary,
        "outliers_removed": False,
    }

    lscpu_text = diagnostics["lscpu"]["output"]
    model_name = next((line.split(":", 1)[1].strip() for line in lscpu_text.splitlines() if line.startswith("Model name:")), platform.processor())
    governors = sorted({
        path.read_text(encoding="utf-8").strip()
        for path in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")
        if path.exists()
    })
    metadata = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "benchmark": {
            "inferences": args.inferences,
            "samples_per_passage": args.samples_per_passage,
            "warmup_inferences": args.warmup,
            "idle_seconds": args.idle_seconds,
            "telemetry_ms": args.telemetry_ms,
            "seed": args.seed,
            "batch_size": 1,
            "synchronous": True,
            "jit_compile": False,
            "mixed_precision": False,
            "outliers_removed": False,
        },
        "cpu": {
            "model_name": model_name,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "governors": governors,
            "tensorflow_intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "tensorflow_inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        },
        "dataset": {
            "train_samples": len(X_train), "test_samples": len(X_test),
            "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
            "train_indices": train_indices, "test_indices": test_indices,
        },
        "model": {"path": str(args.model), "sha256": core.sha256(args.model), "layers": layers, "parameters": model.count_params()},
        "scaler": {
            "path": str(args.scaler), "sha256": core.sha256(args.scaler),
            "matches_reconstructed_split": scaler_matches,
            "load_warnings": [str(item.message) for item in caught],
        },
        "software": {
            "python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "psutil": psutil.__version__, "oneDNN_enabled": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "default/on"),
        },
        "power_energy": {
            "available": rapl.available,
            "source": "Linux powercap intel-rapl energy_uj" if rapl.available else None,
            "primary_scope": "CPU package-0; não inclui fonte/tomada",
            "domains": {
                key: {
                    "name": domain["name"],
                    "energy_path": str(domain["energy_path"]),
                    "max_energy_range_uj": domain["max_energy_range_uj"],
                }
                for key, domain in rapl.domains.items()
            },
            "turbostat_diagnostic_exit_code": diagnostics["turbostat_access"]["exit_code"],
        },
        "diagnostics": diagnostics,
    }

    np.save(args.output_dir / "latencias_inference_only_cpu.npy", inference_only_ms)
    np.save(args.output_dir / "latencias_end_to_end_cpu.npy", end_to_end_ms)
    __import__("pandas").DataFrame(passage_rows).to_csv(args.output_dir / "passagens_cpu.csv", index=False)
    core.write_json(args.output_dir / "metricas_resumo.json", summary)
    core.write_json(args.output_dir / "resumo_telemetria.json", telemetry_summary)
    core.write_json(args.output_dir / "resumo_baseline.json", idle_summary)
    core.write_json(args.output_dir / "resumo_rapl.json", rapl_summary)
    core.write_json(args.output_dir / "validacao_execucao.json", validation)
    core.write_json(args.output_dir / "metadados.json", metadata)
    generate_readme(args.output_dir, summary, telemetry_summary, validation, metadata)

    print(f"resultados: {args.output_dir}")
    print(f"inference-only média: {summary['inference_only']['mean_ms']:.6f} ms")
    print(f"end-to-end efetiva: {summary['end_to_end_effective_mean_ms']:.6f} ms")
    print(f"throughput efetivo: {summary['throughput_effective_fps']:.2f} inferências/s")
    if rapl_summary["available"]:
        print(f"potência média package RAPL: {summary['power_w']:.6f} W")
        print(f"energia package por inferência: {1000 * summary['energy_per_inference_j']:.6f} mJ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
