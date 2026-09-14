#!/usr/bin/env python3
"""Benchmark exato de 100 mil inferências da MLP Iris e comparação com 30 mil."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import benchmark_mlp_iris_gpu as core


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "iris_mlp_clean.h5")
    parser.add_argument("--scaler", type=Path, default=root / "iris_scaler.joblib")
    parser.add_argument("--reference-dir", type=Path, default=root / "GPU" / "resultados")
    parser.add_argument("--output-dir", type=Path, default=root / "GPU" / "resultados_100000")
    parser.add_argument("--comparison-readme", type=Path, default=root / "GPU" / "COMPARACAO_30000_100000.md")
    parser.add_argument("--inferences", type=int, default=100_000)
    parser.add_argument("--samples-per-passage", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--telemetry-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.inferences < 60:
        parser.error("use pelo menos 60 inferências")
    if args.samples_per_passage < 2 or args.samples_per_passage > 30:
        parser.error("samples-per-passage deve estar entre 2 e 30")
    if args.warmup < 1 or args.idle_seconds < 1 or args.telemetry_ms < 20:
        parser.error("use warmup >= 1, idle-seconds >= 1 e telemetry-ms >= 20")
    return args


def latency_summary(values: Any, passage_means: Any) -> dict[str, float]:
    import numpy as np
    from scipy.stats import t

    means = np.asarray(passage_means, dtype=np.float64)
    if len(means) < 2:
        raise RuntimeError("são necessárias ao menos duas passagens completas para o IC95%")
    standard_error = float(np.std(means, ddof=1) / math.sqrt(len(means)))
    critical = float(t.ppf(0.975, len(means) - 1))
    mean_passages = float(np.mean(means))
    return {
        "mean_ms": float(np.mean(values)),
        "mean_ci95_lower_ms": mean_passages - critical * standard_error,
        "mean_ci95_upper_ms": mean_passages + critical * standard_error,
        "median_ms": float(np.median(values)),
        "std_sample_ms": float(np.std(values, ddof=1)),
        "cv_percent": float(100 * np.std(values, ddof=1) / np.mean(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "ci_unit": "médias das passagens completas",
        "ci_passages": int(len(means)),
    }


def percent_delta(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def welch_passage_comparison(reference_values: Any, new_values: Any) -> dict[str, Any]:
    """Compara médias de passagens sem assumir variâncias iguais."""
    import numpy as np
    from scipy.stats import t, ttest_ind

    old = np.asarray(reference_values, dtype=np.float64)
    new = np.asarray(new_values, dtype=np.float64)
    old_variance = float(np.var(old, ddof=1))
    new_variance = float(np.var(new, ddof=1))
    difference = float(np.mean(new) - np.mean(old))
    old_term = old_variance / len(old)
    new_term = new_variance / len(new)
    standard_error = math.sqrt(old_term + new_term)
    degrees_freedom = (old_term + new_term) ** 2 / (old_term**2 / (len(old) - 1) + new_term**2 / (len(new) - 1))
    critical = float(t.ppf(0.975, degrees_freedom))
    test = ttest_ind(new, old, equal_var=False)
    standardized = difference / math.sqrt((old_variance + new_variance) / 2)
    return {
        "unit": "média de uma passagem completa de 30 inferências",
        "reference_passages": len(old),
        "new_passages": len(new),
        "reference_mean_ms": float(np.mean(old)),
        "new_mean_ms": float(np.mean(new)),
        "difference_new_minus_reference_ms": difference,
        "difference_ci95_lower_ms": difference - critical * standard_error,
        "difference_ci95_upper_ms": difference + critical * standard_error,
        "welch_t": float(test.statistic),
        "welch_degrees_freedom": degrees_freedom,
        "welch_p_value_two_sided": float(test.pvalue),
        "standardized_mean_difference": standardized,
    }


def load_reference(reference_dir: Path) -> dict[str, Any]:
    import pandas as pd

    performance = pd.read_csv(reference_dir / "metricas_desempenho.csv").iloc[-1].to_dict()
    telemetry = json.loads((reference_dir / "resumo_telemetria.json").read_text(encoding="utf-8"))
    stats = json.loads((reference_dir / "validacao_estatistica.json").read_text(encoding="utf-8"))
    metadata = json.loads((reference_dir / "metadados.json").read_text(encoding="utf-8"))
    return {"performance": performance, "telemetry": telemetry, "statistics": stats, "metadata": metadata}


def build_comparison(reference: dict[str, Any], new_summary: dict[str, Any], new_telemetry: dict[str, Any]) -> dict[str, Any]:
    old = reference["performance"]
    old_tel = reference["telemetry"]
    new_inference = new_summary["inference_only"]
    new_end = new_summary["end_to_end"]
    old_end_mean = 1000.0 / float(old["throughput_effective_fps"])
    new_end_mean = float(new_summary["end_to_end_effective_mean_ms"])
    old_energy_mj = 1000.0 * float(old["energy_total_per_inference_j"])
    new_energy_mj = 1000.0 * float(new_summary["energy_total_per_inference_j"])
    rows = {
        "inference_only_mean_ms": (float(old["latency_mean_ms"]), new_inference["mean_ms"]),
        "inference_only_median_ms": (float(old["latency_median_ms"]), new_inference["median_ms"]),
        "inference_only_std_ms": (float(old["latency_std_sample_ms"]), new_inference["std_sample_ms"]),
        "inference_only_p95_ms": (float(old["latency_p95_ms"]), new_inference["p95_ms"]),
        "inference_only_p99_ms": (float(old["latency_p99_ms"]), new_inference["p99_ms"]),
        "end_to_end_effective_mean_ms": (old_end_mean, new_end_mean),
        "end_to_end_effective_throughput_fps": (float(old["throughput_effective_fps"]), float(new_summary["throughput_effective_fps"])),
        "power_board_mean_w": (float(old_tel["power.draw.instant"]["mean"]), float(new_telemetry["power.draw.instant"]["mean"])),
        "energy_total_per_inference_mj": (old_energy_mj, new_energy_mj),
        "gpu_utilization_mean_percent": (float(old_tel["utilization.gpu"]["mean"]), float(new_telemetry["utilization.gpu"]["mean"])),
        "temperature_mean_c": (float(old_tel["temperature.gpu"]["mean"]), float(new_telemetry["temperature.gpu"]["mean"])),
    }
    return {
        "reference_inferences": int(old["inferences"]),
        "new_inferences": int(new_summary["inferences"]),
        "metrics": {
            name: {"reference_30000": before, "new_100000": after, "delta_percent": percent_delta(after, before)}
            for name, (before, after) in rows.items()
        },
    }


def generate_readme(
    output_dir: Path,
    comparison_path: Path,
    summary: dict[str, Any],
    comparison: dict[str, Any],
    statistical_comparison: dict[str, Any],
    telemetry: dict[str, Any],
    validation: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    inf = summary["inference_only"]
    e2e = summary["end_to_end"]
    metrics = comparison["metrics"]
    power = telemetry["power.draw.instant"]
    gpu_util = telemetry["utilization.gpu"]
    memory = telemetry["memory.used"]
    clock = telemetry["clocks.current.sm"]
    temperature = telemetry["temperature.gpu"]
    fan = telemetry["fan.speed"]
    pstate = ", ".join(f"{name}: {count}" for name, count in telemetry.get("pstate_counts", {}).items())
    statistical_inference = statistical_comparison["inference_only"]
    statistical_end = statistical_comparison["end_to_end_effective"]

    def row(label: str, key: str, unit: str, decimals: int = 6) -> str:
        metric = metrics[key]
        return (
            f"| {label} | {metric['reference_30000']:.{decimals}f} {unit} | "
            f"{metric['new_100000']:.{decimals}f} {unit} | {metric['delta_percent']:+.2f}% |"
        )

    report = f"""# Comparação da MLP Iris na GPU — 30.000 × 100.000 inferências

## Resultado principal

A campanha nova executou exatamente **{summary['inferences']:,} inferências** na NVIDIA GeForce RTX 3050 OEM. Foram {summary['full_passages']} passagens completas de {metadata['benchmark']['samples_per_passage']} amostras e uma passagem final de {summary['final_passage_samples']} amostras. Nenhum outlier foi removido.

| Métrica | 30.000 inferências | 100.000 inferências | Variação |
|---|---:|---:|---:|
{row('Inference-only — média', 'inference_only_mean_ms', 'ms')}
{row('Inference-only — mediana', 'inference_only_median_ms', 'ms')}
{row('Inference-only — desvio-padrão', 'inference_only_std_ms', 'ms')}
{row('Inference-only — p95', 'inference_only_p95_ms', 'ms')}
{row('Inference-only — p99', 'inference_only_p99_ms', 'ms')}
{row('End-to-end — média efetiva', 'end_to_end_effective_mean_ms', 'ms')}
{row('End-to-end — vazão efetiva', 'end_to_end_effective_throughput_fps', 'inf/s', 2)}
{row('Potência média da placa', 'power_board_mean_w', 'W', 3)}
{row('Energia total por inferência', 'energy_total_per_inference_mj', 'mJ')}
{row('Utilização média da GPU', 'gpu_utilization_mean_percent', '%', 2)}
{row('Temperatura média', 'temperature_mean_c', '°C', 2)}

Variação positiva significa que o valor da campanha de 100.000 foi maior. Para latência e energia, menor é melhor; para vazão, maior é melhor.

## Comparação estatística entre campanhas

O teste t de Welch foi aplicado às médias das passagens completas de 30 inferências, sem assumir variâncias iguais.

| Janela | Diferença 100k − 30k | IC95% da diferença | p bilateral | Diferença padronizada |
|---|---:|---:|---:|---:|
| Inference-only | {1000 * statistical_inference["difference_new_minus_reference_ms"]:.3f} µs | [{1000 * statistical_inference["difference_ci95_lower_ms"]:.3f}, {1000 * statistical_inference["difference_ci95_upper_ms"]:.3f}] µs | {statistical_inference["welch_p_value_two_sided"]:.3e} | {statistical_inference["standardized_mean_difference"]:.3f} |
| End-to-end efetiva | {1000 * statistical_end["difference_new_minus_reference_ms"]:.3f} µs | [{1000 * statistical_end["difference_ci95_lower_ms"]:.3f}, {1000 * statistical_end["difference_ci95_upper_ms"]:.3f}] µs | {statistical_end["welch_p_value_two_sided"]:.3e} | {statistical_end["standardized_mean_difference"]:.3f} |

Na janela inference-only, um IC que inclua zero indica que a diferença não é distinguível da variação entre passagens. Como as campanhas ocorreram em sessões sequenciais e têm autocorrelação temporal, os testes são exploratórios e não demonstram causalidade.

## Campanha de 100.000 — inference-only

`Inference-only` começa depois de `tf.convert_to_tensor` e termina depois de `output.numpy()`. Inclui a execução do grafo TensorFlow na GPU, sincronização e materialização das três probabilidades no host. Como a cópia da entrada pode ser assíncrona, esta não é uma medição pura de kernel CUDA.

| Métrica | Resultado |
|---|---:|
| Média | {inf['mean_ms']:.6f} ms |
| IC95% da média | [{inf['mean_ci95_lower_ms']:.6f}, {inf['mean_ci95_upper_ms']:.6f}] ms |
| Mediana | {inf['median_ms']:.6f} ms |
| Desvio-padrão | {inf['std_sample_ms']:.6f} ms |
| CV | {inf['cv_percent']:.2f}% |
| p90 / p95 / p99 | {inf['p90_ms']:.6f} / {inf['p95_ms']:.6f} / {inf['p99_ms']:.6f} ms |
| Mínimo / máximo | {inf['minimum_ms']:.6f} / {inf['maximum_ms']:.6f} ms |

## Campanha de 100.000 — end-to-end

A latência individual `end-to-end` começa antes da criação do tensor e termina depois da sincronização, transferência/materialização da saída e `argmax`. A vazão efetiva usa a soma dos tempos das passagens e também inclui laço Python e armazenamento dos resultados.

| Métrica | Resultado |
|---|---:|
| Média individual | {e2e['mean_ms']:.6f} ms |
| IC95% da média | [{e2e['mean_ci95_lower_ms']:.6f}, {e2e['mean_ci95_upper_ms']:.6f}] ms |
| Mediana | {e2e['median_ms']:.6f} ms |
| Desvio-padrão | {e2e['std_sample_ms']:.6f} ms |
| p90 / p95 / p99 | {e2e['p90_ms']:.6f} / {e2e['p95_ms']:.6f} / {e2e['p99_ms']:.6f} ms |
| Mínimo / máximo | {e2e['minimum_ms']:.6f} / {e2e['maximum_ms']:.6f} ms |
| Média efetiva pelas passagens | {summary['end_to_end_effective_mean_ms']:.6f} ms |
| Vazão efetiva | {summary['throughput_effective_fps']:.2f} inferências/s |

A média individual exclui somente o custo de registrar a própria duração. A média efetiva pelas passagens é mais abrangente e, por isso, é a recomendada para comparação end-to-end.

## Telemetria NVIDIA

O `nvidia-smi` foi executado como processo persistente, com intervalo nominal de {metadata['benchmark']['telemetry_ms']} ms. A potência foi integrada no tempo com regra trapezoidal. A potência de baseline foi medida durante {metadata['benchmark']['idle_seconds']:.1f} s depois do aquecimento e com o contexto TensorFlow ainda carregado.

| Métrica | Resultado |
|---|---:|
| Amostras durante benchmark | {telemetry['samples']} |
| Janela energética | {telemetry['duration_s']:.3f} s |
| Potência média / p95 / máxima | {power['mean']:.3f} / {power['p95']:.3f} / {power['maximum']:.3f} W |
| Potência ociosa | {telemetry['power_baseline_w']:.3f} W |
| Energia total | {telemetry['energy_total_j']:.3f} J |
| Energia total por inferência | {1000 * summary['energy_total_per_inference_j']:.6f} mJ |
| Energia dinâmica por inferência | {1000 * summary['energy_dynamic_per_inference_j']:.6f} mJ |
| Utilização GPU média / p95 / máxima | {gpu_util['mean']:.2f}% / {gpu_util['p95']:.2f}% / {gpu_util['maximum']:.2f}% |
| Memória usada média / máxima | {memory['mean']:.1f} / {memory['maximum']:.1f} MiB |
| Clock SM médio / máximo | {clock['mean']:.1f} / {clock['maximum']:.1f} MHz |
| Temperatura média / máxima | {temperature['mean']:.1f} / {temperature['maximum']:.1f} °C |
| Ventoinha média / máxima | {fan['mean']:.1f}% / {fan['maximum']:.1f}% |
| P-state | {pstate} |

A energia é referente ao sensor da placa GPU e não inclui CPU, RAM, fonte ou perdas na tomada. A energia dinâmica é a diferença entre duas leituras próximas e deve ser interpretada com cautela frente à incerteza do sensor e à granularidade de 100 ms.

## Validação funcional e estatística

- Arquitetura conferida: 4–8–8–3, ReLU/ReLU/softmax, 139 parâmetros, `float32`.
- Divisão reconstruída: 120 treino / 30 teste, `test_size=0.2`, `random_state=42`, estratificada.
- Holdout balanceado: 10 amostras por classe.
- Acurácia: {validation['correct']}/{validation['samples']} = {100 * validation['accuracy']:.2f}%.
- Predições iguais às da campanha de 30.000: {validation['matches_reference_predictions']}.
- Concordância NumPy/CPU × TensorFlow/GPU: {100 * validation['cpu_gpu_prediction_agreement']:.2f}%.
- Diferença máxima de probabilidade NumPy/CPU × TensorFlow/GPU: {validation['cpu_gpu_max_abs_probability_difference']:.3e}.
- Posicionamento confirmado: `{validation['gpu_output_device']}`; soft placement desabilitado.
- Todas as {summary['inferences']:,} classes repetidas coincidiram com a predição funcional inicial.

As 100.000 repetições caracterizam desempenho e determinismo, mas não aumentam o tamanho amostral da acurácia: a unidade estatística continua sendo o holdout com 30 flores únicas. O resultado estatístico completo permanece em `GPU/resultados/validacao_estatistica.json`.

## Metodologia completa

1. Expôs somente a GPU de índice {metadata['gpu']['index']} com `CUDA_VISIBLE_DEVICES`.
2. Desabilitou soft placement e fixou modelo e grafo em `/GPU:0`; uma saída fora desse dispositivo abortaria o teste.
3. Habilitou determinismo do TensorFlow e manteve `jit_compile=False`, batch 1, `float32`, sem XLA, TensorRT, mixed precision ou quantização.
4. Reconstruiu o holdout original a partir do Iris e confirmou que média e escala coincidem com o `StandardScaler` salvo.
5. Executou {metadata['benchmark']['warmup_inferences']} inferências de aquecimento, fora das estatísticas.
6. Confirmou probabilidades, classes, paridade NumPy/CPU e igualdade com a campanha de referência.
7. Iniciou telemetria persistente e mediu {metadata['benchmark']['idle_seconds']:.1f} s de baseline ocioso.
8. Processou uma amostra por vez, em ordem permutada deterministicamente pela seed {metadata['benchmark']['seed']}.
9. Materializou cada saída com `.numpy()`, impedindo medir apenas o enfileiramento assíncrono.
10. Executou exatamente {summary['inferences']:,} inferências: {summary['full_passages']} × {metadata['benchmark']['samples_per_passage']} + {summary['final_passage_samples']}.
11. Não removeu outliers. Calculou média, mediana, desvio-padrão amostral, CV, p90, p95, p99, mínimo e máximo.
12. Calculou IC95% com distribuição t sobre as médias das {inf['ci_passages']} passagens completas; o ciclo parcial não entrou no IC, mas todas as amostras entraram nas demais estatísticas.
13. Calculou throughput como inferências totais divididas pela soma dos tempos end-to-end das passagens.
14. Integrou `power.draw.instant` na janela completa e dividiu a energia pelo número exato de inferências.
15. Preservou CSVs, arrays NumPy, snapshots `nvidia-smi -q`, hashes, versões e índices do dataset.

## Ambiente

- GPU: {metadata['gpu']['name']}, compute capability {metadata['gpu']['compute_capability']}, {metadata['gpu']['memory_total_mib']} MiB.
- Driver: {metadata['gpu']['driver_version']}; limite de potência: {metadata['gpu']['power_limit_w']} W.
- TensorFlow {metadata['software']['tensorflow']}, Keras {metadata['software']['keras']}.
- TensorFlow compilado com CUDA {metadata['software']['tensorflow_cuda']} e cuDNN {metadata['software']['cudnn']}.
- Python {metadata['software']['python_short']}; NumPy {metadata['software']['numpy']}; scikit-learn {metadata['software']['scikit_learn']}.
- Modelo SHA-256: `{metadata['model']['sha256']}`.
- Scaler SHA-256: `{metadata['scaler']['sha256']}`.

O “CUDA Version” mostrado no cabeçalho do `nvidia-smi` representa a versão máxima suportada pelo driver; o runtime efetivamente usado pelo TensorFlow é o registrado acima.

## Arquivos produzidos

- `latencias_inference_only_gpu.npy`: 100.000 tempos individuais.
- `latencias_end_to_end_gpu.npy`: 100.000 tempos individuais.
- `passagens_gpu.csv`: uma linha por passagem.
- `metricas_resumo.json`: estatísticas agregadas.
- `comparacao_30k_100k.json`: valores e variações da comparação.
- `comparacao_estatistica_passagens.json`: teste de Welch e IC95% da diferença.
- `telemetria_nvidia_smi.csv` e `resumo_telemetria.json`: telemetria bruta e resumo.
- `validacao_execucao.json`: qualidade, paridade e posicionamento.
- `metadados.json`: ambiente, configuração, hashes e divisão.
- `nvidia_smi_*.txt`: snapshots antes e depois.

## Limitações de comparação

- As campanhas foram sequenciais, não alternadas nem pareadas por reinicialização; diferenças pequenas podem refletir temperatura, DVFS, desktop, driver e escalonamento.
- A campanha antiga não registrou a distribuição individual end-to-end. Para ela, a média end-to-end foi reconstruída como `1000 / throughput_efetivo`; percentis end-to-end não podem ser recuperados.
- A RTX 3050 também atendia ao ambiente gráfico. A memória e potência do `nvidia-smi` representam a placa, não somente o processo do benchmark.
- Este é throughput sustentado batch 1 e síncrono, não throughput máximo com batching ou requisições concorrentes.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    comparison_path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.scaler = args.scaler.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.comparison_readme = args.comparison_readme.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"diretório não vazio: {args.output_dir}; use --overwrite ou outro caminho")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.comparison_readme.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import joblib
    import keras
    import numpy as np
    import pandas as pd
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
    fitted = StandardScaler().fit(X_train.astype(np.float32))
    scaler_matches = bool(
        scaler.n_samples_seen_ == 120
        and np.allclose(scaler.mean_, fitted.mean_, rtol=0, atol=2e-7)
        and np.allclose(scaler.scale_, fitted.scale_, rtol=0, atol=2e-7)
    )
    if not scaler_matches:
        raise RuntimeError("o scaler não coincide com a divisão reconstruída")
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    with tf.device("/GPU:0"):
        model = tf.keras.models.load_model(args.model, compile=False)
    layers = [(layer.units, layer.activation.__name__) for layer in model.layers]
    if model.input_shape != (None, 4) or layers != [(8, "relu"), (8, "relu"), (3, "softmax")] or model.count_params() != 139:
        raise RuntimeError("arquitetura do modelo diferente de 4-8-8-3/139 parâmetros")

    @tf.function(input_signature=[tf.TensorSpec((1, 4), tf.float32)], autograph=False, jit_compile=False)
    def gpu_infer(batch: Any) -> Any:
        with tf.device("/GPU:0"):
            return model(batch, training=False)

    for index in range(args.warmup):
        row = X_test_scaled[index % len(X_test_scaled)]
        warmed = gpu_infer(tf.convert_to_tensor(row[np.newaxis, :], dtype=tf.float32))
        _ = warmed.numpy()
    if "GPU:0" not in warmed.device.upper():
        raise RuntimeError(f"saída do aquecimento em dispositivo inesperado: {warmed.device}")

    gpu_probabilities, gpu_device = core.infer_all(gpu_infer, X_test_scaled, "GPU:0")
    weights = [np.asarray(value, dtype=np.float32) for value in model.get_weights()]
    hidden1 = np.maximum(np.matmul(X_test_scaled, weights[0]) + weights[1], 0.0)
    hidden2 = np.maximum(np.matmul(hidden1, weights[2]) + weights[3], 0.0)
    logits = np.matmul(hidden2, weights[4]) + weights[5]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    cpu_probabilities = (exponentials / np.sum(exponentials, axis=1, keepdims=True)).astype(np.float64)
    gpu_predictions = np.argmax(gpu_probabilities, axis=1)
    cpu_predictions = np.argmax(cpu_probabilities, axis=1)

    reference = load_reference(args.reference_dir)
    reference_predictions = pd.read_csv(args.reference_dir / "predicoes_holdout.csv")["predicted_id"].to_numpy(dtype=np.int64)
    validation = {
        "samples": len(y_test),
        "correct": int(np.sum(gpu_predictions == y_test)),
        "accuracy": float(np.mean(gpu_predictions == y_test)),
        "matches_reference_predictions": bool(np.array_equal(gpu_predictions, reference_predictions)),
        "cpu_gpu_prediction_agreement": float(np.mean(cpu_predictions == gpu_predictions)),
        "cpu_gpu_max_abs_probability_difference": float(np.max(np.abs(cpu_probabilities - gpu_probabilities))),
        "probabilities_finite": bool(np.isfinite(gpu_probabilities).all()),
        "probability_sum_max_abs_error": float(np.max(np.abs(np.sum(gpu_probabilities, axis=1) - 1.0))),
        "gpu_output_device": gpu_device,
    }
    if not validation["matches_reference_predictions"] or validation["cpu_gpu_prediction_agreement"] != 1.0:
        raise RuntimeError("a validação funcional divergiu da referência")

    static_fields = ["index", "name", "uuid", "pci_bus_id", "driver_version", "pstate_before", "memory_total_mib", "power_limit_w", "clock_max_sm_mhz", "clock_max_memory_mhz", "compute_capability"]
    static_command = [
        "nvidia-smi", f"--id={args.gpu_index}",
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,memory.total,power.limit,clocks.max.sm,clocks.max.memory,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    static_code, static_output = core.run_capture(static_command)
    if static_code != 0:
        raise RuntimeError(f"consulta nvidia-smi falhou: {static_output}")
    static_values = next(csv.reader([static_output.strip()], skipinitialspace=True))
    gpu_static = dict(zip(static_fields, [value.strip() for value in static_values]))

    core.write_snapshot(args.output_dir, "antes")
    sampler = core.NvidiaSmiSampler(args.gpu_index, args.telemetry_ms)
    sampler.start()
    idle_start_ns = time.perf_counter_ns()
    time.sleep(args.idle_seconds)
    idle_end_ns = time.perf_counter_ns()

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
            host_output = gpu_infer(batch).numpy()[0]
            inference_end_ns = time.perf_counter_ns()
            observed[position] = int(np.argmax(host_output))
            end_to_end_end_ns = time.perf_counter_ns()
            inference_only_ms[global_index] = (inference_end_ns - inference_start_ns) / 1e6
            end_to_end_ms[global_index] = (end_to_end_end_ns - end_to_end_start_ns) / 1e6
        passage_end_ns = time.perf_counter_ns()
        if not np.array_equal(observed, gpu_predictions[order]):
            raise RuntimeError(f"não determinismo detectado na passagem {passage + 1}")
        passage_duration = (passage_end_ns - passage_start_ns) / 1e9
        current_inference = inference_only_ms[completed : completed + count]
        current_end = end_to_end_ms[completed : completed + count]
        passage_durations_s.append(passage_duration)
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
            "passage_duration_s": passage_duration,
            "throughput_effective_fps": count / passage_duration,
        })
        completed += count
        passage += 1
        if passage % progress_every == 0 or completed == args.inferences:
            print(f"progresso: {completed}/{args.inferences} inferências", flush=True)
    benchmark_end_ns = time.perf_counter_ns()
    time.sleep(max(0.5, 2 * args.telemetry_ms / 1000))
    sampler.stop()
    core.write_snapshot(args.output_dir, "depois")

    telemetry_frame = core.telemetry_frame(sampler.records)
    telemetry_frame.to_csv(args.output_dir / "telemetria_nvidia_smi.csv", index=False)
    idle_frame = core.interval_frame(telemetry_frame, idle_start_ns, idle_end_ns)
    if len(idle_frame) < 2:
        raise RuntimeError("telemetria ociosa insuficiente")
    baseline_power_w = float(idle_frame["power.draw.instant"].dropna().mean())
    telemetry_summary = core.summarize_telemetry(
        telemetry_frame, benchmark_start_ns, benchmark_end_ns, baseline_power_w
    )
    telemetry_summary["sampling_interval_nominal_ms"] = args.telemetry_ms
    energy_total_per_inference_j = telemetry_summary["energy_total_j"] / args.inferences
    energy_dynamic_per_inference_j = telemetry_summary["energy_dynamic_j"] / args.inferences
    total_passage_time_s = float(np.sum(passage_durations_s))

    summary = {
        "inferences": args.inferences,
        "passages": passage,
        "full_passages": args.inferences // args.samples_per_passage,
        "final_passage_samples": args.inferences % args.samples_per_passage,
        "inference_only": latency_summary(inference_only_ms, passage_means_inference),
        "end_to_end": latency_summary(end_to_end_ms, passage_means_end_to_end),
        "passage_time_total_s": total_passage_time_s,
        "benchmark_power_window_s": (benchmark_end_ns - benchmark_start_ns) / 1e9,
        "throughput_effective_fps": args.inferences / total_passage_time_s,
        "end_to_end_effective_mean_ms": 1000.0 * total_passage_time_s / args.inferences,
        "energy_total_per_inference_j": energy_total_per_inference_j,
        "energy_dynamic_per_inference_j": energy_dynamic_per_inference_j,
        "outliers_removed": False,
    }
    comparison = build_comparison(reference, summary, telemetry_summary)
    reference_latency_passages = np.load(args.reference_dir / "latencias_gpu.npy").mean(axis=1)
    reference_passages = pd.read_csv(args.reference_dir / "passagens_gpu.csv")
    reference_end_passages = (
        reference_passages["cycle_duration_s"].to_numpy(dtype=np.float64)
        / reference_passages["inferences"].to_numpy(dtype=np.float64)
        * 1000.0
    )
    full_passages = args.inferences // args.samples_per_passage
    new_end_passages = (
        np.asarray(passage_durations_s[:full_passages], dtype=np.float64)
        / args.samples_per_passage
        * 1000.0
    )
    statistical_comparison = {
        "inference_only": welch_passage_comparison(reference_latency_passages, passage_means_inference),
        "end_to_end_effective": welch_passage_comparison(reference_end_passages, new_end_passages),
        "interpretation": "exploratória; campanhas sequenciais e temporalmente autocorrelacionadas",
    }
    tf_build = tf.sysconfig.get_build_info()
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
        "dataset": {
            "name": "sklearn Iris",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
            "train_indices": train_indices,
            "test_indices": test_indices,
            "class_map": {index: name for index, name in enumerate(core.CLASS_NAMES)},
        },
        "model": {"path": str(args.model), "sha256": core.sha256(args.model), "layers": layers, "parameters": model.count_params()},
        "scaler": {
            "path": str(args.scaler), "sha256": core.sha256(args.scaler), "matches_reconstructed_split": scaler_matches,
            "load_warnings": [str(item.message) for item in caught],
        },
        "gpu": gpu_static,
        "software": {
            "python": sys.version,
            "python_short": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow_cuda": tf_build.get("cuda_version"),
            "cudnn": tf_build.get("cudnn_version"),
        },
        "reference_dir": str(args.reference_dir),
    }

    np.save(args.output_dir / "latencias_inference_only_gpu.npy", inference_only_ms)
    np.save(args.output_dir / "latencias_end_to_end_gpu.npy", end_to_end_ms)
    pd.DataFrame(passage_rows).to_csv(args.output_dir / "passagens_gpu.csv", index=False)
    core.write_json(args.output_dir / "metricas_resumo.json", summary)
    core.write_json(args.output_dir / "comparacao_30k_100k.json", comparison)
    core.write_json(args.output_dir / "comparacao_estatistica_passagens.json", statistical_comparison)
    core.write_json(args.output_dir / "resumo_telemetria.json", telemetry_summary)
    core.write_json(args.output_dir / "validacao_execucao.json", validation)
    core.write_json(args.output_dir / "metadados.json", metadata)
    if sampler.stderr.strip():
        (args.output_dir / "telemetria_nvidia_smi_stderr.txt").write_text(sampler.stderr, encoding="utf-8")
    generate_readme(
        args.output_dir, args.comparison_readme, summary, comparison,
        statistical_comparison, telemetry_summary, validation, metadata
    )

    print(f"resultados: {args.output_dir}")
    print(f"comparação: {args.comparison_readme}")
    print(f"inference-only média: {summary['inference_only']['mean_ms']:.6f} ms")
    print(f"end-to-end efetiva: {summary['end_to_end_effective_mean_ms']:.6f} ms")
    print(f"throughput efetivo: {summary['throughput_effective_fps']:.2f} inferências/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
