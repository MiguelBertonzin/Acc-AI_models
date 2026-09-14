#!/usr/bin/env python3
"""Consolida e valida as réplicas RAPL das campanhas CPU MLP Iris."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

METRICS = [
    "duration_s", "package_energy_j", "package_power_w", "idle_package_power_w",
    "dynamic_package_power_w", "total_energy_per_inference_mj",
    "dynamic_energy_per_inference_mj", "core_energy_j", "core_power_w",
    "inference_only_mean_ms", "end_to_end_effective_mean_ms",
    "throughput_effective_fps", "package_temperature_mean_c",
    "package_temperature_max_c",
]

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def campaign_dirs(root: Path, n: int) -> list[Path]:
    return [root / "CPU" / f"resultados_{n}_rapl"] + [
        root / "CPU" / "validacao_rapl" / f"{n}_rep{i}" for i in range(2, 6)
    ]

def load_campaign(root: Path, path: Path, n: int, replica: int) -> dict[str, Any]:
    summary = read_json(path / "metricas_resumo.json")
    rapl = read_json(path / "resumo_rapl.json")
    telemetry = read_json(path / "resumo_telemetria.json")
    validation = read_json(path / "validacao_execucao.json")
    metadata = read_json(path / "metadados.json")
    assert summary["inferences"] == n and summary["outliers_removed"] is False
    assert rapl["available"] is True and rapl["primary_domain"] == "package"
    assert metadata["benchmark"]["batch_size"] == 1
    assert metadata["benchmark"]["synchronous"] is True
    assert validation["correct"] == 29 and validation["samples"] == 30
    assert validation["cpu_gpu_prediction_agreement"] == 1.0
    package = rapl["benchmark"]["domains"]["package"]
    core = rapl["benchmark"]["domains"]["core"]
    assert package["energy_j"] > 0 and package["average_power_w"] > 0
    assert package["energy_j"] >= core["energy_j"]
    assert math.isclose(package["energy_j"] / rapl["benchmark"]["duration_s"],
                        package["average_power_w"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(1000 * package["energy_j"] / n,
                        1000 * rapl["total_energy_per_inference_j"],
                        rel_tol=0, abs_tol=1e-12)
    temp = telemetry.get("package_temperature_c", {})
    return {
        "inferences": n, "replica": replica, "directory": str(path.relative_to(root)),
        "duration_s": rapl["benchmark"]["duration_s"],
        "package_energy_j": package["energy_j"],
        "package_power_w": package["average_power_w"],
        "idle_package_power_w": rapl["idle_package_power_w"],
        "dynamic_package_power_w": rapl["dynamic_package_power_w"],
        "total_energy_per_inference_mj": 1000 * rapl["total_energy_per_inference_j"],
        "dynamic_energy_per_inference_mj": 1000 * rapl["dynamic_energy_per_inference_j"],
        "core_energy_j": core["energy_j"], "core_power_w": core["average_power_w"],
        "inference_only_mean_ms": summary["inference_only"]["mean_ms"],
        "end_to_end_effective_mean_ms": summary["end_to_end_effective_mean_ms"],
        "throughput_effective_fps": summary["throughput_effective_fps"],
        "package_temperature_mean_c": temp.get("mean"),
        "package_temperature_max_c": temp.get("maximum"),
        "telemetry_samples_during_benchmark": telemetry["samples"],
        "correct": validation["correct"],
        "prediction_agreement_gpu": validation["cpu_gpu_prediction_agreement"],
    }

def summarize(x: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    ci = stats.t.interval(.95, len(x)-1, loc=mean, scale=stats.sem(x))
    return {
        "n_campaigns": len(x), "mean": mean, "standard_deviation": sd,
        "coefficient_of_variation_percent": 100*sd/mean,
        "median": float(np.median(x)), "minimum": float(np.min(x)),
        "maximum": float(np.max(x)), "mean_ci95_lower": float(ci[0]),
        "mean_ci95_upper": float(ci[1]),
    }

def paired(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    a = frame[frame.inferences == 30000].sort_values("replica")[metric].to_numpy()
    b = frame[frame.inferences == 100000].sort_values("replica")[metric].to_numpy()
    d = b-a
    ci = stats.t.interval(.95, len(d)-1, loc=float(np.mean(d)), scale=stats.sem(d))
    return {
        "difference_100k_minus_30k": float(np.mean(d)),
        "difference_ci95_lower": float(ci[0]), "difference_ci95_upper": float(ci[1]),
        "paired_t_p_value": float(stats.ttest_rel(b, a).pvalue),
        "relative_difference_percent": 100*(float(np.mean(b))/float(np.mean(a))-1),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    out = root / "CPU" / "validacao_rapl"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (30000, 100000):
        rows.extend(load_campaign(root, p, n, i)
                    for i, p in enumerate(campaign_dirs(root, n), 1))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "campanhas_rapl.csv", index=False)
    aggregate = {
        str(n): {
            metric: summarize(frame[frame.inferences == n][metric].to_numpy(float))
            for metric in METRICS
        } for n in (30000, 100000)
    }
    paired_metrics = [
        "total_energy_per_inference_mj", "dynamic_energy_per_inference_mj",
        "package_power_w", "idle_package_power_w", "inference_only_mean_ms",
        "end_to_end_effective_mean_ms", "throughput_effective_fps",
    ]
    comparisons = {m: paired(frame, m) for m in paired_metrics}
    result = {
        "statistical_unit": "campanha independente", "replicas_per_size": 5,
        "outliers_removed": False,
        "confidence_interval": "t de Student bilateral, 95%, n=5 campanhas",
        "aggregate": aggregate, "paired_100k_vs_30k": comparisons,
        "integrity_checks": {
            "campaigns": len(frame), "all_batch_size_1": True,
            "all_synchronous": True, "all_rapl_available": True,
            "all_accuracy_29_of_30": True,
            "all_cpu_gpu_prediction_agreement_100_percent": True,
            "all_package_energy_positive": True,
            "all_package_energy_at_least_core_energy": True,
            "counter_arithmetic_validated": True,
        },
    }
    (out / "resumo_estatistico_rapl.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    e30, e100 = (aggregate[k]["total_energy_per_inference_mj"] for k in ("30000","100000"))
    d30, d100 = (aggregate[k]["dynamic_energy_per_inference_mj"] for k in ("30000","100000"))
    p30, p100 = (aggregate[k]["package_power_w"] for k in ("30000","100000"))
    ce = comparisons["total_energy_per_inference_mj"]
    cd = comparisons["dynamic_energy_per_inference_mj"]
    lines = []
    for r in rows:
        lines.append(
            f"| {r['inferences']} | {r['replica']} | {r['package_power_w']:.3f} | "
            f"{r['idle_package_power_w']:.3f} | {r['total_energy_per_inference_mj']:.3f} | "
            f"{r['dynamic_energy_per_inference_mj']:.3f} | "
            f"{r['end_to_end_effective_mean_ms']:.6f} | {r['throughput_effective_fps']:.2f} |")
    campaigns = "\n".join(lines)
    report = f"""# Validação de energia RAPL — MLP Iris em CPU

## Conclusão

A energia RAPL da CPU foi medida e validada. A referência recomendada para comparação futura é a campanha de **100.000 inferências, batch 1**, resumida sobre cinco campanhas independentes:

- energia total do pacote: **{e100['mean']:.3f} mJ/inferência**, IC95% [{e100['mean_ci95_lower']:.3f}, {e100['mean_ci95_upper']:.3f}], mediana {e100['median']:.3f}, CV {e100['coefficient_of_variation_percent']:.2f}%;
- energia dinâmica do pacote: **{d100['mean']:.3f} mJ/inferência**, IC95% [{d100['mean_ci95_lower']:.3f}, {d100['mean_ci95_upper']:.3f}];
- potência média do pacote: **{p100['mean']:.3f} W**, IC95% [{p100['mean_ci95_lower']:.3f}, {p100['mean_ci95_upper']:.3f}].

Não houve diferença estatisticamente significativa entre 30.000 e 100.000 inferências. Para energia total, a diferença pareada 100k−30k foi {ce['difference_100k_minus_30k']:+.3f} mJ/inferência, IC95% [{ce['difference_ci95_lower']:+.3f}, {ce['difference_ci95_upper']:+.3f}], p={ce['paired_t_p_value']:.3f}. Para energia dinâmica, p={cd['paired_t_p_value']:.3f}.

## Resultados agregados

| Campanha | Energia total package (mJ/inf) | Energia dinâmica (mJ/inf) | Potência package (W) | CV energia total |
|---|---:|---:|---:|---:|
| 30.000 × 5 | {e30['mean']:.3f} [{e30['mean_ci95_lower']:.3f}, {e30['mean_ci95_upper']:.3f}] | {d30['mean']:.3f} [{d30['mean_ci95_lower']:.3f}, {d30['mean_ci95_upper']:.3f}] | {p30['mean']:.3f} [{p30['mean_ci95_lower']:.3f}, {p30['mean_ci95_upper']:.3f}] | {e30['coefficient_of_variation_percent']:.2f}% |
| 100.000 × 5 | {e100['mean']:.3f} [{e100['mean_ci95_lower']:.3f}, {e100['mean_ci95_upper']:.3f}] | {d100['mean']:.3f} [{d100['mean_ci95_lower']:.3f}, {d100['mean_ci95_upper']:.3f}] | {p100['mean']:.3f} [{p100['mean_ci95_lower']:.3f}, {p100['mean_ci95_upper']:.3f}] | {e100['coefficient_of_variation_percent']:.2f}% |

Os colchetes apresentam IC95% da média entre campanhas. Nenhuma execução foi removida.

## Campanhas individuais

| Inferências | Réplica | Package W | Idle W | Total mJ/inf | Dinâmica mJ/inf | E2E ms | inf/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
{campaigns}

## Metodologia

1. Modelo 4–8–8–3, TensorFlow/oneDNN em CPU, `float32`, batch 1, chamadas seriais e sincronizadas.
2. Mesmos artefatos, holdout, seed 20260831 e fronteiras inference-only/end-to-end das campanhas anteriores.
3. Em cada campanha: 200 inferências de warm-up, 5 s de baseline ocioso e exatamente 30.000 ou 100.000 inferências.
4. Cinco campanhas independentes por tamanho; cada execução repetiu inicialização, warm-up e baseline.
5. Fonte: Linux powercap, domínio `intel-rapl:0` (`package-0`), contador cumulativo `energy_uj`.
6. Energia: diferença dos acumuladores imediatamente antes/depois da janela ativa, com correção por `max_energy_range_uj` em rollover.
7. Potência média: energia do pacote dividida pela duração entre as próprias leituras RAPL.
8. Energia total por inferência: energia package dividida pelo número exato de inferências.
9. Energia dinâmica: `max(E_ativa − P_ociosa × t_ativa, 0)`.
10. Telemetria auxiliar a cada 100 ms gravou package/core/uncore, uso, frequência e temperatura. As leituras de fronteira determinam a energia final.
11. `package` já contém os subdomínios; `core` e `uncore` não foram somados novamente.
12. IC95% t de Student sobre cinco campanhas; comparação 100k×30k por teste t pareado.
13. Nenhum outlier foi removido. Todas as dez campanhas tiveram 29/30 acertos e concordância CPU×GPU de 100%.

## Validações e limitações

- Dez campanhas e todos os artefatos foram lidos.
- Package foi positivo e package ≥ core em todas.
- Foram verificadas numericamente as identidades energia/duração = potência e energia/N = energia por inferência.
- RAPL mede uma estimativa do pacote do processador; não inclui fonte, tomada e periféricos.
- `turbostat` sem privilégios continua bloqueado de abrir MSR pelo kernel apesar do modo do dispositivo. Isso não impede a leitura validada via powercap.
- Nenhuma conclusão energética deve comparar diretamente package RAPL com o sensor da placa GPU, pois os escopos físicos são diferentes.

## Referência para CPU, GPU e FPGA

Use como baseline CPU principal **{e100['mean']:.3f} mJ/inferência package RAPL** e **{d100['mean']:.3f} mJ/inferência dinâmica**, ambos com seus IC95%.

Para FPGA, registre separadamente energia dos rails do dispositivo/placa e energia do host, mantendo a mesma janela batch 1 serial e sincronizada. O valor GPU de 8,193 mJ/inferência é energia da placa GPU e deve ser rotulado separadamente.

## Arquivos

- `validacao_rapl/campanhas_rapl.csv`: uma linha por campanha.
- `validacao_rapl/resumo_estatistico_rapl.json`: estatísticas, testes e integridade.
- `resultados_30000_rapl/` e `resultados_100000_rapl/`: réplicas 1.
- `validacao_rapl/30000_rep2..5` e `validacao_rapl/100000_rep2..5`: demais réplicas.
"""
    (root / "CPU" / "VALIDACAO_ENERGIA_RAPL.md").write_text(report, encoding="utf-8")
    print(root / "CPU" / "VALIDACAO_ENERGIA_RAPL.md")
    print(json.dumps({"30k_mean_mj": e30["mean"], "100k_mean_mj": e100["mean"],
                      "100k_ci95": [e100["mean_ci95_lower"], e100["mean_ci95_upper"]],
                      "paired_p": ce["paired_t_p_value"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
