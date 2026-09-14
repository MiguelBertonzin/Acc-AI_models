#!/usr/bin/env python3
"""Gera READMEs comparativos e sincronizados para os benchmarks CPU e GPU."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SIZES = (100, 1000, 10000)
PREFIXES = (10, 20, 50, 100)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_results(device: str) -> tuple[list[dict[str, str]], dict[tuple[int, int], dict[str, str]]]:
    path = ROOT / device / "resultados" / "resultados.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(int(row["unique_images"]), int(row["repetitions"])): row for row in rows}
    expected = {(size, prefix) for size in SIZES for prefix in PREFIXES}
    if set(indexed) != expected:
        raise RuntimeError(f"grade incompleta em {path}: {set(indexed)}")
    return rows, indexed


def number(value: str | float | int, digits: int = 4) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return "N/A"
    conventional = f"{numeric:,.{digits}f}"
    return conventional.replace(",", "X").replace(".", ",").replace("X", ".")


def integer(value: str | float | int) -> str:
    return f"{int(float(value)):,}".replace(",", ".")


def value(row: dict[str, str], field: str, digits: int = 4, suffix: str = "") -> str:
    return f"{number(row[field], digits)}{suffix}"


def table(lines: list[str], headers: list[str], rows: list[list[str]], align_right: set[int]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    separators = ["---:" if index in align_right else "---" for index in range(len(headers))]
    lines.append("|" + "|".join(separators) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def primary_comparison(cpu: dict[str, str], gpu: dict[str, str]) -> list[list[str]]:
    specifications = [
        ("Variante", "model_variant", None, ""),
        ("Dispositivo", "device", None, ""),
        ("Precisão", "precision", None, ""),
        ("Batch", "batch_size", 0, ""),
        ("Imagens únicas", "unique_images", 0, ""),
        ("Ciclos", "repetitions", 0, ""),
        ("Inferências cronometradas", "collected_inferences", 0, ""),
        ("Acertos", "correct_unique_images", 0, ""),
        ("Acurácia", "accuracy_percent", 4, "%"),
        ("IC95 Wilson — limite inferior", "accuracy_wilson95_low_percent", 4, "%"),
        ("IC95 Wilson — limite superior", "accuracy_wilson95_high_percent", 4, "%"),
        ("Latência média", "latency_mean_ms", 4, " ms"),
        ("IC95 média/ciclo — limite inferior", "latency_mean_pass_ci95_low_ms", 4, " ms"),
        ("IC95 média/ciclo — limite superior", "latency_mean_pass_ci95_high_ms", 4, " ms"),
        ("Latência mediana", "latency_median_ms", 4, " ms"),
        ("Desvio-padrão amostral", "latency_std_ms", 4, " ms"),
        ("Coeficiente de variação", "latency_cv_percent", 4, "%"),
        ("Percentil 90", "latency_p90_ms", 4, " ms"),
        ("Percentil 95", "latency_p95_ms", 4, " ms"),
        ("Percentil 99", "latency_p99_ms", 4, " ms"),
        ("Latência mínima", "latency_min_ms", 4, " ms"),
        ("Latência máxima", "latency_max_ms", 4, " ms"),
        ("Vazão média dos ciclos", "throughput_mean_fps", 4, " inf/s"),
        ("Vazão mediana dos ciclos", "throughput_median_fps", 4, " inf/s"),
        ("Desvio-padrão da vazão", "throughput_std_fps", 4, " inf/s"),
        ("IC95 vazão — limite inferior", "throughput_mean_ci95_low_fps", 4, " inf/s"),
        ("IC95 vazão — limite superior", "throughput_mean_ci95_high_fps", 4, " inf/s"),
        ("Vazão efetiva global", "throughput_effective_fps", 4, " inf/s"),
        ("Tempo total dos ciclos", "wall_time_total_s", 4, " s"),
        ("SHA-256 das predições", "prediction_sha256", None, ""),
    ]
    rows: list[list[str]] = []
    for label, field, digits, suffix in specifications:
        if digits is None:
            cpu_value, gpu_value = cpu[field], gpu[field]
        elif digits == 0:
            cpu_value, gpu_value = integer(cpu[field]), integer(gpu[field])
        else:
            cpu_value = value(cpu, field, digits, suffix)
            gpu_value = value(gpu, field, digits, suffix)
        rows.append([label, cpu_value, gpu_value])
    return rows


def convergence_performance(
    cpu_rows: dict[tuple[int, int], dict[str, str]],
    gpu_rows: dict[tuple[int, int], dict[str, str]],
) -> list[list[str]]:
    output: list[list[str]] = []
    for size in SIZES:
        for prefix in PREFIXES:
            cpu, gpu = cpu_rows[(size, prefix)], gpu_rows[(size, prefix)]
            output.append(
                [
                    integer(size),
                    integer(prefix),
                    value(cpu, "accuracy_percent", 3, "%"),
                    value(gpu, "accuracy_percent", 3, "%"),
                    value(cpu, "latency_mean_ms", 4),
                    value(gpu, "latency_mean_ms", 4),
                    value(cpu, "latency_median_ms", 4),
                    value(gpu, "latency_median_ms", 4),
                    value(cpu, "latency_p95_ms", 4),
                    value(gpu, "latency_p95_ms", 4),
                    value(cpu, "throughput_effective_fps", 2),
                    value(gpu, "throughput_effective_fps", 2),
                ]
            )
    return output


def convergence_energy(
    cpu_rows: dict[tuple[int, int], dict[str, str]],
    gpu_rows: dict[tuple[int, int], dict[str, str]],
) -> list[list[str]]:
    output: list[list[str]] = []
    for size in SIZES:
        for prefix in PREFIXES:
            cpu, gpu = cpu_rows[(size, prefix)], gpu_rows[(size, prefix)]
            output.append(
                [
                    integer(size),
                    integer(prefix),
                    value(cpu, "power_package_mean_w", 3),
                    value(gpu, "power_total_mean_w", 3),
                    value(cpu, "power_dynamic_mean_w", 3),
                    value(gpu, "power_dynamic_mean_w", 3),
                    value(cpu, "energy_per_inference_mj", 4),
                    value(gpu, "energy_per_inference_mj", 4),
                    value(cpu, "dynamic_energy_per_inference_mj", 4),
                    value(gpu, "dynamic_energy_per_inference_mj", 4),
                ]
            )
    return output


def generate() -> str:
    cpu_list, cpu_rows = read_results("CPU")
    gpu_list, gpu_rows = read_results("GPU")
    cpu = cpu_rows[(10000, 100)]
    gpu = gpu_rows[(10000, 100)]
    cpu_meta = read_json(ROOT / "CPU" / "resultados" / "metadados.json")
    gpu_meta = read_json(ROOT / "GPU" / "resultados" / "metadados.json")
    cpu_check = read_json(ROOT / "CPU" / "resultados" / "validacao_integridade.json")
    gpu_check = read_json(ROOT / "GPU" / "resultados" / "validacao_integridade.json")

    if not (cpu_check["status"] == gpu_check["status"] == "valid"):
        raise RuntimeError("uma das campanhas nao passou na validacao")
    if cpu["prediction_sha256"] != gpu["prediction_sha256"]:
        raise RuntimeError("CPU e GPU possuem classes diferentes")

    latency_reduction = 100.0 * (1.0 - float(gpu["latency_mean_ms"]) / float(cpu["latency_mean_ms"]))
    throughput_gain = 100.0 * (float(gpu["throughput_effective_fps"]) / float(cpu["throughput_effective_fps"]) - 1.0)
    total_campaign = sum(size * 100 for size in SIZES)
    lines = [
        "# LeNet/MNIST — benchmark comparativo CPU × GPU",
        "",
        "## Conclusão",
        "",
        "As campanhas CPU e GPU foram concluídas e validadas com o mesmo modelo, dataset, "
        "pré-processamento, seed, batch e protocolo síncrono. Cada dispositivo produziu "
        f"{integer(total_campaign)} latências: 100 ciclos para 100, 1.000 e 10.000 imagens.",
        "",
        f"No resultado principal, ambos acertaram 9.898/10.000 imagens (98,9800%) e "
        "produziram exatamente as mesmas classes. A GPU apresentou redução de "
        f"**{number(latency_reduction, 2)}%** na latência média e ganho de "
        f"**{number(throughput_gain, 2)}%** na vazão efetiva em relação à CPU.",
        "",
        "A energia não deve ser comparada como se tivesse o mesmo limite físico: CPU usa "
        "RAPL do package; GPU usa o sensor da placa. Os valores ficam lado a lado para "
        "referência, sempre com o escopo explicitado.",
        "",
        "## Resultado principal completo — 10.000 imagens × 100 ciclos",
        "",
    ]
    table(lines, ["Métrica", "CPU", "GPU"], primary_comparison(cpu, gpu), {1, 2})

    lines.extend(["## Potência, energia e telemetria lado a lado", ""])
    resource_rows = [
        ["Fonte de energia", "Intel RAPL via turbostat", "Sensor da placa via nvidia-smi"],
        ["Escopo", "Package da CPU", "Placa GPU"],
        ["Intervalo", "100 ms", "100 ms"],
        ["Potência ociosa", f"{number(cpu_meta['rapl_baseline']['idle_power_package_w'], 3)} W", f"{number(gpu_meta['idle_power_mean_w'], 3)} W"],
        ["Potência total média", value(cpu, "power_package_mean_w", 4, " W"), value(gpu, "power_total_mean_w", 4, " W")],
        ["Potência dos cores", value(cpu, "power_cores_mean_w", 4, " W"), "N/A"],
        ["Potência média do sensor de 1 s", "N/A", value(gpu, "power_sensor_1s_mean_w", 4, " W")],
        ["Desvio médio intraciclo da potência", "Não agregado", value(gpu, "power_total_std_w", 4, " W")],
        ["Mínimo médio intraciclo da potência", "Não agregado", value(gpu, "power_total_min_w", 4, " W")],
        ["Máximo médio intraciclo da potência", "Não agregado", value(gpu, "power_total_max_w", 4, " W")],
        ["Potência dinâmica", value(cpu, "power_dynamic_mean_w", 4, " W"), value(gpu, "power_dynamic_mean_w", 4, " W")],
        ["Energia total por inferência", value(cpu, "energy_per_inference_mj", 4, " mJ"), value(gpu, "energy_per_inference_mj", 4, " mJ")],
        ["Energia dinâmica por inferência", value(cpu, "dynamic_energy_per_inference_mj", 4, " mJ"), value(gpu, "dynamic_energy_per_inference_mj", 4, " mJ")],
        ["Energia total acumulada", value(cpu, "energy_package_j", 4, " J"), value(gpu, "energy_total_j", 4, " J")],
        ["Energia dos cores acumulada", value(cpu, "energy_cores_j", 4, " J"), "N/A"],
        ["Energia dinâmica acumulada", value(cpu, "energy_dynamic_j", 4, " J"), value(gpu, "energy_dynamic_j", 4, " J")],
        ["Utilização global", value(cpu, "cpu_busy_percent_turbostat", 4, "% Busy"), value(gpu, "gpu_utilization_percent_mean", 4, "%")],
        ["Desvio médio da utilização", "Não agregado", value(gpu, "gpu_utilization_percent_std", 4, " p.p.")],
        ["Utilização do sistema, psutil", value(cpu, "cpu_system_utilization_mean_percent", 4, "%"), "N/A"],
        ["Utilização do processo, soma dos núcleos", value(cpu, "cpu_process_utilization_mean_percent", 4, "%"), "N/A"],
        ["Utilização normalizada/24 CPUs", value(cpu, "cpu_process_utilization_normalized_percent", 4, "%"), "N/A"],
        ["Utilização da memória", "N/A", value(gpu, "memory_utilization_percent_mean", 4, "%")],
        ["Memória", value(cpu, "process_rss_mean_mib", 4, " MiB RSS"), value(gpu, "memory_used_mib_mean", 4, " MiB VRAM")],
        ["Frequência principal", value(cpu, "cpu_busy_frequency_mhz", 4, " MHz Bzy"), value(gpu, "gpu_clock_sm_mhz_mean", 4, " MHz SM")],
        ["Desvio médio do clock SM", "N/A", value(gpu, "gpu_clock_sm_mhz_std", 4, " MHz")],
        ["Clock da memória", "N/A", value(gpu, "gpu_clock_memory_mhz_mean", 4, " MHz")],
        ["Temperatura média", value(cpu, "cpu_package_temperature_mean_c", 4, " °C package"), value(gpu, "gpu_temperature_c_mean", 4, " °C")],
        ["Máximo médio entre ciclos", value(cpu, "cpu_package_temperature_max_c", 4, " °C"), value(gpu, "gpu_temperature_c_max", 4, " °C")],
        ["Ventoinha", "N/A", value(gpu, "fan_speed_percent_mean", 4, "%")],
        ["Cobertura RAPL média", value(cpu, "rapl_coverage_percent", 4, "%"), "N/A"],
        ["Menor cobertura RAPL em qualquer ciclo", f"{number(cpu_check['minimum_rapl_coverage_percent'], 4)}%", "N/A"],
        ["Amostras na série principal", integer(cpu["rapl_samples_total"]) + " RAPL", integer(gpu["power_samples_total"]) + " nvidia-smi"],
        ["Amostras brutas na campanha", integer(cpu_meta["turbostat_samples_parsed"]) + " RAPL", integer(gpu_check["nvidia_smi_samples"]) + " nvidia-smi"],
        ["Telemetria auxiliar", integer(cpu["telemetry_samples_total"]) + " amostras psutil", "Incluída no nvidia-smi"],
        ["Tempo de CPU acumulado", value(cpu, "process_cpu_time_total_s", 4, " s"), "N/A"],
        ["Tempo de CPU por inferência", value(cpu, "cpu_time_per_inference_mean_ms", 4, " ms"), "N/A"],
        ["Temperatura auxiliar psutil", value(cpu, "cpu_package_temperature_psutil_mean_c", 4, " °C"), "N/A"],
        ["Frequência psutil", "Inválida por escala do kernel; não usada", "N/A"],
    ]
    table(lines, ["Métrica", "CPU", "GPU"], resource_rows, {1, 2})

    lines.extend(["## Convergência de desempenho", ""])
    table(
        lines,
        ["Imagens", "Ciclos", "Acc CPU", "Acc GPU", "Média CPU ms", "Média GPU ms", "Mediana CPU", "Mediana GPU", "p95 CPU", "p95 GPU", "FPS CPU", "FPS GPU"],
        convergence_performance(cpu_rows, gpu_rows),
        set(range(12)),
    )

    lines.extend(
        [
            "## Convergência de potência e energia",
            "",
            "CPU W significa package RAPL; GPU W significa potência da placa. As colunas "
            "não representam o mesmo limite físico.",
            "",
        ]
    )
    table(
        lines,
        ["Imagens", "Ciclos", "CPU W", "GPU W", "CPU W din.", "GPU W din.", "CPU mJ/inf", "GPU mJ/inf", "CPU mJ din.", "GPU mJ din."],
        convergence_energy(cpu_rows, gpu_rows),
        set(range(10)),
    )

    lines.extend(
        [
            "## Modelo e dados",
            "",
            "O arquivo canônico é `lenet_mnist_final.h5`, SHA-256 "
            f"`{cpu_meta['model_sha256']}`. Ele possui entrada `(1, 28, 28, 1)`, "
            "44.426 parâmetros, 281.640 MACs e saída `(1, 10)` linear. Não há softmax; "
            "a classe é `argmax(logits)`.",
            "",
            "| Etapa | Configuração | Saída |",
            "|---|---|---:|",
            "| Entrada | MNIST float32 | 28×28×1 |",
            "| Conv1 | 6 filtros 5×5, valid, ReLU | 24×24×6 |",
            "| Pool1 | MaxPool 2×2 | 12×12×6 |",
            "| Conv2 | 16 filtros 5×5, valid, ReLU | 8×8×16 |",
            "| Pool2 | MaxPool 2×2 | 4×4×16 |",
            "| Flatten | — | 256 |",
            "| Dense1 | 120, ReLU | 120 |",
            "| Dense2 | 84, ReLU | 84 |",
            "| Saída | 10, linear | 10 logits |",
            "",
            "O teste oficial do MNIST foi normalizado antes das medições com "
            "`float32 / 255.0` e recebeu uma dimensão de canal. Os conjuntos de 100 e "
            "1.000 imagens são balanceados e aninhados. O conjunto de 10.000 contém "
            "todo o teste oficial, cuja distribuição é "
            "`[980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]`.",
            "",
            "## Protocolo comum",
            "",
            "1. Seed 20260825, precisão float32 e batch 1.",
            "2. Cem inferências de aquecimento, excluídas das medições.",
            "3. `tf.function` com assinatura fixa, `autograph=False` e `jit_compile=False`.",
            "4. Sem XLA, TensorRT, mixed precision, quantização, batching ou concorrência.",
            "5. Uma imagem por vez; `.numpy()` sincroniza e materializa a saída no host.",
            "6. Normalização fora do cronômetro de latência.",
            "7. Vazão inclui laço Python, criação do tensor, inferência, sincronização, "
            "`argmax` e armazenamento da classe.",
            "8. Cem ciclos para cada tamanho; 10, 20 e 50 são prefixos da mesma série.",
            "9. Ordem interna permutada deterministicamente em cada ciclo.",
            "10. Nenhum outlier removido.",
            "",
            "Na GPU, o dispositivo `/GPU:0` foi obrigatório e o fallback foi desativado. "
            "Na CPU, todas as GPUs foram ocultadas antes da importação do TensorFlow, o "
            "soft placement foi desativado e cada saída foi confirmada em `/CPU:0`.",
            "",
            "## Estatística",
            "",
            "Média, mediana, desvio-padrão amostral, CV, p90, p95, p99, mínimo e "
            "máximo usam todas as latências do prefixo. O IC95% da média usa as médias "
            "dos ciclos como observações:",
            "",
            "```text",
            "IC95 = média_dos_ciclos ± 1,9599639845 × desvio_dos_ciclos / sqrt(K)",
            "```",
            "",
            "A acurácia e o IC95% de Wilson usam apenas imagens únicas. Repetir imagens "
            "serve para caracterizar desempenho; não aumenta artificialmente a amostra "
            "de acurácia. Os ciclos são sequenciais e podem ter autocorrelação temporal; "
            "os prefixos não são campanhas independentes.",
            "",
            "## Potência e energia",
            "",
            "Na CPU, um `turbostat` root independente coletou `Busy%`, `Bzy_MHz`, "
            "`PkgTmp`, `PkgWatt` e `CorWatt` a cada 100 ms. Cada amostra foi ponderada "
            "pela sobreposição entre seu intervalo e o ciclo. `PkgWatt` já contém "
            "`CorWatt`; os dois não foram somados.",
            "",
            "Na GPU, um processo persistente do `nvidia-smi` coletou potência, uso, "
            "memória, clocks, temperatura, ventoinha e estado P a cada 100 ms. Em ambos:",
            "",
            "```text",
            "energia_total = potência_média × duração",
            "potência_dinâmica = max(potência_total − potência_ociosa, 0)",
            "energia_por_inferência = energia_total / número_de_imagens",
            "```",
            "",
            "RAPL é uma estimativa do package da CPU; `nvidia-smi` mede a placa GPU. "
            "Nenhum deles mede o sistema completo na tomada.",
            "",
            "A linha de base GPU foi medida depois do aquecimento e da criacao do contexto ",
            "CUDA. Como o contexto manteve a placa em estado de desempenho elevado, o ",
            "baseline foi 27,651 W e a potencia dinamica calculada ficou pequena. Para a ",
            "GPU, energia total e a referencia mais robusta; energia dinamica deve sempre ",
            "ser apresentada junto dessa ressalva.",
            "",
            "## Integridade",
            "",
            f"- {integer(cpu_check['total_timed_inferences'])} latências CPU e "
            f"{integer(gpu_check['total_timed_inferences'])} latências GPU, todas finitas.",
            "- 100 ciclos completos para cada um dos três tamanhos em ambos os dispositivos.",
            "- Um único hash de predições em todos os ciclos de cada tamanho.",
            "- CPU × GPU: 10.000/10.000 classes iguais, zero divergência.",
            "- 9.898 acertos em ambos os dispositivos.",
            "- Nenhum erro de telemetria GPU ou do coletor auxiliar CPU.",
            "- Energia package positiva e package ≥ cores nos 300 ciclos CPU.",
            f"- Cobertura RAPL mínima: {number(cpu_check['minimum_rapl_coverage_percent'], 4)}%.",
            "- Identidades energia/potência/inferência verificadas numericamente.",
            "",
            "## Ambiente",
            "",
            f"- CPU: {cpu_meta['cpu_model_name']}, {cpu_meta['physical_cpu_count']} núcleos "
            f"físicos e {cpu_meta['logical_cpu_count']} CPUs lógicas; threads TensorFlow automáticas.",
            f"- GPU: {gpu_meta['gpu']['name']}, {gpu_meta['gpu']['memory.total']} MiB, "
            f"driver {gpu_meta['gpu']['driver_version']}, compute capability {gpu_meta['gpu']['compute_cap']}.",
            f"- TensorFlow {cpu_meta['tensorflow_version']}, Keras {cpu_meta['keras_version']}, "
            f"NumPy {cpu_meta['numpy_version']} e Python {cpu_meta['python_version']}.",
            f"- CPU: {cpu_meta['started_utc']} até {cpu_meta['finished_utc']}.",
            f"- GPU: {gpu_meta['started_utc']} até {gpu_meta['finished_utc']}.",
            "- A GPU também dirigia a interface gráfica; CPU e GPU foram medidas em sessões separadas.",
            "- `psutil.cpu_freq()` apresentou escala incorreta (~3 MHz) e foi invalidado; "
            "a frequência CPU válida é `Bzy_MHz` do turbostat.",
            "- LUT, FF, DSP, BRAM, URAM e frequência de síntese não se aplicam a estes ",
            "ensaios; pertencem às implementações hls4ml e Vitis AI.",
            "",
            "## Arquivos reproduzíveis e dados brutos",
            "",
            "Os caminhos abaixo são relativos à raiz do projeto LeNet.",
            "",
            "### CPU",
            "",
            "- `CPU/run_benchmark_rapl.sh`: coordena autenticação, turbostat e benchmark.",
            "- `CPU/scripts/benchmark_lenet_mnist_cpu.py`: inferência, estatística e integração RAPL.",
            "- `CPU/resultados/resultados.csv`: todos os 53 campos agregados em 12 linhas.",
            "- `CPU/resultados/passagens_n*.csv`: todos os valores dos 300 ciclos.",
            "- `CPU/resultados/latencias_n*.npy`: 1.110.000 latências individuais.",
            "- `CPU/resultados/predicoes_n*.csv`: índices, rótulos, classes e logits.",
            "- `CPU/resultados/turbostat_bruto.log`: 4.918 amostras RAPL brutas.",
            "- `CPU/resultados/telemetria_cpu_psutil.csv`: telemetria auxiliar bruta.",
            "- `CPU/resultados/metadados.json` e `validacao_integridade.json`.",
            "",
            "### GPU",
            "",
            "- `GPU/scripts/benchmark_lenet_mnist_gpu.py`: benchmark e telemetria NVIDIA.",
            "- `GPU/resultados/resultados.csv`: todos os 51 campos agregados em 12 linhas.",
            "- `GPU/resultados/passagens_n*.csv`: todos os valores dos 300 ciclos.",
            "- `GPU/resultados/latencias_n*.npy`: 1.110.000 latências individuais.",
            "- `GPU/resultados/predicoes_n*.csv`: índices, rótulos, classes e logits.",
            "- `GPU/resultados/telemetria_nvidia_smi.csv`: 4.030 amostras brutas.",
            "- `GPU/resultados/nvidia_smi_antes.txt` e `nvidia_smi_depois.txt`.",
            "- `GPU/resultados/metadados.json` e `validacao_integridade.json`.",
            "- `scripts/generate_cpu_gpu_readmes.py`: recria os dois READMEs a partir dos CSVs.",
            "",
            "Os READMEs mostram todos os agregados relevantes lado a lado. Os milhões "
            "de valores individuais permanecem nos NPY/CSV para evitar um documento "
            "impraticável e preservar precisão total.",
        ]
    )

    documented_common = {
        "model_variant", "device", "precision", "batch_size", "unique_images",
        "repetitions", "collected_inferences", "correct_unique_images",
        "accuracy_percent", "accuracy_wilson95_low_percent",
        "accuracy_wilson95_high_percent", "latency_mean_ms", "latency_median_ms",
        "latency_std_ms", "latency_cv_percent", "latency_p90_ms", "latency_p95_ms",
        "latency_p99_ms", "latency_min_ms", "latency_max_ms",
        "latency_mean_pass_ci95_low_ms", "latency_mean_pass_ci95_high_ms",
        "throughput_mean_fps", "throughput_median_fps", "throughput_std_fps",
        "throughput_mean_ci95_low_fps", "throughput_mean_ci95_high_fps",
        "throughput_effective_fps", "wall_time_total_s", "prediction_sha256",
    }
    cpu_specific = set(cpu_list[0]) - documented_common
    gpu_specific = set(gpu_list[0]) - documented_common
    expected_cpu = {
        "process_cpu_time_total_s", "cpu_time_per_inference_mean_ms",
        "cpu_system_utilization_mean_percent", "cpu_process_utilization_mean_percent",
        "cpu_process_utilization_normalized_percent", "cpu_frequency_psutil_mean_mhz",
        "cpu_package_temperature_psutil_mean_c", "process_rss_mean_mib",
        "rapl_coverage_percent", "cpu_busy_percent_turbostat", "cpu_busy_frequency_mhz",
        "cpu_package_temperature_mean_c", "cpu_package_temperature_max_c",
        "power_package_mean_w", "power_cores_mean_w", "power_dynamic_mean_w",
        "energy_per_inference_mj", "dynamic_energy_per_inference_mj",
        "energy_package_j", "energy_cores_j", "energy_dynamic_j",
        "telemetry_samples_total", "rapl_samples_total",
    }
    expected_gpu = {
        "power_total_mean_w", "power_total_std_w", "power_total_min_w",
        "power_total_max_w", "power_sensor_1s_mean_w", "power_dynamic_mean_w",
        "energy_per_inference_mj", "dynamic_energy_per_inference_mj",
        "gpu_utilization_percent_mean", "gpu_utilization_percent_std",
        "memory_utilization_percent_mean", "memory_used_mib_mean",
        "gpu_clock_sm_mhz_mean", "gpu_clock_sm_mhz_std",
        "gpu_clock_memory_mhz_mean", "gpu_temperature_c_mean",
        "gpu_temperature_c_max", "fan_speed_percent_mean", "energy_total_j",
        "energy_dynamic_j", "power_samples_total",
    }
    if cpu_specific != expected_cpu or gpu_specific != expected_gpu:
        raise RuntimeError(
            f"campos nao documentados: CPU={cpu_specific ^ expected_cpu}, "
            f"GPU={gpu_specific ^ expected_gpu}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = generate()
    for device in ("CPU", "GPU"):
        path = ROOT / device / "README.md"
        path.write_text(report, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
