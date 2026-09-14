# Resultados CPU — MLP Iris

Metodologia normativa comum às plataformas: [`../METODOLOGIA_BENCHMARK_MLP_IRIS.md`](../METODOLOGIA_BENCHMARK_MLP_IRIS.md).

Este README consolida todos os ensaios da MLP Iris na CPU Intel Core i7-13700. A referência principal é a média de **cinco campanhas independentes de 100.000 inferências, batch 1**.

## Resumo executivo

| Métrica | 30.000 × 5 | 100.000 × 5 |
|---|---:|---:|
| Inference-only média | 0.112886 ms | 0.112399 ms |
| End-to-end efetiva | 0.137473 ms | 0.137069 ms |
| Vazão efetiva | 7275.55 inf/s | 7301.88 inf/s |
| Potência package RAPL | 53.444 W | 55.245 W |
| Energia total package | 7.685 mJ/inf | 7.927 mJ/inf |
| Energia dinâmica package | 5.974 mJ/inf | 6.121 mJ/inf |
| Acurácia | 29/30 = 96,67% | 29/30 = 96,67% |
| Concordância de classes CPU×GPU | 100% | 100% |

## Configuração

| Item | Valor |
|---|---|
| CPU | Intel Core i7-13700 |
| Núcleos | 16 físicos / 24 lógicos |
| Rede | 4 → 8 ReLU → 8 ReLU → 3 softmax |
| Parâmetros / precisão | 139 / `float32` |
| Runtime | TensorFlow 2.21.0, oneDNN, `tf.function` |
| Batch / fluxo | 1 / serial e síncrono |
| Warm-up / baseline | 200 inferências / 5 s |
| Telemetria | 100 ms |
| Réplicas | 5 de 30k + 5 de 100k |
| Seed / outliers | 20260831 / nenhum removido |
| Fonte energética | `intel-rapl:0/package-0/energy_uj` |
| Escopo | Pacote da CPU; não inclui tomada |

## Validação do modelo

| Métrica | Resultado |
|---|---:|
| Holdout único | 30 amostras |
| Acertos / acurácia | 29/30 / 96,67% |
| IC95% Wilson | [83,33%; 99,41%] |
| F1 macro | 0,96658 |
| MCC | 0,95159 |
| ROC AUC macro | 0,99667 |
| Concordância CPU×GPU | 100% |
| Máxima diferença de probabilidade | 1,192 × 10⁻⁷ |
| Determinismo | 100% |

As repetições medem desempenho e determinismo; não aumentam o tamanho do holdout usado para acurácia.

## Desempenho agregado

| Métrica | 30k média [IC95%] | Mediana | CV | 100k média [IC95%] | Mediana | CV |
|---|---:|---:|---:|---:|---:|---:|
| Inference-only (ms) | 0.112886 [0.110670; 0.115102] | 0.112725 | 1.58% | 0.112399 [0.107717; 0.117082] | 0.110878 | 3.36% |
| End-to-end efetiva (ms) | 0.137473 [0.134838; 0.140108] | 0.137186 | 1.54% | 0.137069 [0.131422; 0.142715] | 0.135240 | 3.32% |
| Vazão efetiva (inf/s) | 7275.55 [7136.58; 7414.51] | 7289.40 | 1.54% | 7301.88 [7009.16; 7594.59] | 7394.24 | 3.23% |

## Energia e potência RAPL

| Métrica | 30k média [IC95%] | Mediana | CV | 100k média [IC95%] | Mediana | CV |
|---|---:|---:|---:|---:|---:|---:|
| Potência package (W) | 53.444 [50.392; 56.497] | 53.103 | 4.60% | 55.245 [50.288; 60.202] | 54.140 | 7.23% |
| Potência ociosa package (W) | 11.898 [10.793; 13.003] | 11.512 | 7.48% | 12.569 [10.624; 14.513] | 11.740 | 12.46% |
| Potência dinâmica package (W) | 41.546 [39.448; 43.644] | 41.515 | 4.07% | 42.677 [39.505; 45.849] | 42.400 | 5.99% |
| Potência core (W) | 44.653 [41.704; 47.603] | 44.286 | 5.32% | 46.312 [41.590; 51.035] | 45.210 | 8.21% |
| Energia total package (mJ/inf) | 7.685 [7.128; 8.243] | 7.557 | 5.85% | 7.927 [6.899; 8.954] | 7.651 | 10.44% |
| Energia dinâmica package (mJ/inf) | 5.974 [5.591; 6.356] | 5.918 | 5.16% | 6.121 [5.424; 6.819] | 5.992 | 9.18% |

`package` é a métrica principal e já contém os subdomínios. `core` e `uncore` não são somados novamente.

## Temperatura

| Métrica | 30.000 × 5 | 100.000 × 5 |
|---|---:|---:|
| Package média | 68.8 °C | 73.6 °C |
| IC95% da média | [66.7; 70.8] °C | [70.4; 76.8] °C |
| Pico médio | 74.6 °C | 78.8 °C |
| Faixa dos picos | 71–77 °C | 73–82 °C |

## Testes pareados: 100k menos 30k

| Métrica | Diferença | IC95% | p | Resultado |
|---|---:|---:|---:|---|
| Energia total (mJ/inf) | +0.241 | [-0.398; +0.881] | 0.354 | Não significativa |
| Energia dinâmica (mJ/inf) | +0.147 | [-0.241; +0.536] | 0.351 | Não significativa |
| Potência package (W) | +1.801 | [-0.900; +4.502] | 0.138 | Não significativa |
| Inference-only (ms) | -0.000487 | [-0.004570; +0.003597] | 0.757 | Não significativa |
| End-to-end (ms) | -0.000404 | [-0.005098; +0.004290] | 0.823 | Não significativa |
| Vazão (inf/s) | +26.33 | [-221.17; +273.83] | 0.782 | Não significativa |

A unidade estatística é a campanha independente. Nenhuma campanha foi removida. Não foi encontrada diferença significativa entre 30k e 100k.

## Campanhas individuais

| Inferências | Réplica | Inf-only ms | E2E ms | inf/s | Package W | Idle W | Total mJ/inf | Dinâmica mJ/inf | Temp. média °C | Pico °C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 1 | 0.115357 | 0.140597 | 7112.53 | 57.411 | 13.487 | 8.442 | 6.459 | 66.5 | 77.0 |
| 30000 | 2 | 0.113360 | 0.137990 | 7246.92 | 53.103 | 11.588 | 7.659 | 5.987 | 70.9 | 75.0 |
| 30000 | 3 | 0.110375 | 0.134728 | 7422.37 | 53.348 | 11.512 | 7.521 | 5.898 | 69.4 | 77.0 |
| 30000 | 4 | 0.112725 | 0.136864 | 7306.53 | 50.658 | 11.480 | 7.249 | 5.606 | 68.0 | 71.0 |
| 30000 | 5 | 0.112612 | 0.137186 | 7289.40 | 52.701 | 11.426 | 7.557 | 5.918 | 69.1 | 73.0 |
| 100000 | 1 | 0.118547 | 0.144457 | 6922.47 | 60.356 | 14.248 | 9.098 | 6.950 | 75.5 | 82.0 |
| 100000 | 2 | 0.110878 | 0.135240 | 7394.24 | 54.140 | 11.740 | 7.651 | 5.992 | 73.9 | 80.0 |
| 100000 | 3 | 0.113392 | 0.138320 | 7229.62 | 58.375 | 14.283 | 8.443 | 6.378 | 76.6 | 82.0 |
| 100000 | 4 | 0.109510 | 0.133577 | 7486.30 | 50.915 | 11.476 | 7.110 | 5.508 | 70.5 | 73.0 |
| 100000 | 5 | 0.109670 | 0.133748 | 7476.75 | 52.441 | 11.096 | 7.330 | 5.779 | 71.4 | 77.0 |

## Resultados históricos sem energia

| Campanha | Inference-only | End-to-end efetiva | Vazão |
|---|---:|---:|---:|
| 30.000 original | 0,111120 ms | 0,135388 ms | 7.386,17 inf/s |
| 100.000 original | 0,108883 ms | 0,132613 ms | 7.540,74 inf/s |

Esses ensaios permanecem válidos para desempenho, mas não possuem energia. Os ensaios RAPL são novas execuções, portanto pequenas diferenças decorrem de temperatura, DVFS e carga do sistema.

## Fronteiras metodológicas

| Janela | Início | Fim |
|---|---|---|
| Inference-only | Após criar o tensor | Após `.numpy()`, com saída sincronizada |
| End-to-end | Antes de criar o tensor | Após saída e `argmax` |
| Vazão efetiva | Início da passagem | Fim da passagem, incluindo laço e armazenamento |
| Energia | Leitura RAPL antes da carga | Leitura RAPL após a carga |
| Energia dinâmica | Energia ativa | Menos potência ociosa × duração |

A normalização fica fora da janela. O resultado é batch 1 serial e síncrono, não throughput saturado.

## Comparação CPU, GPU e FPGA

| Plataforma | Escopo energético |
|---|---|
| CPU | Package RAPL |
| GPU | Sensor da placa via `nvidia-smi` |
| FPGA futura | Rails da placa/dispositivo e host separados |

Baseline CPU recomendado: **7.927 mJ/inferência**, IC95% [6.899; 8.954]. O valor GPU de 8,193 mJ/inferência tem outro escopo físico; a comparação direta é apenas indicativa.

## Reprodução

```bash
python3 scripts/benchmark_mlp_iris_cpu.py \
  --inferences 100000 \
  --warmup 200 \
  --idle-seconds 5 \
  --telemetry-ms 100 \
  --output-dir CPU/nova_execucao_100000_rapl \
  --require-rapl

python3 scripts/analyze_cpu_rapl.py
```

## Após o ensaio: RAPL e módulo MSR

Nenhum processo ou serviço ficou executando. `modprobe msr` apenas carregou um módulo do kernel. As permissões voltam ao padrão após reiniciar.

Para restaurar agora:

```bash
sudo find /dev/cpu -mindepth 2 -maxdepth 2 -name msr -exec chmod 600 {} +
sudo chmod 400 /sys/class/powercap/intel-rapl:0/energy_uj
sudo chmod 400 /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj
sudo chmod 400 /sys/class/powercap/intel-rapl:0/intel-rapl:0:1/energy_uj
sudo modprobe -r msr
```

Não descarregue `intel_rapl_msr`, `intel_rapl_common` nem `processor_thermal_rapl`. Eles fornecem o powercap normal. Para novos testes, recarregue `msr` e reaplique as permissões.

## Arquivos

| Caminho | Conteúdo |
|---|---|
| `VALIDACAO_ENERGIA_RAPL.md` | Relatório detalhado |
| `validacao_rapl/campanhas_rapl.csv` | Dez campanhas |
| `validacao_rapl/resumo_estatistico_rapl.json` | Estatísticas e testes |
| `resultados_30000_rapl/` | Primeira réplica 30k |
| `resultados_100000_rapl/` | Primeira réplica 100k |
| `validacao_rapl/30000_rep2..5/` | Outras réplicas 30k |
| `validacao_rapl/100000_rep2..5/` | Outras réplicas 100k |
| `resultados_30000/` e `resultados_100000/` | Ensaios históricos |
| `COMPARACAO_CPU_30000_100000_E_GPU.md` | Comparação histórica |
