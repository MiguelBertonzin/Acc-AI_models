# LeNet/MNIST — benchmark comparativo CPU × GPU

## Conclusão

As campanhas CPU e GPU foram concluídas e validadas com o mesmo modelo, dataset, pré-processamento, seed, batch e protocolo síncrono. Cada dispositivo produziu 1.110.000 latências: 100 ciclos para 100, 1.000 e 10.000 imagens.

No resultado principal, ambos acertaram 9.898/10.000 imagens (98,9800%) e produziram exatamente as mesmas classes. A GPU apresentou redução de **38,14%** na latência média e ganho de **23,47%** na vazão efetiva em relação à CPU.

A energia não deve ser comparada como se tivesse o mesmo limite físico: CPU usa RAPL do package; GPU usa o sensor da placa. Os valores ficam lado a lado para referência, sempre com o escopo explicitado.

## Resultado principal completo — 10.000 imagens × 100 ciclos

| Métrica | CPU | GPU |
|---|---:|---:|
| Variante | no_softmax_logits | no_softmax_logits |
| Dispositivo | CPU | GPU |
| Precisão | float32 | float32 |
| Batch | 1 | 1 |
| Imagens únicas | 10.000 | 10.000 |
| Ciclos | 100 | 100 |
| Inferências cronometradas | 1.000.000 | 1.000.000 |
| Acertos | 9.898 | 9.898 |
| Acurácia | 98,9800% | 98,9800% |
| IC95 Wilson — limite inferior | 98,7634% | 98,7634% |
| IC95 Wilson — limite superior | 99,1590% | 99,1590% |
| Latência média | 0,3872 ms | 0,2395 ms |
| IC95 média/ciclo — limite inferior | 0,3843 ms | 0,2349 ms |
| IC95 média/ciclo — limite superior | 0,3900 ms | 0,2441 ms |
| Latência mediana | 0,3423 ms | 0,2085 ms |
| Desvio-padrão amostral | 0,1339 ms | 0,0824 ms |
| Coeficiente de variação | 34,5779% | 34,3814% |
| Percentil 90 | 0,5552 ms | 0,3285 ms |
| Percentil 95 | 0,6395 ms | 0,4143 ms |
| Percentil 99 | 0,9520 ms | 0,5695 ms |
| Latência mínima | 0,2385 ms | 0,1795 ms |
| Latência máxima | 5,6823 ms | 2,1021 ms |
| Vazão média dos ciclos | 2.292,7288 inf/s | 2.841,6973 inf/s |
| Vazão mediana dos ciclos | 2.281,1518 inf/s | 2.829,5367 inf/s |
| Desvio-padrão da vazão | 86,9430 inf/s | 201,7835 inf/s |
| IC95 vazão — limite inferior | 2.275,6883 inf/s | 2.802,1485 inf/s |
| IC95 vazão — limite superior | 2.309,7694 inf/s | 2.881,2462 inf/s |
| Vazão efetiva global | 2.289,4858 inf/s | 2.826,7257 inf/s |
| Tempo total dos ciclos | 436,7793 s | 353,7662 s |
| SHA-256 das predições | 2e1bd57ae813232b84b89428876c305a3cd6f07dd5b2f8f8566f808aec004beb | 2e1bd57ae813232b84b89428876c305a3cd6f07dd5b2f8f8566f808aec004beb |

## Potência, energia e telemetria lado a lado

| Métrica | CPU | GPU |
|---|---:|---:|
| Fonte de energia | Intel RAPL via turbostat | Sensor da placa via nvidia-smi |
| Escopo | Package da CPU | Placa GPU |
| Intervalo | 100 ms | 100 ms |
| Potência ociosa | 12,567 W | 27,651 W |
| Potência total média | 39,7820 W | 28,2483 W |
| Potência dos cores | 30,4457 W | N/A |
| Potência média do sensor de 1 s | N/A | 28,2525 W |
| Desvio médio intraciclo da potência | Não agregado | 0,1353 W |
| Mínimo médio intraciclo da potência | Não agregado | 28,0784 W |
| Máximo médio intraciclo da potência | Não agregado | 28,4921 W |
| Potência dinâmica | 27,2154 W | 0,5973 W |
| Energia total por inferência | 17,3548 mJ | 9,9970 mJ |
| Energia dinâmica por inferência | 11,8660 mJ | 0,2150 mJ |
| Energia total acumulada | 17.354,7949 J | 9.996,9597 J |
| Energia dos cores acumulada | 13.274,5651 J | N/A |
| Energia dinâmica acumulada | 11.865,9937 J | 214,9707 J |
| Utilização global | 17,6018% Busy | 21,7057% |
| Desvio médio da utilização | Não agregado | 2,0210 p.p. |
| Utilização do sistema, psutil | 12,0047% | N/A |
| Utilização do processo, soma dos núcleos | 289,7041% | N/A |
| Utilização normalizada/24 CPUs | 12,0710% | N/A |
| Utilização da memória | N/A | 0,8885% |
| Memória | 795,3217 MiB RSS | 723,9833 MiB VRAM |
| Frequência principal | 3.040,2443 MHz Bzy | 1.755,0000 MHz SM |
| Desvio médio do clock SM | N/A | 0,0000 MHz |
| Clock da memória | N/A | 6.801,0000 MHz |
| Temperatura média | 67,6227 °C package | 49,9955 °C |
| Máximo médio entre ciclos | 71,6600 °C | 50,1000 °C |
| Ventoinha | N/A | 31,0000% |
| Cobertura RAPL média | 99,2149% | N/A |
| Menor cobertura RAPL em qualquer ciclo | 95,0823% | N/A |
| Amostras na série principal | 4.433 RAPL | 3.504 nvidia-smi |
| Amostras brutas na campanha | 4.918 RAPL | 4.030 nvidia-smi |
| Telemetria auxiliar | 2.777 amostras psutil | Incluída no nvidia-smi |
| Tempo de CPU acumulado | 1.268,8521 s | N/A |
| Tempo de CPU por inferência | 1,2689 ms | N/A |
| Temperatura auxiliar psutil | 68,6872 °C | N/A |
| Frequência psutil | Inválida por escala do kernel; não usada | N/A |

## Convergência de desempenho

| Imagens | Ciclos | Acc CPU | Acc GPU | Média CPU ms | Média GPU ms | Mediana CPU | Mediana GPU | p95 CPU | p95 GPU | FPS CPU | FPS GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 99,000% | 99,000% | 0,3971 | 0,3610 | 0,3371 | 0,3014 | 0,7082 | 0,7297 | 2.232,07 | 2.039,46 |
| 100 | 20 | 99,000% | 99,000% | 0,4034 | 0,3716 | 0,3440 | 0,3343 | 0,6851 | 0,6969 | 2.195,53 | 1.959,72 |
| 100 | 50 | 99,000% | 99,000% | 0,3873 | 0,3187 | 0,3395 | 0,2609 | 0,6603 | 0,5821 | 2.286,01 | 2.220,67 |
| 100 | 100 | 99,000% | 99,000% | 0,3816 | 0,2835 | 0,3378 | 0,2199 | 0,6347 | 0,5363 | 2.320,76 | 2.507,82 |
| 1.000 | 10 | 98,800% | 98,800% | 0,3830 | 0,2586 | 0,3367 | 0,2118 | 0,6442 | 0,5053 | 2.312,53 | 2.677,08 |
| 1.000 | 20 | 98,800% | 98,800% | 0,3853 | 0,2814 | 0,3394 | 0,2187 | 0,6395 | 0,5421 | 2.298,73 | 2.468,76 |
| 1.000 | 50 | 98,800% | 98,800% | 0,3802 | 0,2684 | 0,3370 | 0,2164 | 0,6316 | 0,5198 | 2.330,41 | 2.587,04 |
| 1.000 | 100 | 98,800% | 98,800% | 0,3813 | 0,2847 | 0,3382 | 0,2252 | 0,6330 | 0,5447 | 2.323,75 | 2.425,72 |
| 10.000 | 10 | 98,980% | 98,980% | 0,3790 | 0,2462 | 0,3371 | 0,2127 | 0,6263 | 0,4317 | 2.337,86 | 2.729,23 |
| 10.000 | 20 | 98,980% | 98,980% | 0,3816 | 0,2506 | 0,3388 | 0,2124 | 0,6303 | 0,4550 | 2.321,49 | 2.709,97 |
| 10.000 | 50 | 98,980% | 98,980% | 0,3824 | 0,2481 | 0,3392 | 0,2108 | 0,6332 | 0,4506 | 2.318,31 | 2.739,60 |
| 10.000 | 100 | 98,980% | 98,980% | 0,3872 | 0,2395 | 0,3423 | 0,2085 | 0,6395 | 0,4143 | 2.289,49 | 2.826,73 |

## Convergência de potência e energia

CPU W significa package RAPL; GPU W significa potência da placa. As colunas não representam o mesmo limite físico.

| Imagens | Ciclos | CPU W | GPU W | CPU W din. | GPU W din. | CPU mJ/inf | GPU mJ/inf | CPU mJ din. | GPU mJ din. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 38,962 | 27,810 | 26,396 | 0,159 | 16,6432 | 14,9096 | 11,0132 | 0,0852 |
| 100 | 20 | 37,390 | 28,421 | 24,824 | 0,770 | 16,5496 | 15,5763 | 10,8259 | 0,4276 |
| 100 | 50 | 39,208 | 28,460 | 26,642 | 0,809 | 16,8430 | 14,0968 | 11,3458 | 0,4032 |
| 100 | 100 | 39,528 | 28,252 | 26,962 | 0,600 | 16,8188 | 12,3945 | 11,4040 | 0,2777 |
| 1.000 | 10 | 37,971 | 28,097 | 25,405 | 0,446 | 16,3135 | 10,5010 | 10,8794 | 0,1723 |
| 1.000 | 20 | 37,512 | 28,568 | 24,945 | 0,917 | 16,2399 | 11,5873 | 10,7732 | 0,3869 |
| 1.000 | 50 | 38,093 | 28,271 | 25,526 | 0,620 | 16,2650 | 10,9372 | 10,8726 | 0,2489 |
| 1.000 | 100 | 38,137 | 28,729 | 25,570 | 1,078 | 16,3369 | 11,8713 | 10,9290 | 0,4723 |
| 10.000 | 10 | 39,224 | 28,643 | 26,658 | 0,992 | 16,7580 | 10,4957 | 11,3828 | 0,3643 |
| 10.000 | 20 | 39,259 | 28,548 | 26,693 | 0,897 | 16,8974 | 10,5347 | 11,4842 | 0,3313 |
| 10.000 | 50 | 39,999 | 28,463 | 27,433 | 0,812 | 17,2351 | 10,3919 | 11,8145 | 0,2988 |
| 10.000 | 100 | 39,782 | 28,248 | 27,215 | 0,597 | 17,3548 | 9,9970 | 11,8660 | 0,2150 |

## Modelo e dados

O arquivo canônico é `lenet_mnist_final.h5`, SHA-256 `7d052aa27f18565af0a948e656cbd23eac842374ac73c0aaf18e98d5d913226c`. Ele possui entrada `(1, 28, 28, 1)`, 44.426 parâmetros, 281.640 MACs e saída `(1, 10)` linear. Não há softmax; a classe é `argmax(logits)`.

| Etapa | Configuração | Saída |
|---|---|---:|
| Entrada | MNIST float32 | 28×28×1 |
| Conv1 | 6 filtros 5×5, valid, ReLU | 24×24×6 |
| Pool1 | MaxPool 2×2 | 12×12×6 |
| Conv2 | 16 filtros 5×5, valid, ReLU | 8×8×16 |
| Pool2 | MaxPool 2×2 | 4×4×16 |
| Flatten | — | 256 |
| Dense1 | 120, ReLU | 120 |
| Dense2 | 84, ReLU | 84 |
| Saída | 10, linear | 10 logits |

O teste oficial do MNIST foi normalizado antes das medições com `float32 / 255.0` e recebeu uma dimensão de canal. Os conjuntos de 100 e 1.000 imagens são balanceados e aninhados. O conjunto de 10.000 contém todo o teste oficial, cuja distribuição é `[980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]`.

## Protocolo comum

1. Seed 20260825, precisão float32 e batch 1.
2. Cem inferências de aquecimento, excluídas das medições.
3. `tf.function` com assinatura fixa, `autograph=False` e `jit_compile=False`.
4. Sem XLA, TensorRT, mixed precision, quantização, batching ou concorrência.
5. Uma imagem por vez; `.numpy()` sincroniza e materializa a saída no host.
6. Normalização fora do cronômetro de latência.
7. Vazão inclui laço Python, criação do tensor, inferência, sincronização, `argmax` e armazenamento da classe.
8. Cem ciclos para cada tamanho; 10, 20 e 50 são prefixos da mesma série.
9. Ordem interna permutada deterministicamente em cada ciclo.
10. Nenhum outlier removido.

Na GPU, o dispositivo `/GPU:0` foi obrigatório e o fallback foi desativado. Na CPU, todas as GPUs foram ocultadas antes da importação do TensorFlow, o soft placement foi desativado e cada saída foi confirmada em `/CPU:0`.

## Estatística

Média, mediana, desvio-padrão amostral, CV, p90, p95, p99, mínimo e máximo usam todas as latências do prefixo. O IC95% da média usa as médias dos ciclos como observações:

```text
IC95 = média_dos_ciclos ± 1,9599639845 × desvio_dos_ciclos / sqrt(K)
```

A acurácia e o IC95% de Wilson usam apenas imagens únicas. Repetir imagens serve para caracterizar desempenho; não aumenta artificialmente a amostra de acurácia. Os ciclos são sequenciais e podem ter autocorrelação temporal; os prefixos não são campanhas independentes.

## Potência e energia

Na CPU, um `turbostat` root independente coletou `Busy%`, `Bzy_MHz`, `PkgTmp`, `PkgWatt` e `CorWatt` a cada 100 ms. Cada amostra foi ponderada pela sobreposição entre seu intervalo e o ciclo. `PkgWatt` já contém `CorWatt`; os dois não foram somados.

Na GPU, um processo persistente do `nvidia-smi` coletou potência, uso, memória, clocks, temperatura, ventoinha e estado P a cada 100 ms. Em ambos:

```text
energia_total = potência_média × duração
potência_dinâmica = max(potência_total − potência_ociosa, 0)
energia_por_inferência = energia_total / número_de_imagens
```

RAPL é uma estimativa do package da CPU; `nvidia-smi` mede a placa GPU. Nenhum deles mede o sistema completo na tomada.

A linha de base GPU foi medida depois do aquecimento e da criacao do contexto 
CUDA. Como o contexto manteve a placa em estado de desempenho elevado, o 
baseline foi 27,651 W e a potencia dinamica calculada ficou pequena. Para a 
GPU, energia total e a referencia mais robusta; energia dinamica deve sempre 
ser apresentada junto dessa ressalva.

## Integridade

- 1.110.000 latências CPU e 1.110.000 latências GPU, todas finitas.
- 100 ciclos completos para cada um dos três tamanhos em ambos os dispositivos.
- Um único hash de predições em todos os ciclos de cada tamanho.
- CPU × GPU: 10.000/10.000 classes iguais, zero divergência.
- 9.898 acertos em ambos os dispositivos.
- Nenhum erro de telemetria GPU ou do coletor auxiliar CPU.
- Energia package positiva e package ≥ cores nos 300 ciclos CPU.
- Cobertura RAPL mínima: 95,0823%.
- Identidades energia/potência/inferência verificadas numericamente.

## Ambiente

- CPU: 13th Gen Intel(R) Core(TM) i7-13700, 16 núcleos físicos e 24 CPUs lógicas; threads TensorFlow automáticas.
- GPU: NVIDIA GeForce RTX 3050 OEM, 8192 MiB, driver 580.173.02, compute capability 8.6.
- TensorFlow 2.21.0, Keras 3.13.2, NumPy 1.26.4 e Python 3.12.7.
- CPU: 2026-09-03T20:16:43.728303+00:00 até 2026-09-03T20:24:52.512449+00:00.
- GPU: 2026-09-03T18:39:07.962040+00:00 até 2026-09-03T18:45:49.690009+00:00.
- A GPU também dirigia a interface gráfica; CPU e GPU foram medidas em sessões separadas.
- `psutil.cpu_freq()` apresentou escala incorreta (~3 MHz) e foi invalidado; a frequência CPU válida é `Bzy_MHz` do turbostat.
- LUT, FF, DSP, BRAM, URAM e frequência de síntese não se aplicam a estes 
ensaios; pertencem às implementações hls4ml e Vitis AI.

## Arquivos reproduzíveis e dados brutos

Os caminhos abaixo são relativos à raiz do projeto LeNet.

### CPU

- `CPU/run_benchmark_rapl.sh`: coordena autenticação, turbostat e benchmark.
- `CPU/scripts/benchmark_lenet_mnist_cpu.py`: inferência, estatística e integração RAPL.
- `CPU/resultados/resultados.csv`: todos os 53 campos agregados em 12 linhas.
- `CPU/resultados/passagens_n*.csv`: todos os valores dos 300 ciclos.
- `CPU/resultados/latencias_n*.npy`: 1.110.000 latências individuais.
- `CPU/resultados/predicoes_n*.csv`: índices, rótulos, classes e logits.
- `CPU/resultados/turbostat_bruto.log`: 4.918 amostras RAPL brutas.
- `CPU/resultados/telemetria_cpu_psutil.csv`: telemetria auxiliar bruta.
- `CPU/resultados/metadados.json` e `validacao_integridade.json`.

### GPU

- `GPU/scripts/benchmark_lenet_mnist_gpu.py`: benchmark e telemetria NVIDIA.
- `GPU/resultados/resultados.csv`: todos os 51 campos agregados em 12 linhas.
- `GPU/resultados/passagens_n*.csv`: todos os valores dos 300 ciclos.
- `GPU/resultados/latencias_n*.npy`: 1.110.000 latências individuais.
- `GPU/resultados/predicoes_n*.csv`: índices, rótulos, classes e logits.
- `GPU/resultados/telemetria_nvidia_smi.csv`: 4.030 amostras brutas.
- `GPU/resultados/nvidia_smi_antes.txt` e `nvidia_smi_depois.txt`.
- `GPU/resultados/metadados.json` e `validacao_integridade.json`.
- `scripts/generate_cpu_gpu_readmes.py`: recria os dois READMEs a partir dos CSVs.

Os READMEs mostram todos os agregados relevantes lado a lado. Os milhões de valores individuais permanecem nos NPY/CSV para evitar um documento impraticável e preservar precisão total.
