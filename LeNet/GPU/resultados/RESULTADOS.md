# Resultados — LeNet/MNIST sem softmax na GPU

## Resultado principal

O resultado principal usa todas as 10.000 imagens oficiais de teste e 100 ciclos. Os pontos de 10, 20 e 50 ciclos sao prefixos da mesma coleta.

| Metrica | Resultado |
|---|---:|
| GPU | NVIDIA GeForce RTX 3050 OEM |
| Precisao | float32 |
| Batch | 1 |
| Imagens unicas | 10000 |
| Ciclos | 100 |
| Inferencias cronometradas | 1000000 |
| Acertos | 9898 |
| Acuracia | 98.9800% |
| IC95 Wilson | [98.7634%; 99.1590%] |
| Latencia media | 0.2395 ms |
| IC95 da media por ciclo | [0.2349; 0.2441] ms |
| Mediana | 0.2085 ms |
| Desvio padrao | 0.0824 ms |
| Coeficiente de variacao | 34.38% |
| p90 / p95 / p99 | 0.3285 / 0.4143 / 0.5695 ms |
| Minimo / maximo | 0.1795 / 2.1021 ms |
| Vazao media | 2841.697 inferencias/s |
| Vazao efetiva global | 2826.726 inferencias/s |
| Potencia total media | 28.248 W |
| Potencia dinamica media | 0.597 W |
| Energia total/inferencia | 9.9970 mJ |
| Energia dinamica/inferencia | 0.2150 mJ |
| Utilizacao GPU media | 21.706% |
| Memoria GPU usada | 723.983 MiB |
| Clock SM medio | 1755.000 MHz |
| Temperatura media / maxima | 49.995 / 50.100 C |
| Amostras nvidia-smi | 3504 |
| Potencia ociosa | 27.651 W |

## Convergencia de todas as series

| Imagens | Ciclos | Inferencias | Acuracia (%) | Media (ms) | Mediana (ms) | p95 (ms) | FPS medio | FPS efetivo | Potencia (W) | Energia/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 1000 | 99.000 | 0.3610 | 0.3014 | 0.7297 | 2185.44 | 2039.46 | 27.810 | 14.9096 |
| 100 | 20 | 2000 | 99.000 | 0.3716 | 0.3343 | 0.6969 | 2097.28 | 1959.72 | 28.421 | 15.5763 |
| 100 | 50 | 5000 | 99.000 | 0.3187 | 0.2609 | 0.5821 | 2345.54 | 2220.67 | 28.460 | 14.0968 |
| 100 | 100 | 10000 | 99.000 | 0.2835 | 0.2199 | 0.5363 | 2665.61 | 2507.82 | 28.252 | 12.3945 |
| 1000 | 10 | 10000 | 98.800 | 0.2586 | 0.2118 | 0.5053 | 2728.11 | 2677.08 | 28.097 | 10.5010 |
| 1000 | 20 | 20000 | 98.800 | 0.2814 | 0.2187 | 0.5421 | 2533.27 | 2468.76 | 28.568 | 11.5873 |
| 1000 | 50 | 50000 | 98.800 | 0.2684 | 0.2164 | 0.5198 | 2643.69 | 2587.04 | 28.271 | 10.9372 |
| 1000 | 100 | 100000 | 98.800 | 0.2847 | 0.2252 | 0.5447 | 2493.44 | 2425.72 | 28.729 | 11.8713 |
| 10000 | 10 | 100000 | 98.980 | 0.2462 | 0.2127 | 0.4317 | 2735.88 | 2729.23 | 28.643 | 10.4957 |
| 10000 | 20 | 200000 | 98.980 | 0.2506 | 0.2124 | 0.4550 | 2716.67 | 2709.97 | 28.548 | 10.5347 |
| 10000 | 50 | 500000 | 98.980 | 0.2481 | 0.2108 | 0.4506 | 2754.11 | 2739.60 | 28.463 | 10.3919 |
| 10000 | 100 | 1000000 | 98.980 | 0.2395 | 0.2085 | 0.4143 | 2841.70 | 2826.73 | 28.248 | 9.9970 |

## Escopo metodologico

Inferencia serial, sincrona e batch 1. A entrada ja esta normalizada em float32/[0,1]. A latencia comeca depois de `tf.convert_to_tensor` e termina depois de `.numpy()`, que sincroniza a GPU e materializa os 10 logits no host. O throughput inclui laco Python, criacao do tensor, inferencia, sincronizacao, argmax e armazenamento da predicao.

Nao foram usados XLA, TensorRT, mixed precision, quantizacao, batching maior que 1 ou inferencias concorrentes. Nenhum outlier foi removido. A potencia e da placa GPU, medida pelo sensor NVIDIA a cada 100 ms; energia e uma estimativa integrada, e nao uma medicao na tomada.

Os conjuntos de 100 e 1.000 imagens sao balanceados e aninhados. O conjunto de 10.000 usa toda a divisao oficial do MNIST e, portanto, mantem sua distribuicao real por classe.

LUT, FF, DSP, BRAM, URAM e frequencia atingida da implementacao nao se aplicam a GPU; essas metricas pertencem aos fluxos hls4ml/Vitis AI.

## Evidencias

- `resultados.csv`: agregados completos.
- `passagens_n*.csv`: uma linha por ciclo.
- `latencias_n*.npy`: cada latencia individual.
- `predicoes_n*.csv`: rotulo, predicao e logits da primeira passagem.
- `telemetria_nvidia_smi.csv`: telemetria bruta.
- `metadados.json`: ambiente, arquitetura, hashes e configuracao.
- `nvidia_smi_antes.txt` e `nvidia_smi_depois.txt`: snapshots integrais.
