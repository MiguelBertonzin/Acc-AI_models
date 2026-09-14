# Resultados — LeNet/MNIST sem softmax na GPU

## Resultado principal

O resultado principal usa todas as 10.000 imagens oficiais de teste e 100 ciclos. Os pontos de 10, 20 e 50 ciclos sao prefixos da mesma coleta.

| Metrica | Resultado |
|---|---:|
| GPU | NVIDIA GeForce RTX 3050 OEM |
| Precisao | float32 |
| Batch | 1 |
| Imagens unicas | 100 |
| Ciclos | 2 |
| Inferencias cronometradas | 200 |
| Acertos | 99 |
| Acuracia | 99.0000% |
| IC95 Wilson | [94.5514%; 99.8233%] |
| Latencia media | 0.3367 ms |
| IC95 da media por ciclo | [0.1247; 0.5487] ms |
| Mediana | 0.2164 ms |
| Desvio padrao | 0.2399 ms |
| Coeficiente de variacao | 71.25% |
| p90 / p95 / p99 | 0.7828 / 0.8427 / 1.1205 ms |
| Minimo / maximo | 0.1943 / 1.6224 ms |
| Vazao media | 2501.199 inferencias/s |
| Vazao efetiva global | 2254.151 inferencias/s |
| Potencia total media | 26.900 W |
| Potencia dinamica media | 0.000 W |
| Energia total/inferencia | 15.6840 mJ |
| Energia dinamica/inferencia | 0.0000 mJ |
| Utilizacao GPU media | 1.000% |
| Memoria GPU usada | 707.000 MiB |
| Clock SM medio | 1755.000 MHz |
| Temperatura media / maxima | 42.000 / 42.000 C |
| Amostras nvidia-smi | 1 |
| Potencia ociosa | 26.993 W |

## Convergencia de todas as series

| Imagens | Ciclos | Inferencias | Acuracia (%) | Media (ms) | Mediana (ms) | p95 (ms) | FPS medio | FPS efetivo | Potencia (W) | Energia/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 100 | 99.000 | 0.4449 | 0.2943 | 0.9380 | 1715.12 | 1715.12 | 26.900 | 15.6840 |
| 100 | 2 | 200 | 99.000 | 0.3367 | 0.2164 | 0.8427 | 2501.20 | 2254.15 | 26.900 | 15.6840 |

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
