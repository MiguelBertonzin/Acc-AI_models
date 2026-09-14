# Resultados — ResNet-8 com softmax na GPU

Modelo: `/home/miguel/Downloads/Plano testes TCC/resnet8/resnet8_cifar10_keras3.h5`  
GPU: NVIDIA GeForce RTX 3050 OEM  
Seed: 20260825  
Potência ociosa média após aquecimento: 28.729 W

Cada linha resume os primeiros K ciclos da mesma execução de 100 ciclos. A acurácia e seu IC de Wilson usam somente imagens únicas; repetições não são tratadas como novas amostras de acurácia.

| Imagens | Ciclos | Medições | Acurácia (%) | Latência média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | Vazão média (FPS) | Potência total (W) | Potência dinâmica (W) | Energia/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 1000 | 74.000 | 0.6709 | 0.5772 | 0.2225 | 1.0874 | 1297.008 | 32.325 | 3.614 | 26.5366 |
| 100 | 20 | 2000 | 74.000 | 0.6059 | 0.5469 | 0.1736 | 0.9936 | 1401.922 | 34.304 | 5.585 | 25.5726 |
| 100 | 50 | 5000 | 74.000 | 0.5605 | 0.5220 | 0.1207 | 0.7497 | 1482.445 | 36.338 | 7.614 | 25.1044 |
| 100 | 100 | 10000 | 74.000 | 0.6079 | 0.5282 | 0.1878 | 1.0482 | 1418.742 | 36.788 | 8.061 | 27.7936 |
| 1000 | 10 | 10000 | 74.000 | 0.5756 | 0.5110 | 0.1481 | 0.9669 | 1453.372 | 38.159 | 9.430 | 26.5017 |
| 1000 | 20 | 20000 | 74.000 | 0.6247 | 0.5119 | 0.2078 | 1.0491 | 1391.730 | 37.787 | 9.058 | 27.9934 |
| 1000 | 50 | 50000 | 74.000 | 0.6957 | 0.5376 | 0.2487 | 1.1242 | 1292.570 | 36.362 | 7.633 | 29.3640 |
| 1000 | 100 | 100000 | 74.000 | 0.6952 | 0.5411 | 0.2476 | 1.1166 | 1291.288 | 36.071 | 7.342 | 29.2268 |
| 10000 | 10 | 100000 | 74.890 | 0.5638 | 0.5097 | 0.1559 | 1.0210 | 1487.879 | 36.675 | 7.947 | 24.9230 |
| 10000 | 20 | 200000 | 74.890 | 0.5732 | 0.5089 | 0.1689 | 1.0434 | 1462.172 | 36.553 | 7.825 | 25.2256 |
| 10000 | 50 | 500000 | 74.890 | 0.6232 | 0.5115 | 0.2180 | 1.0846 | 1382.613 | 35.757 | 7.029 | 26.4098 |
| 10000 | 100 | 1000000 | 74.890 | 0.6110 | 0.5104 | 0.2071 | 1.0768 | 1400.912 | 35.918 | 7.190 | 26.1127 |

## Definições

- Latência: tempo do grafo TensorFlow (`tf.function`, sem XLA) até a saída estar disponível no host (`.numpy()`), batch 1. Normalização e seleção da imagem ficam fora do cronômetro.
- Vazão média: média aritmética da vazão de cada ciclo completo.
- Potência total: potência da placa (`power.draw.instant`) amostrada pelo `nvidia-smi`.
- Potência dinâmica: potência total menos a potência ociosa após o aquecimento, limitada a zero.
- Energia por inferência: potência total média vezes o tempo do ciclo, dividida pelo número de imagens.
- A telemetria tem resolução de 100 ms e precisão própria do sensor NVIDIA (aproximadamente ±5 W). Nos ciclos curtos de 100 imagens, potência e energia têm maior incerteza; consulte `power_samples_total` no CSV.
- LUT, FF, DSP, BRAM, URAM e frequência atingida não se aplicam a esta execução em GPU; serão coletadas na implementação FPGA.

Os dados completos estão em `resultados.csv`, as estatísticas de cada ciclo em `passagens_n*.csv` e todas as latências individuais em `latencias_n*.npy`.
