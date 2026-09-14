# Resultados — ResNet-8 sem softmax na GPU

Modelo: `/home/miguel/Downloads/Plano testes TCC/resnet8/resnet8_cifar10_keras3_no_softmax.h5`  
GPU: NVIDIA GeForce RTX 3050 OEM  
Seed: 20260825  
Potência ociosa média após aquecimento: 29.063 W

Cada linha resume os primeiros K ciclos da mesma execução de 100 ciclos. A acurácia e seu IC de Wilson usam somente imagens únicas; repetições não são tratadas como novas amostras de acurácia.

| Imagens | Ciclos | Medições | Acurácia (%) | Latência média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | Vazão média (FPS) | Potência total (W) | Potência dinâmica (W) | Energia/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 1000 | 74.000 | 0.6239 | 0.5458 | 0.2326 | 0.9889 | 1364.484 | 31.586 | 2.592 | 24.6813 |
| 100 | 20 | 2000 | 74.000 | 0.6250 | 0.5428 | 0.2115 | 0.9955 | 1373.194 | 33.789 | 4.761 | 25.8141 |
| 100 | 50 | 5000 | 74.000 | 0.5612 | 0.5093 | 0.1498 | 0.8420 | 1493.420 | 36.057 | 7.010 | 25.0693 |
| 100 | 100 | 10000 | 74.000 | 0.5771 | 0.5100 | 0.1577 | 0.9274 | 1467.267 | 36.775 | 7.720 | 26.4236 |
| 1000 | 10 | 10000 | 74.000 | 0.5725 | 0.5107 | 0.1433 | 0.9687 | 1455.707 | 38.001 | 8.938 | 26.2794 |
| 1000 | 20 | 20000 | 74.000 | 0.6092 | 0.5248 | 0.1878 | 1.0256 | 1395.482 | 37.754 | 8.691 | 27.5923 |
| 1000 | 50 | 50000 | 74.000 | 0.6270 | 0.5111 | 0.2109 | 1.0318 | 1383.381 | 37.531 | 8.468 | 27.9486 |
| 1000 | 100 | 100000 | 74.000 | 0.6628 | 0.5156 | 0.2347 | 1.0710 | 1337.919 | 36.848 | 7.785 | 28.6547 |
| 10000 | 10 | 100000 | 74.890 | 0.6505 | 0.5084 | 0.2336 | 1.0771 | 1326.188 | 36.312 | 7.249 | 27.9193 |
| 10000 | 20 | 200000 | 74.890 | 0.6690 | 0.5085 | 0.2497 | 1.1026 | 1298.840 | 35.837 | 6.774 | 28.2102 |
| 10000 | 50 | 500000 | 74.890 | 0.6829 | 0.5107 | 0.2582 | 1.1180 | 1277.128 | 35.287 | 6.224 | 28.2658 |
| 10000 | 100 | 1000000 | 74.890 | 0.6912 | 0.5122 | 0.2609 | 1.1219 | 1268.493 | 35.182 | 6.119 | 28.4617 |

## Definições

- Latência: tempo do grafo TensorFlow (`tf.function`, sem XLA) até a saída estar disponível no host (`.numpy()`), batch 1. Normalização e seleção da imagem ficam fora do cronômetro.
- Vazão média: média aritmética da vazão de cada ciclo completo.
- Potência total: potência da placa (`power.draw.instant`) amostrada pelo `nvidia-smi`.
- Potência dinâmica: potência total menos a potência ociosa após o aquecimento, limitada a zero.
- Energia por inferência: potência total média vezes o tempo do ciclo, dividida pelo número de imagens.
- A telemetria tem resolução de 100 ms e precisão própria do sensor NVIDIA (aproximadamente ±5 W). Nos ciclos curtos de 100 imagens, potência e energia têm maior incerteza; consulte `power_samples_total` no CSV.
- LUT, FF, DSP, BRAM, URAM e frequência atingida não se aplicam a esta execução em GPU; serão coletadas na implementação FPGA.

Os dados completos estão em `resultados.csv`, as estatísticas de cada ciclo em `passagens_n*.csv` e todas as latências individuais em `latencias_n*.npy`.
