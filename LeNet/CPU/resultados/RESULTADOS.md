# Resultados — LeNet/MNIST sem softmax na CPU

## Resultado principal

| Metrica | Resultado |
|---|---:|
| CPU | 13th Gen Intel(R) Core(TM) i7-13700 |
| Precisao / batch | float32 / 1 |
| Imagens / ciclos / inferencias | 10000 / 100 / 1000000 |
| Acuracia | 98.9800% |
| IC95 Wilson | [98.7634%; 99.1590%] |
| Latencia media | 0.3872 ms |
| IC95 da media por ciclo | [0.3843; 0.3900] ms |
| Mediana / desvio | 0.3423 / 0.1339 ms |
| p90 / p95 / p99 | 0.5552 / 0.6395 / 0.9520 ms |
| Minimo / maximo | 0.2385 / 5.6823 ms |
| Vazao media / efetiva | 2292.729 / 2289.486 inf/s |
| Tempo CPU por inferencia | 1.2689 ms |
| Ocupacao / Bzy_MHz turbostat | 17.602% / 3040.2 MHz |
| Temperatura package media | 67.62 C |
| Potencia package / cores | 39.782 / 30.446 W |
| Potencia dinamica package | 27.215 W |
| Energia package/inferencia | 17.3548 mJ |
| Energia dinamica/inferencia | 11.8660 mJ |
| Cobertura RAPL media | 99.215% |
| Amostras RAPL | 4433 |
| Potencia package ociosa | 12.567 W |

## Convergencia

| Imagens | Ciclos | Inferencias | Acuracia | Media (ms) | Mediana | p95 | FPS efetivo | Pkg W | Energia mJ/inf. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 1000 | 99.000% | 0.3971 | 0.3371 | 0.7082 | 2232.07 | 38.962 | 16.6432 |
| 100 | 20 | 2000 | 99.000% | 0.4034 | 0.3440 | 0.6851 | 2195.53 | 37.390 | 16.5496 |
| 100 | 50 | 5000 | 99.000% | 0.3873 | 0.3395 | 0.6603 | 2286.01 | 39.208 | 16.8430 |
| 100 | 100 | 10000 | 99.000% | 0.3816 | 0.3378 | 0.6347 | 2320.76 | 39.528 | 16.8188 |
| 1000 | 10 | 10000 | 98.800% | 0.3830 | 0.3367 | 0.6442 | 2312.53 | 37.971 | 16.3135 |
| 1000 | 20 | 20000 | 98.800% | 0.3853 | 0.3394 | 0.6395 | 2298.73 | 37.512 | 16.2399 |
| 1000 | 50 | 50000 | 98.800% | 0.3802 | 0.3370 | 0.6316 | 2330.41 | 38.093 | 16.2650 |
| 1000 | 100 | 100000 | 98.800% | 0.3813 | 0.3382 | 0.6330 | 2323.75 | 38.137 | 16.3369 |
| 10000 | 10 | 100000 | 98.980% | 0.3790 | 0.3371 | 0.6263 | 2337.86 | 39.224 | 16.7580 |
| 10000 | 20 | 200000 | 98.980% | 0.3816 | 0.3388 | 0.6303 | 2321.49 | 39.259 | 16.8974 |
| 10000 | 50 | 500000 | 98.980% | 0.3824 | 0.3392 | 0.6332 | 2318.31 | 39.999 | 17.2351 |
| 10000 | 100 | 1000000 | 98.980% | 0.3872 | 0.3423 | 0.6395 | 2289.49 | 39.782 | 17.3548 |

## Metodologia

Inferencia float32, batch 1, serial e sincrona, com entrada normalizada antes do cronometro. TensorFlow foi restrito a `/CPU:0`; GPU e soft placement foram desabilitados. O grafo usa `tf.function`, assinatura fixa e `jit_compile=False`.

O turbostat root amostrou a cada 100 ms `Busy%`, `Bzy_MHz`, `PkgTmp`, `PkgWatt` e `CorWatt`. Cada leitura representa o intervalo anterior de 100 ms. A energia foi integrada pela sobreposicao exata entre esse intervalo e cada ciclo. `PkgWatt` ja inclui `CorWatt`; ambos nao foram somados.

Potencia e energia RAPL representam o pacote do processador, nao a tomada. O escopo fisico difere da potencia da placa GPU. Nenhum outlier foi removido.

Arquivos brutos, ciclos, latencias, predicoes, telemetria e metadados estao preservados neste diretorio.
