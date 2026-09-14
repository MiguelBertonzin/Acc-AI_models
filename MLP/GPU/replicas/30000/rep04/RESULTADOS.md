# Resultados — MLP Iris em GPU

## Protocolo

- Modelo: MLP 4–8–8–3 com softmax, 139 parâmetros, `float32`.
- Holdout: 30 amostras balanceadas, reconstruído com `test_size=0.2`, `random_state=42` e `stratify=y`.
- Desempenho: batch 1, serial e síncrono; entrada padronizada antes do cronômetro e saída materializada no host.
- Repetições: 1000 ciclos × 30 amostras = 30000 inferências.
- GPU: NVIDIA GeForce RTX 3050 OEM; driver 580.173.02.

## Validação estatística

| Métrica | Resultado |
|---|---:|
| Acurácia | 29/30 = 96.67% |
| IC95% da acurácia (Wilson) | [83.33%, 99.41%] |
| Acurácia balanceada | 0.966667 |
| F1 macro | 0.966583 |
| MCC | 0.951587 |
| Cohen kappa | 0.950000 |
| Log loss | 0.15358392 |
| Brier multiclasse | 0.08240225 |
| ROC AUC OvR macro | 0.996667 |
| ECE (10 bins) | 0.07649873 |
| Teste binomial exato contra 1/3 | p = 2.963e-13 |

Matriz de confusão (linhas = classe real; colunas = classe predita):

```text
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

O holdout é pequeno. O IC95% de Wilson e os intervalos bootstrap devem acompanhar a estimativa pontual; repetir uma mesma amostra melhora a caracterização temporal, mas não aumenta o tamanho estatístico da avaliação de acurácia.

## Paridade NumPy/CPU × TensorFlow/GPU e determinismo

- Concordância de classes NumPy/CPU × TensorFlow/GPU: 100.00%.
- Maior diferença absoluta de probabilidade NumPy/CPU × TensorFlow/GPU: 1.192e-07.
- `allclose(rtol=1e-5, atol=1e-6)`: True.
- Maior diferença entre execuções repetidas na GPU: 0.000e+00.
- Concordância do `argmax` nas repetições: 100.00%.

## Desempenho e energia

| Métrica | Resultado |
|---|---:|
| Latência média | 0.177197 ms |
| IC95% da latência média | [0.174728, 0.179667] ms |
| Mediana | 0.160485 ms |
| Desvio-padrão | 0.051828 ms |
| p95 / p99 | 0.272111 / 0.403367 ms |
| Mínimo / máximo | 0.138017 / 1.327895 ms |
| Vazão média por ciclo | 3886.44 inferências/s |
| Vazão efetiva global | 3796.08 inferências/s |
| Potência média da placa | 29.253 W |
| Potência ociosa após aquecimento | 29.385 W |
| Potência dinâmica média estimada | 0.000 W |
| Energia total da campanha | 235.937 J |
| Energia dinâmica da campanha | 0.000 J |
| Energia total por inferência | 7.864411 mJ |
| Energia dinâmica por inferência | 0.000000 mJ |
| Utilização GPU média / máxima | 10.66% / 13.00% |
| Utilização de memória média / máxima | 1.00% / 1.00% |
| Memória usada média / máxima | 663.0 / 663.0 MiB |
| Clock SM médio / máximo | 1755.0 / 1755.0 MHz |
| Temperatura média / máxima | 58.0 / 58.0 °C |
| Ventoinha média / máxima | 31.0% / 31.0% |
| P-state observado | P2: 79 |
| Telemetria | 79 amostras em 8.065 s |

A energia integra `power.draw.instant` do sensor da placa. A energia dinâmica subtrai a potência ociosa medida depois do aquecimento. O intervalo nominal da telemetria (100 ms) é muito maior que uma única inferência; a integração é interpretável sobre a campanha inteira, não por amostra isolada. O consumo do host não está incluído. A diferença entre potência ativa e ociosa é muito pequena frente à incerteza aproximada de ±5 W registrada na metodologia; portanto, a energia dinâmica deve ser tratada como estimativa de baixa confiança.

## Evidências

- `validacao_estatistica.json`: métricas, intervalos e métricas por classe.
- `validacao_cpu_gpu.json`: paridade numérica e teste pareado.
- `metricas_desempenho.csv`: prefixos de ciclos e convergência.
- `passagens_gpu.csv`: uma linha por ciclo.
- `latencias_gpu.npy`: todas as latências individuais.
- `predicoes_holdout.csv`: entradas, probabilidades CPU/GPU e classes.
- `telemetria_nvidia_smi.csv`: amostras brutas do sensor.
- `metadados.json`: ambiente, hashes, índices da divisão e configuração.
- `nvidia_smi_*.txt`: snapshots completos antes e depois.
