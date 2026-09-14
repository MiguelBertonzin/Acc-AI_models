# ResNet8 hls4ml — Latência do acelerador saturado

## 1. Objetivo

Este documento registra exclusivamente a caracterização de latência do caminho acelerado da ResNet8/CIFAR-10 implementada com hls4ml na AMD/Xilinx ZCU104.

O ensaio mede o tempo de uma inferência com a entrada já previamente preparada e presente no `PynqBuffer`, mantendo execução **batch 1, serial e síncrona**.

## 2. Fronteira de medição

O intervalo temporizado corresponde a:

```text
DMA MM2S
    ↓
ResNet8 hls4ml
    ↓
DMA S2MM
    ↓
wait()
```

Não entram nesta medição:

- normalização da imagem;
- quantização;
- packing;
- cópia de uma nova imagem para o `PynqBuffer`;
- decode dos logits;
- `argmax`;
- leitura do dataset;
- carregamento do bitstream;
- inicialização do `Overlay`.

Não foram utilizadas múltiplas inferências simultâneas, double buffering ou várias requisições pendentes.

## 3. Configuração

- Plataforma: AMD/Xilinx ZCU104
- Dispositivo: XCZU7EV-FFVC1156-2-E
- Modelo: ResNet8 / CIFAR-10
- Batch: 1
- Clock do acelerador: 100 MHz
- Backend hls4ml: Vitis
- IOType: `io_stream`
- Strategy: `Resource`
- ConvImplementation: `LineBuffer`
- FIFO optimization: habilitada
- Reuse Factor máximo: 288
- Precisão: `ap_fixed<22,12,AP_RND_CONV,AP_SAT>`
- Warm-up: 100 inferências
- Outliers removidos: nenhum
- Inferências temporizadas: 50.000

## 4. Resultados de latência

| Métrica | Resultado |
|---|---:|
| Inferências temporizadas | **50.000** |
| Latência média | **0,832636 ms** |
| Mediana | **0,833510 ms** |
| Desvio-padrão amostral | **0,019895 ms** |
| Coeficiente de variação | **2,389%** |
| p95 | **0,845190 ms** |
| p99 | **0,850360 ms** |
| Mínimo | **0,818670 ms** |
| Máximo | **4,656440 ms** |
| Taxa equivalente pela latência média | **1201,01 FPS** |

## 5. Interpretação

A mediana de **0,833510 ms** está muito próxima da média de **0,832636 ms**, indicando que o comportamento típico das inferências se concentra em torno de aproximadamente 0,83 ms.

O p95 de **0,845190 ms** indica que 95% das inferências apresentaram latência menor ou igual a esse valor. O p99 de **0,850360 ms** mostra que 99% das inferências permaneceram abaixo de aproximadamente 0,85 ms.

O maior valor observado foi de **4,656440 ms**. Esse pico foi mantido nos dados, sem remoção de outliers. Como p95 e p99 permanecem muito próximos da mediana, esse valor máximo representa uma ocorrência rara de jitter e não o comportamento típico do caminho acelerado.

## 6. Relação com o throughput sustentado

A taxa equivalente derivada da latência média é:

```text
1000 / 0,832636 ms = 1201,01 FPS
```

Em um ensaio independente de throughput sustentado do mesmo caminho acelerado foi obtido:

```text
1200,94 FPS
```

Portanto:

```text
Taxa equivalente pela latência = 1201,01 FPS
Throughput sustentado medido   = 1200,94 FPS
```

A diferença entre as duas grandezas é de aproximadamente **0,005%**, fornecendo uma forte validação cruzada da consistência das medições.

A taxa equivalente pela latência não deve ser confundida com uma medição independente de throughput. O valor de **1200,94 FPS** foi medido diretamente em oito janelas independentes de 10 s.

## 7. Diferença para o cenário inference batch 1

No cenário saturado:

```text
mesmo input já presente no PynqBuffer
→ DMA
→ FPGA
→ DMA
→ wait()
→ repetir
```

No cenário `inference batch 1`, imagens diferentes são processadas sequencialmente e existe trabalho do host entre inferências, incluindo a movimentação da próxima entrada para o `PynqBuffer`, decode dos logits e `argmax`.

Por isso, a latência saturada caracteriza o caminho acelerado sob mínima interferência do processamento da próxima entrada e deve ser apresentada como métrica adicional da implementação FPGA, não como substituta do throughput efetivo batch 1 ou do cenário end-to-end.

## 8. Arquivos de evidência

Os dados deste ensaio estão preservados em:

- `latencias_saturated.npy` — contém as 50.000 latências individuais, em milissegundos;
- `SATURATED_LATENCY_SUMMARY.json` — contém as estatísticas agregadas;
- `README-SATURADO.md` — este documento.

O arquivo `latencias_saturated.npy` deve ser considerado a fonte bruta principal para reconstrução das estatísticas.

Nenhum outlier foi removido dos resultados.
