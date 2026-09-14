# ResNet8 hls4ml — Benchmark final ZCU104 10k × 100

## 1. Objetivo

Esta coleta foi realizada para aumentar a paridade do protocolo experimental
entre a ZCU104 e os benchmarks realizados em CPU e GPU.

Foram utilizadas as 10.000 imagens oficiais do conjunto de teste CIFAR-10
em 100 ciclos completos, totalizando 1.000.000 de inferências.

## 2. Modelo e plataforma

- Modelo: ResNet8
- Dataset: CIFAR-10
- Entrada: 32 × 32 × 3
- Classes: 10
- Saída: 10 logits, sem softmax
- Predição: argmax(logits)
- Batch: 1
- Plataforma: AMD/Xilinx ZCU104
- Dispositivo: XCZU7EV-FFVC1156-2-E
- Clock do acelerador: 100 MHz
- Backend hls4ml: Vitis
- IOType: io_stream
- Strategy: Resource
- ConvImplementation: LineBuffer
- FIFO optimization: habilitada
- Precisão: ap_fixed<22,12,AP_RND_CONV,AP_SAT>

## 3. Validação de acurácia

A validação de acurácia foi executada separadamente do benchmark de desempenho.

- Imagens únicas: 10,000
- Acertos: 7,492
- Acurácia top-1: **74.9200%**
- IC95% Wilson: **[74.0609%; 75.7599%]**

Os tempos obtidos durante a validação funcional não são utilizados como
métricas de latência ou throughput.

Foram preservados:

- `predictions_accuracy_10000.npy`
- `logits_accuracy_10000.npy`
- `confusion_matrix_10000.npy`
- `confusion_matrix_10000.csv`
- `accuracy_10000.json`

## 4. Protocolo de desempenho

- Imagens únicas por ciclo: 10,000
- Ciclos: 100
- Inferências totais: 1,000,000
- Warm-up: 100
- Seed: 20260825
- Batch: 1
- Execução: serial e síncrona
- Outliers removidos: nenhum

As 10.000 imagens foram normalizadas, quantizadas e empacotadas antes
da janela temporizada.

## 5. Fronteira da latência inference-only

A latência individual corresponde exclusivamente ao caminho:

```text
entrada já presente no PynqBuffer
        ↓
DMA MM2S
        ↓
ResNet8 hls4ml
        ↓
DMA S2MM
        ↓
wait()
```

A cópia da entrada pré-empacotada para o PynqBuffer ocorre antes do início
do cronômetro da latência individual.

Também ficam fora da latência individual:

- normalização /255;
- quantização para ponto fixo;
- packing;
- decode dos logits;
- argmax;
- carregamento do dataset;
- inicialização do Overlay.

## 6. Fronteira do throughput efetivo

O throughput efetivo possui fronteira mais ampla que a latência individual.

Cada ciclo medido contém:

```text
entrada previamente empacotada
        ↓
cópia para PynqBuffer
        ↓
DMA MM2S
        ↓
ResNet8 hls4ml
        ↓
DMA S2MM
        ↓
decode dos logits
        ↓
argmax
        ↓
próxima imagem
```

Portanto, o throughput efetivo e a latência inference-only não possuem
exatamente a mesma fronteira experimental.

## 7. Resultados de latência

| Métrica | Resultado |
|---|---:|
| Inferências temporizadas | **1,000,000** |
| Latência média | **0.849300 ms** |
| Mediana | **0.849360 ms** |
| Desvio-padrão | 0.015460 ms |
| Coeficiente de variação | 1.8204% |
| p95 | **0.862090 ms** |
| p99 | **0.875650 ms** |
| Mínimo | 0.825990 ms |
| Máximo | 4.762210 ms |
| Taxa equivalente pela latência média | **1177.4402 FPS** |

O valor máximo é mantido nos dados. Nenhum outlier foi removido.

## 8. Resultados de throughput

| Métrica | Resultado |
|---|---:|
| Média dos FPS por ciclo | **899.2073 FPS** |
| FPS efetivo global | **899.2063 FPS** |
| Desvio-padrão dos ciclos | 0.9882 FPS |
| CV dos ciclos | **0.1099%** |
| IC95% | [899.0136; 899.4010] FPS |
| IC95% bootstrap | [899.0196; 899.4038] FPS |
| Taxa equivalente média do caminho DMA-FPGA-DMA | 1177.4412 FPS |
| Drift primeiros/últimos 20 ciclos | **+0.1496%** |
| Divergências de predição | **0** |

Para comparação principal de throughput entre CPU, GPU e FPGA deve ser
utilizado o **FPS efetivo global**, calculado pela razão entre o número total
de inferências e a soma dos tempos dos 100 ciclos.

## 9. Convergência do experimento

| Ciclos | Inferências | Latência média | p95 | FPS efetivo global |
|---:|---:|---:|---:|---:|
| 10 | 100,000 | 0.849460 ms | 0.862230 ms | 899.230 FPS |
| 20 | 200,000 | 0.849442 ms | 0.862120 ms | 899.136 FPS |
| 50 | 500,000 | 0.849526 ms | 0.862200 ms | 898.905 FPS |
| 100 | 1,000,000 | 0.849300 ms | 0.862090 ms | 899.206 FPS |

Os resultados demonstram alta estabilidade do benchmark conforme cresce o
número de ciclos.

A média dos primeiros 20 ciclos foi de **899.1364 FPS** e a
média dos últimos 20 ciclos foi de **900.4819 FPS**, correspondendo
a um drift de apenas **+0.1496%**.

## 10. Resultado principal

```text
Acurácia            = 74.9200%
Latência média      = 0.849300 ms
Mediana             = 0.849360 ms
p95                 = 0.862090 ms
p99                 = 0.875650 ms
FPS efetivo global  = 899.2063 FPS
Path equivalente    = 1177.4412 FPS
CV entre ciclos     = 0.1099%
Drift               = +0.1496%
DMA/pred mismatches = 0
```

## 11. Arquivos de desempenho

- `latencies_inference_only_10k100.npy`
- `cycle_wall_s_10k100.npy`
- `fps_effective_10k100.npy`
- `fps_path_10k100.npy`
- `cycle_latency_mean_ms_10k100.npy`
- `cycle_mismatches_10k100.npy`
- `cycles_10k100.csv`
- `FINAL_SUMMARY_10k100.json`

## 12. Reprodutibilidade

A pasta também preserva:

- bitstream utilizado;
- arquivo HWH;
- dataset CIFAR-10 utilizado;
- notebook do benchmark;
- `ENVIRONMENT.json`;
- hashes SHA-256.

## 13. Observação sobre CPU/GPU

Esta coleta utiliza o mesmo número de imagens únicas e ciclos do protocolo
CPU/GPU: **10.000 imagens × 100 ciclos**.

Isso não significa que as plataformas possuam uma fronteira física de
medição idêntica. A latência da FPGA é host-observável no caminho síncrono
DMA-FPGA-DMA, enquanto CPU e GPU utilizam seus respectivos runtimes.

As diferenças de fronteira devem permanecer explicitamente documentadas
na comparação entre plataformas.

Nenhum outlier foi removido.
