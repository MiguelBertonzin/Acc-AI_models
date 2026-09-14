# ResNet8 hls4ml — Benchmark na AMD/Xilinx ZCU104

## 1. Objetivo

Este diretório contém os resultados finais de desempenho e eficiência energética da implementação da ResNet8 para o conjunto CIFAR-10 utilizando hls4ml na AMD/Xilinx ZCU104.

O benchmark foi dividido em três cenários com fronteiras de medição distintas:

1. **Inference batch 1 síncrono** — principal cenário para comparação com CPU/GPU;
2. **End-to-end host-to-class** — inclui pré-processamento e pós-processamento;
3. **Throughput sustentado do caminho acelerado** — caracteriza a capacidade máxima sustentada DMA + FPGA + DMA.

Os três valores não representam a mesma fronteira experimental e, portanto, não devem ser utilizados de forma intercambiável.

## 2. Plataforma e configuração

- Plataforma: AMD/Xilinx ZCU104
- FPGA/SoC: XCZU7EV-FFVC1156-2-E
- Modelo: ResNet8
- Dataset: CIFAR-10
- Entrada: 32 × 32 × 3
- Classes: 10
- Parâmetros: 78.714
- Saída: 10 logits
- Softmax: removida
- Classificação: `argmax(logits)`
- Batch: 1
- Clock do acelerador: 100 MHz
- Backend hls4ml: Vitis
- IOType: `io_stream`
- Strategy: `Resource`
- ConvImplementation: `LineBuffer`
- FIFO optimization: habilitada
- Reuse Factor máximo: 288
- Precisão: `ap_fixed<22,12,AP_RND_CONV,AP_SAT>`

## 3. Validação de acurácia

A acurácia foi validada separadamente utilizando as 10.000 imagens oficiais do conjunto de teste do CIFAR-10.

- Imagens únicas: **10.000**
- Acertos: **7.492**
- Acurácia top-1: **74,9200%**
- IC95% de Wilson: **[74,0609%; 75,7599%]**

As repetições utilizadas para benchmarking não foram consideradas novas observações de acurácia.

# 4. Cenário 1 — Inference batch 1 síncrono

## 4.1 Fronteira de medição

A imagem encontra-se previamente normalizada, quantizada e empacotada.

O throughput efetivo corresponde ao seguinte fluxo:

```text
entrada pré-processada
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

A latência individual `inference-only` possui uma fronteira menor. O cronômetro começa após a entrada já estar presente no PynqBuffer e cobre:

```text
DMA MM2S → ResNet8 FPGA → DMA S2MM → wait()
```

Foram utilizadas 5.000 imagens estratificadas e 20 ciclos, totalizando **100.000 inferências temporizadas**.

## 4.2 Resultados — Inference batch 1

| Métrica | Resultado |
|---|---:|
| Inferências temporizadas | 100000 |
| Latência média | **0.8451 ms** |
| Mediana | 0.8456 ms |
| Desvio-padrão | 0.0148 ms |
| CV das latências | 1.754% |
| p95 | **0.8592 ms** |
| p99 | **0.8759 ms** |
| Mínimo | 0.8276 ms |
| Máximo | 4.5542 ms |
| Throughput efetivo | **901.08 FPS** |
| Desvio do throughput | 1.38 FPS |
| CV do throughput | 0.153% |
| IC95% throughput | [900.48; 901.69] FPS |
| IC95% bootstrap | [900.48; 901.67] FPS |
| Drift temporal | 0.197% |
| Taxa equivalente pela latência | 1183.23 FPS |
| Potência idle | **10.6094 W** |
| Potência ativa | **12.2283 W** |
| Potência dinâmica | **1.6189 W** |
| Energia total/inferência | **13.4994 mJ** |
| Energia dinâmica/inferência | **1.7871 mJ** |
| FPS durante telemetria | 905.84 FPS |
| Overhead da telemetria | 0.528% |

# 5. Cenário 2 — End-to-end

## 5.1 Fronteira de medição

O cenário end-to-end parte de uma imagem CIFAR-10 `uint8` já presente na memória RAM e termina na classe prevista.

```text
imagem uint8 em RAM
    ↓
conversão float32
    ↓
normalização /255
    ↓
quantização ap_fixed<22,12>
    ↓
packing
    ↓
PynqBuffer
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
classe prevista
```

Não entram no tempo: carregamento do dataset a partir do armazenamento, programação do FPGA, criação do Overlay, alocação inicial dos buffers e warm-up.

## 5.2 Resultados — End-to-end

| Métrica | Resultado |
|---|---:|
| Inferências temporizadas | 100000 |
| Latência média | **1.5977 ms** |
| Mediana | 1.5930 ms |
| Desvio-padrão | 0.0412 ms |
| CV das latências | 2.577% |
| p95 | **1.6155 ms** |
| p99 | **1.6699 ms** |
| Mínimo | 1.5728 ms |
| Máximo | 3.8884 ms |
| Throughput efetivo | **620.70 FPS** |
| Desvio do throughput | 1.11 FPS |
| CV do throughput | 0.178% |
| IC95% throughput | [620.22; 621.19] FPS |
| IC95% bootstrap | [620.23; 621.18] FPS |
| Drift temporal | 0.021% |
| Potência idle | **10.6270 W** |
| Potência ativa | **11.8088 W** |
| Potência dinâmica | **1.1818 W** |
| Energia total/inferência | **19.0992 mJ** |
| Energia dinâmica/inferência | **1.9114 mJ** |
| FPS durante telemetria | 618.29 FPS |
| Overhead da telemetria | -0.389% |

# 6. Cenário 3 — Acelerador saturado

## 6.1 Fronteira de medição

Este cenário caracteriza a capacidade máxima sustentada do caminho acelerado. A entrada já está completamente empacotada e presente no PynqBuffer.

Durante a janela de benchmark é repetido somente:

```text
DMA MM2S → ResNet8 hls4ml → DMA S2MM → wait() → próxima execução
```

Não entram: normalização, quantização, packing, cópia de uma nova imagem, decode dos logits ou `argmax`.

Esse ensaio continua sendo serial e síncrono. Não foram utilizadas várias inferências simultâneas, double buffering ou várias requisições pendentes.

Por esse motivo, o resultado deve ser apresentado como uma métrica adicional da FPGA e não substituir o throughput batch 1 na comparação CPU/GPU.

## 6.2 Resultados — Acelerador saturado

| Métrica | Resultado |
|---|---:|
| Janelas de desempenho | 8 |
| Duração por janela | 10 s |
| Throughput sustentado | **1200.94 FPS** |
| Desvio-padrão | 3.24 FPS |
| CV | 0.270% |
| IC95% | [1198.70; 1203.18] FPS |
| IC95% bootstrap | [1199.09; 1203.24] FPS |
| Drift temporal | 0.049% |
| Potência idle | **10.6170 W** |
| Potência ativa | **12.6580 W** |
| Potência dinâmica | **2.0410 W** |
| Energia total/inferência | **10.7266 mJ** |
| Energia dinâmica/inferência | **1.7295 mJ** |
| FPS durante telemetria | 1180.06 FPS |
| Overhead da telemetria | -1.739% |

# 7. Síntese dos três cenários

| Métrica | Inference batch 1 | End-to-end | Acelerador saturado |
|---|---:|---:|---:|
| Latência média | **0.8451 ms** | **1.5977 ms** | — |
| Mediana | 0.8456 ms | 1.5930 ms | — |
| p95 | 0.8592 ms | 1.6155 ms | — |
| p99 | 0.8759 ms | 1.6699 ms | — |
| Throughput | **901.08 FPS** | **620.70 FPS** | **1200.94 FPS** |
| Potência idle | 10.6094 W | 10.6270 W | 10.6170 W |
| Potência ativa | 12.2283 W | 11.8088 W | 12.6580 W |
| Potência dinâmica | 1.6189 W | 1.1818 W | 2.0410 W |
| Energia total/inf. | 13.4994 mJ | 19.0992 mJ | 10.7266 mJ |
| Energia dinâmica/inf. | 1.7871 mJ | 1.9114 mJ | 1.7295 mJ |

# 8. Metodologia energética

A potência foi medida utilizando o sensor PMBus `12V_power` da ZCU104.

A potência dinâmica foi definida como:

```text
P_dinâmica = P_ativa - P_idle
```

A energia por inferência foi obtida a partir da potência e do throughput observados durante a mesma janela de aquisição:

```text
E_total = P_ativa / FPS
E_dinâmica = P_dinâmica / FPS
```

A frequência final de aquisição utilizada foi de 1 s.

Antes da coleta definitiva foi realizado um ensaio para avaliar o impacto da telemetria sobre o throughput saturado:

| Intervalo PMBus | FPS | Overhead |
|---:|---:|---:|
| 0,1 s | 1104,98 | -7,99% |
| 0,2 s | 1149,77 | -4,26% |
| 0,5 s | 1181,01 | -1,66% |
| 1,0 s | 1191,01 | -0,83% |

A amostragem de 1 s foi selecionada por apresentar baixa perturbação e ainda permitir múltiplas observações durante cada janela de potência.

Na coleta energética definitiva, os impactos medidos foram:

- Inference batch 1: **0.528%**
- End-to-end: **-0.389%**
- Acelerador saturado: **-1.739%**

# 9. Validação estatística

Nenhum outlier foi removido.

Foram preservadas todas as observações de latência. Para avaliar estabilidade temporal, os ciclos ou janelas foram utilizados como unidade estatística.

Foram reportados:

- média;
- mediana;
- desvio-padrão amostral;
- coeficiente de variação (CV);
- p95;
- p99;
- mínimo;
- máximo;
- intervalo de confiança de 95%;
- intervalo de confiança bootstrap;
- drift entre o início e o final do ensaio.

Os coeficientes de variação extremamente baixos entre os ciclos demonstram boa repetibilidade das medições.

# 10. Uso dos resultados na comparação CPU × GPU × FPGA

Para a comparação principal entre plataformas recomenda-se:

### Latência

Utilizar a latência `inference-only`, pois a entrada já está no formato esperado pelo runtime/acelerador antes do início do cronômetro.

Resultado ZCU104: **0.8451 ms**.

### Throughput batch 1

Utilizar o throughput efetivo do cenário `inference batch 1`, pois ele inclui a orquestração host necessária para processar sequencialmente imagens diferentes.

Resultado ZCU104: **901.08 FPS**.

### End-to-end

Utilizar somente em comparação com CPU/GPU quando essas plataformas também forem medidas a partir da imagem `uint8` em RAM, incluindo normalização e pré-processamento.

Resultado ZCU104: **1.5977 ms / 620.70 FPS**.

### Capacidade sustentada do caminho acelerado

O resultado de **1200.94 FPS** caracteriza a capacidade sustentada DMA + FPGA + DMA com a entrada já pronta. Essa métrica deve permanecer separada da comparação principal de throughput batch 1.

# 11. Reprodutibilidade

O diretório do experimento preserva os arquivos brutos e agregados necessários para análise posterior:

- `SUMMARY.json`;
- `POWER_SUMMARY.json`;
- `cycles_inference_only.csv`;
- `cycles_end_to_end.csv`;
- `latencias_inference_only.npy`;
- `latencias_end_to_end.npy`;
- `fps_inference_only_effective.npy`;
- `fps_inference_only_path.npy`;
- `fps_end_to_end.npy`;
- `fps_saturated.npy`;
- `subset_indices.npy`;
- `reference_predictions.npy`;
- arquivos `power_*.csv`;
- amostras brutas `power_*_raw_*.csv`;
- bitstream `.bit`;
- hardware handoff `.hwh`;
- dataset utilizado;
- notebook do benchmark;
- hashes SHA-256 dos artefatos.

