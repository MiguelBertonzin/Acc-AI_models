# MLP Iris em FPGA com hls4ml — Benchmark completo na AMD/Xilinx ZCU104

## 1. Objetivo deste documento

Este documento registra de forma completa a implementação, validação funcional, metodologia de benchmark, medições de desempenho, potência, energia, telemetria, dificuldades encontradas, correções metodológicas e resultados finais da implementação de uma MLP para classificação do conjunto Iris utilizando **hls4ml** em uma **AMD/Xilinx ZCU104**.

A intenção é que este README funcione como:

- documentação técnica do experimento;
- registro de reprodutibilidade;
- fonte dos valores utilizados no TCC;
- referência para comparação posterior com Vitis AI, CPU, GPU e outras implementações;
- explicação das diferenças entre latência física do kernel, latência observável pelo software, throughput efetivo, throughput saturado e end-to-end;
- histórico dos problemas encontrados durante as medições e de como eles foram resolvidos.

Nenhum valor de resultado final foi obtido por remoção seletiva de outliers. As observações foram preservadas e, quando uma coleta foi considerada apenas calibração metodológica, ela foi explicitamente identificada e excluída das estatísticas finais por motivo experimental documentado, e não por apresentar um valor desfavorável.

---

# 2. Identificação do experimento

## 2.1 Data real

Data real da campanha experimental:

**10 de setembro de 2026**

O relógio da ZCU104 estava incorreto durante a execução e gerou o identificador:

```text
run_20250504_150558
```

Portanto:

```text
RUN_ID registrado pela placa = 20250504_150558
Data real do experimento      = 2026-09-10
```

Esse erro de relógio foi preservado no nome do diretório para não alterar os caminhos já gerados. O arquivo `METADATA.json` foi atualizado para registrar explicitamente:

```text
board_clock_was_incorrect = true
actual_experiment_date    = 2026-09-10
```

## 2.2 Diretório principal na ZCU104

```text
/home/xilinx/jupyter_notebooks/mlp_iris/results_zcu104/benchmark_complete/run_20250504_150558
```

Arquivos consolidados finais:

```text
FINAL_RESULTS.json
FINAL_RESULTS_TABLE.csv
THROUGHPUT_FINAL.json
ENERGY_FINAL_AGGREGATE.json
TELEMETRY_VALIDATION.json
METADATA.json
```

Além destes, o diretório contém as subpastas e arquivos brutos de desempenho, potência, energia, telemetria, validações e campanhas intermediárias.

---

# 3. Plataforma

## 3.1 Hardware

- Placa: **AMD/Xilinx ZCU104 Rev. C**
- SoC/FPGA: **Zynq UltraScale+ XCZU7EV-FFVC1156-2-E**
- Processador no PS: ARM Cortex-A53
- Clock observado do ARM durante a campanha: **1.199.999 kHz ≈ 1,2 GHz**
- Governor CPUFreq: **userspace**
- Frequência do ARM permaneceu fixa antes e depois das cargas testadas
- Clock do acelerador no PL: **≈ 99,999 MHz**
- Sensor físico de potência utilizado: **PMBus `12V_power`**
- Temperatura disponível via hwmon: **`irps5401:temp1`**
- A temperatura acima deve ser interpretada como temperatura reportada pelo dispositivo/regulador PMBus, e não como temperatura do die da FPGA.

## 3.2 Software

- Sistema: PynqLinux / Linux
- PYNQ: **3.1.1**
- Python: **3.10.4**
- hls4ml: **1.3.0**
- Vitis HLS: **2024.2**
- Backend hls4ml: Vitis
- Notebook executado no ambiente Jupyter da ZCU104

---

# 4. Modelo de rede neural

## 4.1 Dataset

Dataset:

**Iris**

Número de classes:

**3**

Número de atributos de entrada:

**4**

A validação final em hardware utilizou um holdout com:

**30 amostras únicas**

As repetições utilizadas em testes de integridade e benchmark não foram consideradas novas observações independentes de acurácia.

## 4.2 Arquitetura

Arquitetura da MLP:

```text
4 entradas
↓
Dense(8)
↓
ReLU
↓
Dense(8)
↓
ReLU
↓
Dense(3)
↓
Softmax
```

Número total de parâmetros:

**139**

Nesta implementação a **Softmax foi mantida dentro do hardware**, diferentemente de outras redes do TCC em que a saída foi mantida como logits.

## 4.3 Precisão numérica

Precisão principal:

```text
ap_fixed<16,6>
```

Portanto:

```text
largura total     = 16 bits
bits inteiros     = 6
bits fracionários = 10
escala            = 2^10 = 1024
```

Comportamento utilizado:

```text
AP_TRN
AP_WRAP
```

Softmax com tabela de consulta utilizando precisão específica equivalente à configuração gerada pelo hls4ml, incluindo uso de `ap_fixed<18,8>` na implementação correspondente.

## 4.4 Configuração hls4ml

```text
Strategy    = Latency
ReuseFactor = 1
IOType      = io_parallel
Clock       ≈ 100 MHz
```

---

# 5. Arquivos principais da implementação

Overlay da ZCU104:

```text
/home/xilinx/jupyter_notebooks/mlp_iris/overlay/mlp_iris_zcu104.bit
/home/xilinx/jupyter_notebooks/mlp_iris/overlay/mlp_iris_zcu104.hwh
```

Golden/reference:

```text
/home/xilinx/jupyter_notebooks/mlp_iris/golden/iris_holdout_hls_apfixed16_6.npz
```

Hashes SHA-256 registrados:

```text
BIT:
289d1e1d63e383cb379cc7fd30af548ebcf80741b7a79cbbe7d6ae3f58458e46

HWH:
2b7b7619cca9de08627fcda18956cefbbc93417a37519a09fe04ace6906b492c

GOLDEN NPZ:
b6464c106fc58f6d949c244da569501c9409dbf084438cb18cfac90a3dc75143
```

Esses hashes devem ser preservados junto com os resultados para garantir que os dados foram obtidos com exatamente os mesmos artefatos.

---

# 6. Interface física e wrapper AXI4-Lite

## 6.1 Interface do núcleo

Portas principais do núcleo hls4ml:

```text
ap_clk
ap_rst
ap_start
ap_done
ap_idle
ap_ready

features[63:0]
features_ap_vld

3 saídas de 16 bits
valid de saída
```

## 6.2 Packing de entrada

Os quatro valores de entrada `ap_fixed<16,6>` foram empacotados em um registrador de 64 bits:

```text
x0 → bits 15:0
x1 → bits 31:16
x2 → bits 47:32
x3 → bits 63:48
```

## 6.3 Mapa de registradores AXI4-Lite

```text
0x00  controle/status
0x10  features low
0x14  features high
0x20  output 0
0x24  output 1
0x28  output 2
0x2C  output valid
0x30  last cycles
0x34  invocation counter
```

No registrador de controle:

```text
write bit 0 → start
write bit 1 → clear done

read bit 1  → done_sticky
read bit 2  → idle
read bit 3  → ready
read bit 4  → busy
```

Endereço base observado:

```text
0xA0000000
```

Range:

```text
0x1000
```

---

# 7. Validação funcional antes do benchmark

## 7.1 Keras vs. HLS

Acurácia Keras no holdout:

```text
29 / 30
96,6667 %
```

Acurácia HLS no mesmo holdout:

```text
29 / 30
96,6667 %
```

Predições de classe Keras vs. HLS:

```text
30 / 30 iguais
```

## 7.2 RTL Co-simulation

A co-simulação RTL foi concluída com sucesso.

## 7.3 Hardware físico vs. referência HLS

Na ZCU104:

```text
saídas RAW exatas vs. HLS = 30 / 30
classes iguais vs. HLS    = 30 / 30
valid correto              = 30 / 30
ciclos físicos             = 6 em todas as amostras
```

Acurácia física:

```text
29 / 30 = 96,6667 %
```

A única classificação incorreta do holdout foi:

```text
amostra 25
classe real     = 1
classe prevista = 2
```

Matriz de confusão:

```text
[[10, 0, 0],
 [ 0, 9, 1],
 [ 0, 0,10]]
```

Métricas adicionais:

```text
Macro F1 = 0.966583
MCC      = 0.951587
Kappa    = 0.950000
```

IC95% de Wilson para a acurácia:

```text
[83,33 % ; 99,41 %]
```

## 7.4 Determinismo

Foram executadas **3000 inferências físicas adicionais** para teste de integridade.

Resultado:

```text
mismatches RAW     = 0
mismatches de classe = 0
erros de valid      = 0
execuções com ciclos != 6 = 0
```

Essas 3000 execuções NÃO foram usadas como 3000 novas observações de acurácia. Elas são apenas uma verificação de determinismo e integridade do hardware.

---

# 8. Latência física do kernel

A estimativa original do HLS foi:

```text
5 ciclos
≈ 50 ns @ 100 MHz
```

Entretanto, o contador implementado fisicamente no wrapper mostrou de forma determinística:

```text
6 ciclos
```

Com FCLK0 ≈ 99,999 MHz:

```text
Tclock ≈ 10 ns

Latência física:
6 × 10 ns ≈ 60 ns
```

Portanto, o valor adotado como referência física do kernel é:

**6 ciclos ≈ 60 ns**

Esse valor NÃO deve ser confundido com a latência observada pelo software.

---

# 9. Fronteiras de medição de latência e throughput

Foram mantidas fronteiras distintas.

## 9.1 Latência software-visible do acelerador

A função de medição cobriu aproximadamente:

```text
clear done
→ start
→ polling do status
→ leitura das três saídas
```

Os registradores de entrada já estavam preparados fora da fronteira menor de latência do acelerador.

Essa latência inclui principalmente overhead de acesso AXI-Lite/MMIO, controle, polling e Python.

Ela NÃO representa somente o tempo interno da MLP.

## 9.2 Inference batch-1

Fronteira conceitual principal:

```text
entrada já normalizada/preparada
→ conversão/packing previamente disponível conforme função usada
→ escrita dos registradores de entrada
→ start
→ polling
→ leitura de saída
→ decode fixed-point
→ argmax
→ próxima amostra
```

Batch:

```text
1
```

Execução:

```text
serial
síncrona
uma inferência por vez
```

Não houve múltiplas requisições pendentes, batching simultâneo ou paralelismo de várias inferências.

## 9.3 End-to-end utilizado neste experimento

O dataset golden disponível na ZCU104 contém os atributos já normalizados.

Consequentemente, o cenário end-to-end deste experimento é:

```text
features normalizadas em float
→ conversão para ap_fixed<16,6>
→ packing
→ escrita AXI
→ acelerador
→ leitura de saída
→ conversão fixed-point
→ argmax
→ classe prevista
```

Logo, ele deve ser denominado:

**normalized-float-to-class**

Ele NÃO inclui:

```text
carregamento do dataset do disco
StandardScaler partindo dos valores Iris brutos
programação do FPGA
criação do Overlay
alocação inicial
```

Essa diferença deve ser explicitada ao comparar com outras plataformas.

## 9.4 Saturated

O caminho saturado reutiliza uma entrada já carregada.

Fronteira aproximada:

```text
clear/start
→ polling
→ leitura dos outputs
→ próxima execução
```

Sem nova preparação de entrada a cada inferência.

Ele caracteriza a capacidade sustentada do caminho de controle/acelerador sob mínima preparação de entrada pelo host.

Não deve substituir o throughput batch-1 na comparação principal.

---

# 10. Metodologia estatística

Warm-up:

```text
200 inferências
```

Nenhum outlier foi removido.

Estatísticas calculadas quando aplicáveis:

```text
média
mediana
desvio-padrão amostral
coeficiente de variação
mínimo
máximo
p90
p95
p99
IC95% t de Student
IC95% bootstrap
drift temporal
```

Para inferência estatística, a unidade independente foi a campanha ou janela, e não cada amostra individual de latência ou PMBus.

Semente usada em procedimentos de bootstrap:

```text
20260831
```

---

# 11. Primeira campanha de desempenho com instrumentação individual

Antes da campanha final de throughput foram executadas campanhas extensas que coletavam também latências por inferência.

Esses dados continuam válidos como caracterização de latência, porém o throughput obtido nessa versão ficou penalizado pelo custo de:

```text
perf_counter/perf_counter_ns por inferência
armazenamento dos tempos individuais
construção de estruturas estatísticas
orquestração Python adicional
```

Por esse motivo, esses valores NÃO são os throughputs principais finais.

## 11.1 Batch-1 — 100.000 inferências × 5

Throughput por campanha:

```text
6484.224747
6344.998866
6446.769383
6420.917084
6438.824679
inf/s
```

Resumo:

```text
média   = 6427.146952 inf/s
mediana = 6438.824679 inf/s
DP      = 51.3965 inf/s
CV      = 0.7997 %
IC95% t = [6363.3298 ; 6490.9641] inf/s
bootstrap = [6382.5292 ; 6462.4832] inf/s
drift ≈ -0.7002 %
```

## 11.2 Batch-1 — 30.000 inferências × 5

Throughputs:

```text
6419.5277
6485.0830
6468.9860
6498.0112
6384.1689
inf/s
```

Resumo:

```text
média ≈ 6451.1554 inf/s
CV ≈ 0.741 %
IC95% t ≈ [6391.763 ; 6510.548]
bootstrap ≈ [6411.423 ; 6487.035]
```

## 11.3 E2E — 100.000 inferências × 5

Throughputs:

```text
4069.870465
4069.953781
4074.802891
4069.476439
4053.749677
inf/s
```

Resumo:

```text
média   = 4067.570651 inf/s
mediana = 4069.870465 inf/s
DP      = 8.0300 inf/s
CV      = 0.1974 %
IC95% t = [4057.6001 ; 4077.5412]
bootstrap = [4060.1980 ; 4072.7511]
drift ≈ -0.3961 %
```

Esses valores foram preservados, mas substituídos como valores principais de throughput pela campanha sem instrumentação individual descrita mais adiante.

---

# 12. Latências finais

## 12.1 Inference-only software-visible

Número de observações:

```text
500.000
```

Resultados:

```text
média    = 0.052496706 ms
mediana  = 0.052310 ms
DP       = 0.00188430 ms
CV       = 3.589 %
mínimo   = 0.050890 ms
máximo   = 0.203800 ms
p90      = 0.052750 ms
p95      = 0.052910 ms
p99      = 0.063580 ms
```

Taxa equivalente calculada por:

```text
1000 / latência média em ms
```

Resultado:

```text
≈ 19048.814 inf/s
```

IMPORTANTE:

**19048.814 inf/s NÃO é throughput medido.**

É somente a taxa matemática equivalente ao inverso da latência média.

## 12.2 End-to-end normalized-float-to-class

Número de observações:

```text
500.000
```

Resultados:

```text
média    = 0.224716096 ms
mediana  = 0.221350 ms
DP       = 0.009931385 ms
CV       = 4.4195 %
mínimo   = 0.217740 ms
máximo   = 1.656380 ms
p90      = 0.244100 ms
p95      = 0.245390 ms
p99      = 0.248820 ms
```

Taxa equivalente:

```text
≈ 4450.060 inf/s
```

Esse valor também NÃO é throughput medido.

## 12.3 Saturated software-visible

Número de observações:

```text
50.000
```

Resultados:

```text
média    = 0.054168530 ms
mediana  = 0.053930 ms
DP       = 0.006200518 ms
CV       = 11.4467 %
mínimo   = 0.052920 ms
máximo   = 0.945520 ms
p90      = 0.054300 ms
p95      = 0.054460 ms
p99      = 0.0636201 ms
```

Taxa equivalente:

```text
≈ 18460.904 inf/s
```

Referência do kernel:

```text
6 ciclos
≈ 60 ns
```

Nenhuma observação de latência foi removida, incluindo os máximos observados.

---

# 13. Throughput final principal

Após identificar que o timing individual alterava o throughput, foram executadas novas campanhas sem `perf_counter` por inferência.

Apenas um cronômetro externo mede o tempo total da campanha:

```text
throughput = número de inferências / duração total da campanha
```

Sem telemetria.

## 13.1 Inference batch-1 final

Configuração:

```text
5 campanhas independentes
100.000 inferências por campanha
500.000 inferências totais
batch = 1
serial e síncrono
sem timing individual
sem telemetria
```

Resultados individuais:

| Repetição | Throughput (inf/s) |
|---:|---:|
| 1 | 7220.226720 |
| 2 | 7246.808921 |
| 3 | 7268.591 |
| 4 | 7288.691861 |
| 5 | 7233.099 |

Resumo:

```text
média    = 7251.483517 inf/s
mediana  = 7246.808921 inf/s
DP       = 27.449862 inf/s
CV       = 0.378541 %
mínimo   = 7220.226720
máximo   = 7288.691861
p90      = 7280.651665
p95      = 7284.671763
p99      = 7287.887842

IC95% t:
[7217.400013 ; 7285.567022] inf/s

IC95% bootstrap:
[7230.691958 ; 7273.553134] inf/s
```

CPU média durante essas campanhas:

```text
25.8371 %
```

Mismatches:

```text
0
```

Contadores de invocação:

```text
todos corretos
```

## 13.2 End-to-end final

Configuração:

```text
5 campanhas independentes
100.000 inferências por campanha
500.000 inferências totais
sem timing individual
sem telemetria
```

Resultados individuais:

| Repetição | Throughput (inf/s) |
|---:|---:|
| 1 | 4568.051 |
| 2 | 4549.175 |
| 3 | 4570.783 |
| 4 | 4557.923805 |
| 5 | 4544.192520 |

Resumo:

```text
média    = 4558.025024 inf/s
mediana  = 4557.923805 inf/s
DP       = 11.542900 inf/s
CV       = 0.253243 %
mínimo   = 4544.192520
máximo   = 4570.783237
p90      = 4569.690160
p95      = 4570.236699
p99      = 4570.673930

IC95% t:
[4543.692622 ; 4572.357427] inf/s

IC95% bootstrap:
[4548.931775 ; 4567.118274] inf/s
```

CPU média:

```text
25.6351 %
```

Mismatches:

```text
0
```

Contadores de invocação:

```text
todos corretos
```

---

# 14. Throughput saturado

## 14.1 Bloco original

Foram executadas 8 janelas independentes de 10 s.

Resultados:

```text
17583.163
17602.723
17520.489
17578.163
17587.241
17611.259
17535.345
17573.521
inf/s
```

Resumo:

```text
média = 17573.988132 inf/s
mediana = 17580.663388 inf/s
DP = 31.276892 inf/s
CV = 0.177973 %
IC95% t = [17547.839999 ; 17600.136265]
bootstrap = [17551.869636 ; 17592.331622]
drift = -0.05484 %
```

## 14.2 Por que foi executado um segundo bloco

Durante testes posteriores curtos, foram observados momentaneamente valores na faixa de aproximadamente 18,3–18,5 k inf/s.

Foi levantada a hipótese de alteração de frequência do ARM/DVFS.

O CPUFreq foi então verificado:

```text
governor = userspace
scaling_cur_freq = 1199999 kHz
cpuinfo_cur_freq = 1199999 kHz
```

Antes e depois de carga de 5 s:

```text
frequência = 1199999 kHz
```

Portanto, DVFS foi descartado como causa.

Para evitar escolher seletivamente o maior valor, foi executado um segundo bloco completo de 8 × 10 s.

## 14.3 Bloco de confirmação

Resultados:

```text
17260.130
17250.706
17413.962
17499.053
17521.922
17517.578
17525.051
17541.994
inf/s
```

Resumo:

```text
média    = 17441.299515 inf/s
mediana  = 17508.315436 inf/s
DP       = 121.127529 inf/s
CV       = 0.694487 %
mínimo   = 17250.706104
máximo   = 17541.993790

IC95% t:
[17340.034378 ; 17542.564653]

bootstrap:
[17358.278424 ; 17513.293181]

drift = +1.6330 %
```

Diferença do segundo bloco para o primeiro:

```text
-0.755028 %
```

## 14.4 Resultado saturado final agrupado

Nenhum dos dois blocos foi descartado.

Foram agrupadas as 16 janelas:

```text
N = 16 janelas
10 s por janela
```

Resultado:

```text
média    = 17507.643824 inf/s
mediana  = 17530.197864 inf/s
DP       = 109.537187 inf/s
CV       = 0.625654 %
mínimo   = 17250.706104
máximo   = 17611.259195
p90      = 17594.981772
p95      = 17604.856908
p99      = 17609.978737

IC95% t:
[17449.287887 ; 17565.999760]

bootstrap:
[17449.996907 ; 17555.663081]
```

Esse é o **throughput saturado final recomendado**.

---

# 15. Medição de potência e energia

## 15.1 Sensor

Foi usado diretamente:

```python
get_rails()["12V"].power
```

Sensor:

```text
12V_power
```

Esse sensor representa potência da alimentação de 12 V da placa e, portanto, caracteriza consumo físico de placa, e não apenas do pequeno kernel MLP.

Não foi reconstruída potência manualmente a partir de campos de tensão e corrente.

## 15.2 Equações

Potência dinâmica:

```text
P_dinâmica = P_ativa - P_idle
```

Energia total por inferência:

```text
E_total = P_ativa / throughput_da_mesma_janela
```

Convertida para mJ:

```text
E_total_mJ = P_ativa / FPS × 1000
```

Energia dinâmica:

```text
E_dinâmica_mJ = P_dinâmica / FPS × 1000
```

O throughput usado para energia é obrigatoriamente o throughput medido **na mesma janela energética**.

Não é correto recalcular energia usando o throughput principal obtido em outro momento.

## 15.3 Protocolo final

Para cada cenário:

```text
5 repetições independentes
5 s de idle antes de cada janela
30 s de carga
PMBus nominalmente a 1 Hz
200 warm-ups antes da coleta
```

Total:

```text
15 janelas energéticas finais
```

Validação:

```text
15 / 15 janelas válidas
30 amostras ativas em TODAS
5 amostras idle em TODAS
0 mismatches
todos os invocation counters corretos
```

---

# 16. Resultados energéticos finais

## 16.1 Inference batch-1

Throughput observado nas próprias janelas de energia:

```text
média = 7168.962639 inf/s
IC95% t = [7135.754695 ; 7202.170583]
```

Potência idle:

```text
média = 9.954800 W
IC95% t = [9.940101 ; 9.969499] W
```

Potência ativa:

```text
média = 10.152207 W
IC95% t = [10.147368 ; 10.157045] W
```

Potência dinâmica:

```text
média = 0.197407 W
IC95% t = [0.182794 ; 0.212020] W
```

Energia total:

```text
média = 1.416150 mJ/inf
DP = 0.005661 mJ/inf
CV = 0.399745 %
IC95% t = [1.409121 ; 1.423179] mJ/inf
bootstrap = [1.411652 ; 1.420648]
```

Energia dinâmica:

```text
média = 0.0275396 mJ/inf
DP = 0.0017037
CV = 6.18647 %
IC95% t = [0.0254241 ; 0.0296551]
bootstrap = [0.0260518 ; 0.0286199]
```

Verificação por integração trapezoidal:

```text
E total integrada média = 1.415720 mJ/inf
E dinâmica integrada média = 0.0271180 mJ/inf
```

CPU média durante energia:

```text
25.5096 %
```

Intervalo de amostragem médio:

```text
0.9999972904 s
```

Temperatura `irps5401:temp1`:

```text
≈ 36.0133 °C
```

## 16.2 End-to-end

Throughput durante energia:

```text
média = 4519.831493 inf/s
IC95% t = [4507.589982 ; 4532.073004]
```

Potência idle:

```text
média = 9.951320 W
IC95% t = [9.947823 ; 9.954817]
```

Potência ativa:

```text
média = 10.153007 W
IC95% t = [10.147186 ; 10.158828]
```

Potência dinâmica:

```text
média = 0.201687 W
IC95% t = [0.196217 ; 0.207157]
```

Energia total:

```text
média = 2.246334 mJ/inf
DP = 0.005722
CV = 0.254747 %
IC95% t = [2.239229 ; 2.253439]
bootstrap = [2.242071 ; 2.250986]
```

Energia dinâmica:

```text
média = 0.0446244 mJ/inf
DP = 0.001068
CV = 2.39358 %
IC95% t = [0.0432981 ; 0.0459506]
bootstrap = [0.0438094 ; 0.0454394]
```

Verificação por integração:

```text
E total integrada média = 2.245661 mJ/inf
E dinâmica integrada média = 0.0439579 mJ/inf
```

CPU média:

```text
25.6406 %
```

Intervalo de amostragem médio:

```text
1.0000021503 s
```

Temperatura:

```text
≈ 36.0333 °C
```

## 16.3 Saturated

Throughput observado durante as janelas energéticas:

```text
média = 18337.802595 inf/s
```

Esse valor NÃO substitui o throughput de desempenho agrupado de 17507.643824 inf/s.

Ele deve ser usado para o cálculo energético porque foi medido simultaneamente à potência.

Potência idle:

```text
média = 9.954320 W
IC95% t = [9.943350 ; 9.965290]
```

Potência ativa:

```text
média = 10.132727 W
IC95% t = [10.128135 ; 10.137318]
```

Potência dinâmica:

```text
média = 0.178407 W
IC95% t = [0.166090 ; 0.190724]
```

Energia total:

```text
média = 0.552614 mJ/inf
DP = 0.006190
CV = 1.12006 %
IC95% t = [0.544929 ; 0.560299]
bootstrap = [0.547879 ; 0.557393]
```

Energia dinâmica:

```text
média = 0.00973458 mJ/inf
DP = 0.00064794
CV = 6.65605 %
IC95% t = [0.00893005 ; 0.01053910]
bootstrap = [0.00924862 ; 0.01022859]
```

Verificação por integração:

```text
E total integrada média = 0.552472 mJ/inf
E dinâmica integrada média = 0.00959358 mJ/inf
```

CPU média:

```text
25.2927 %
```

Intervalo médio de amostragem:

```text
1.0000004933 s
```

Temperatura:

```text
≈ 36.1333 °C
```

---

# 17. Resumo principal dos resultados

| Métrica | Inference batch-1 | End-to-end | Saturated |
|---|---:|---:|---:|
| Throughput principal (inf/s) | **7251.484** | **4558.025** | **17507.644** |
| IC95% throughput | [7217.400; 7285.567] | [4543.693; 4572.357] | [17449.288; 17566.000] |
| Latência média software-visible (ms) | **0.052497** | **0.224716** | **0.054169** |
| Mediana (ms) | 0.052310 | 0.221350 | 0.053930 |
| p95 (ms) | 0.052910 | 0.245390 | 0.054460 |
| p99 (ms) | 0.063580 | 0.248820 | 0.063620 |
| Potência idle (W) | **9.9548** | **9.9513** | **9.9543** |
| Potência ativa (W) | **10.1522** | **10.1530** | **10.1327** |
| Potência dinâmica (W) | **0.1974** | **0.2017** | **0.1784** |
| Energia total (mJ/inf) | **1.41615** | **2.24633** | **0.552614** |
| Energia dinâmica (mJ/inf) | **0.027540** | **0.044624** | **0.009735** |
| CPU durante energia (%) | 25.510 | 25.641 | 25.293 |
| Mismatches | 0 | 0 | 0 |

Acurácia:

```text
29 / 30
96,6667 %
IC95% Wilson = [83,33 ; 99,41] %
```

Kernel físico:

```text
6 ciclos
≈ 60 ns
```

---

# 18. Problemas encontrados e decisões metodológicas

## 18.1 Relógio incorreto da placa

Problema:

O relógio da ZCU104 gerou `RUN_ID = 20250504_150558`, embora a coleta real tenha ocorrido em 2026-09-10.

Solução:

O diretório não foi renomeado para preservar rastreabilidade. O `METADATA.json` registra a data real e informa que o relógio estava incorreto.

---

## 18.2 Primeiro sampler de telemetria dividido entre células

Uma primeira definição da classe `TelemetrySampler` foi dividida em mais de uma célula Jupyter.

Em Python/Jupyter uma definição de classe não pode ser continuada logicamente em outra célula como se fosse o mesmo bloco.

Sintoma:

```text
AttributeError:
'TelemetrySampler' object has no attribute 'start'
```

Solução:

Toda a classe foi redefinida em uma única célula contendo:

```text
__init__
_worker
start
stop
```

---

## 18.3 Validação isolada do sampler baseado em thread

Após correção, teste de aproximadamente 10 s:

```text
amostras = 11

potência média = 9.989545 W

intervalo médio real =
1.000092249 s

CV do intervalo =
0.030578 %

tempo médio de leitura PMBus =
0.000961844 s
≈ 0.962 ms

frequência observada =
0.999907760 Hz
```

A aquisição isolada aparentava estar adequada.

---

## 18.4 Problema da thread de telemetria sob carga

Quando o benchmark batch-1 foi executado simultaneamente à thread, uma janela de 30 s produziu:

```text
23 amostras
```

em vez de aproximadamente 30.

Coleta de calibração:

```text
inferências       = 203336
invocations       = 203336
FPS               = 6777.844747
idle              = 9.989167 W
active            = 10.161174 W
dynamic           = 0.172007 W
E total           = 1.499175 mJ/inf
E dinâmica        = 0.025378 mJ/inf
amostras de power = 23
mismatches        = 0
```

Essa execução foi marcada:

```text
calibration_only = true
exclude_from_final_results = true
```

Motivo:

O sampler de thread perdeu a cadência nominal de 1 Hz sob o loop intenso de Python/MMIO.

Isso NÃO foi remoção de outlier. Foi exclusão de uma calibração metodológica realizada antes das campanhas finais.

---

## 18.5 Primeiro ensaio de impacto da telemetria com sampler antigo

Antes da mudança definitiva do sampler houve um ensaio preliminar com três pares de janelas batch-1.

Impactos observados:

```text
-1.6369 %
-2.2418 %
-1.7166 %
```

Resumo:

```text
média ≈ -1.8651 %
DP ≈ 0.3287 %
IC95% ≈ [-2.6816 ; -1.0486] %
```

Esse valor NÃO é usado como overhead final, pois o sampler e a forma de execução foram modificados posteriormente.

Ele foi mantido apenas como histórico da investigação.

---

# 19. Solução da telemetria: multiprocessing

Para eliminar competição pelo GIL, a leitura PMBus passou a ser executada em processo Python separado utilizando:

```text
multiprocessing
context = fork
```

## 19.1 Teste sob carga

Com batch-1 rodando continuamente por aproximadamente 10 s:

```text
inferências = 69509
throughput  = 6950.598573 inf/s
amostras    = 11
```

Intervalo:

```text
média = 0.999969850 s
CV = 0.006842 %
```

Tempo de leitura PMBus:

```text
média ≈ 1.051 ms
```

Potência média:

```text
10.162273 W
```

Conclusão:

O multiprocess manteve adequadamente a aquisição nominal de 1 Hz mesmo sob carga.

---

# 20. Avaliação do impacto da telemetria final

Foi utilizado um protocolo pareado com execução:

```text
sem telemetria
com telemetria
```

e ordem alternada entre as repetições para reduzir efeito temporal.

Foram usados:

```text
5 pares por cenário
10 s por janela
```

O delta foi calculado por:

```text
delta (%) =
100 × (FPS_com - FPS_sem) / FPS_sem
```

Um delta positivo significa que aquela janela com telemetria foi casualmente mais rápida. Isso NÃO significa que a telemetria melhora desempenho.

## 20.1 Batch-1

```text
FPS sem telemetria:
média = 7279.469946

FPS com telemetria:
média = 7364.698686

delta médio:
+1.183182 %

IC95% t:
[-0.613105 ; +2.979470] %

IC95% bootstrap:
[+0.292096 ; +2.487329] %

amostras por janela:
11 em todas
```

Como o IC95% t inclui zero, não foi considerada evidência robusta de alteração do throughput.

## 20.2 End-to-end

```text
FPS sem:
4579.278395

FPS com:
4588.398993

delta médio:
+0.199786 %

IC95% t:
[-0.667125 ; +1.066698] %

bootstrap:
[-0.424600 ; +0.674237] %

11 amostras em todas
```

Sem evidência de efeito significativo.

## 20.3 Saturated

```text
FPS sem:
18384.231216

FPS com:
18454.450896

delta médio:
+0.383407 %

IC95% t:
[-0.169838 ; +0.936653] %

bootstrap:
[+0.070007 ; +0.780032] %

11 amostras em todas
```

Novamente, o IC95% t inclui zero.

## 20.4 Decisão final

Não foi aplicada qualquer correção matemática de throughput ou energia pela telemetria.

Campo metodológico:

```text
telemetry_performance_correction_applied = false
```

---

# 21. Sincronização temporal da potência

O sampler final registrou timestamps monotônicos absolutos utilizando:

```python
time.perf_counter()
```

A telemetria foi iniciada antes da janela de benchmark e encerrada depois dela.

Foram mantidas amostras de guarda antes e depois da região ativa.

A potência utilizada na média principal foi filtrada para o intervalo:

```text
active_start <= timestamp <= active_end
```

A integração trapezoidal utilizou também interpolação da potência nas bordas da janela.

Isso permitiu calcular:

```text
método principal por potência média / FPS
```

e uma verificação independente por:

```text
integração temporal trapezoidal
```

Os resultados dos dois métodos foram muito próximos, funcionando como validação cruzada.

---

# 22. Por que a potência dinâmica é pequena

A MLP possui somente:

```text
139 parâmetros
```

O consumo total de placa permanece próximo de:

```text
≈ 10 W
```

Enquanto a diferença ativa-idle fica aproximadamente em:

```text
0.18 a 0.20 W
```

Isso significa que a potência dinâmica associada à execução é apenas uma pequena fração do consumo físico total da plataforma.

Consequentemente, a energia dinâmica possui CV maior do que a energia total.

Isso é esperado e deve ser explicitado ao discutir os resultados.

Não é correto interpretar `12V_power` como consumo isolado do kernel MLP.

---

# 23. CPU e polling

A utilização total de CPU observada ficou normalmente próxima de:

```text
25 %
```

Isso é compatível com uma carga intensa de polling/controle no host, mas a interpretação como "um núcleo inteiro" deve ser feita somente considerando a topologia efetiva do sistema.

O benchmark utiliza polling síncrono dos registradores AXI-Lite. Portanto, o host ARM participa ativamente da orquestração.

---

# 24. Diagnóstico CPUFreq/DVFS

Foi investigada uma variação temporal de throughput.

Resultado:

```text
policy0
scaling_governor = userspace
scaling_cur_freq  = 1199999
cpuinfo_cur_freq  = 1199999
scaling_min_freq  = 299999
scaling_max_freq  = 1199999
```

Antes e após carga:

```text
1199999 kHz
```

Portanto, não houve evidência de DVFS como causa da variação.

A variabilidade restante deve ser atribuída ao sistema completo:

```text
Linux
escalonamento
interrupções
Python
Jupyter
MMIO
polling
estado temporal do host
```

---

# 25. Diferença entre throughput, latência e taxa equivalente

É essencial não misturar:

## Kernel físico

```text
≈ 60 ns
```

## Latência software-visible inference-only

```text
≈ 52.50 µs
```

## Throughput batch-1 real

```text
≈ 7251.48 inf/s
```

## Throughput saturado

```text
≈ 17507.64 inf/s
```

O inverso da latência média é uma taxa equivalente matemática e NÃO uma medição de throughput.

Por exemplo:

```text
1 / 52.4967 µs
≈ 19048.8 inf/s
```

Mas o throughput batch-1 efetivamente medido é:

```text
7251.48 inf/s
```

porque o throughput inclui operações do host entre inferências.

---

# 26. Comparação do caminho físico com o overhead de software

A diferença entre:

```text
kernel físico ≈ 0.060 µs
```

e:

```text
latência observável ≈ 52.50 µs
```

mostra que, para esta MLP muito pequena, o tempo interno do kernel é muito menor que o custo de:

```text
AXI-Lite
MMIO
start/status
polling
leitura de outputs
Python
orquestração do ARM
```

Esse é um resultado importante do experimento.

Para redes extremamente pequenas, otimizar somente o datapath interno do kernel não garante redução proporcional da latência percebida pela aplicação.

---

# 27. Observações para comparação com Vitis AI

A comparação hls4ml × Vitis AI deve garantir a mesma fronteira experimental.

Não comparar diretamente:

```text
60 ns de kernel hls4ml
```

com:

```text
latência end-to-end da DPU
```

Também não comparar:

```text
throughput saturado
```

com:

```text
throughput batch-1 da outra plataforma
```

As categorias devem permanecer separadas:

```text
acurácia
latência do caminho acelerado
throughput batch-1
end-to-end
throughput saturado
potência total
potência dinâmica
energia total/inferência
energia dinâmica/inferência
recursos de FPGA
```

A Softmax desta MLP está no hardware hls4ml. Esse detalhe deve ser considerado na comparação com o fluxo Vitis AI caso a DPU deixe Softmax ou argmax para CPU.

---

# 28. Recursos de FPGA

Os valores detalhados de LUT, FF, BRAM, URAM e DSP não foram reconstruídos neste README sem consultar diretamente os relatórios de síntese/implementação correspondentes.

Para evitar inserir valores incorretos, usar como fontes:

```text
COMPARACAO_RECURSOS_VITIS_VIVADO.md
RESULTADOS_IP.md
vivado_zcu104_mlp/
mlp_iris_apfixed16_6_rf1_100mhz/
relatórios de synthesis/implementation/post-route
```

Esses diretórios devem ser incluídos no backup do PC junto com a coleta física da ZCU104.

---

# 29. Arquivos que devem ser preservados

## 29.1 Da ZCU104

Preservar todo:

```text
/home/xilinx/jupyter_notebooks/mlp_iris/results_zcu104/benchmark_complete/run_20250504_150558
```

Além de:

```text
/home/xilinx/jupyter_notebooks/mlp_iris/overlay/
/home/xilinx/jupyter_notebooks/mlp_iris/golden/
```

E os notebooks `.ipynb` associados.

Itens especialmente importantes:

```text
results_zcu104/validation/predictions_unique_30.csv
results_zcu104/validation/validation_unique_30.json

FINAL_RESULTS.json
FINAL_RESULTS_TABLE.csv
THROUGHPUT_FINAL.json
ENERGY_FINAL_AGGREGATE.json
TELEMETRY_VALIDATION.json
METADATA.json

arquivos CSV brutos
arquivos NPY/NPZ
SUMMARY.json de cada execução
telemetria ativa e idle
saturated_16_windows.csv
SATURATED_FINAL.json
arquivos da validação das 30 amostras
```

## 29.2 Do PC

Preservar também:

```text
COMPARACAO_RECURSOS_VITIS_VIVADO.md
RESULTADOS_IP.md
README_CHATGPT_WEB_ZCU104_IP.md
vivado_ooc_post_route.tcl
manifests/
docs/
golden/
logs/
scripts/
mlp_iris_apfixed16_6_rf1_100mhz/
vivado_zcu104_mlp/
```

---

# 30. Estrutura recomendada no PC

Dentro de:

```text
~/Downloads/Plano testes TCC/MLP/hls4ml
```

criar:

```text
coletas_zcu104_mlp_iris/
└── 2026-09-10_run_20250504_150558/
    ├── README_MLP_IRIS_ZCU104_HLS4ML.md
    ├── README_MLP_IRIS_ZCU104_HLS4ML.txt
    ├── board/
    │   ├── results_zcu104/
    │   ├── overlay/
    │   ├── golden/
    │   ├── notebooks/
    │   ├── root_files/
    │   └── environment/
    ├── pc_project_snapshot/
    ├── manifests/
    ├── SHA256SUMS.txt
    └── INVENTARIO.txt
```

---

# 31. Comandos recomendados para coleta na ZCU104

Executar no Terminal do Jupyter/PYNQ.

```bash
cd /home/xilinx/jupyter_notebooks/mlp_iris

COLLECT="coleta_zcu104_mlp_iris_2026-09-10_run_20250504_150558"

rm -rf "$COLLECT"
mkdir -p "$COLLECT"/{results,overlay,golden,notebooks,environment,root_files}

# Copia TODOS os resultados da placa, incluindo:
# - benchmark_complete/
# - validation/
# - arquivos auxiliares que possam existir em results_zcu104/
cp -a results_zcu104 "$COLLECT/results/"

cp -a overlay/. "$COLLECT/overlay/"
cp -a golden/. "$COLLECT/golden/"

# Notebooks
find /home/xilinx/jupyter_notebooks/mlp_iris \
  -maxdepth 1 -type f -name '*.ipynb' \
  -exec cp -a {} "$COLLECT/notebooks/" \;

# Scripts/documentos soltos do projeto na placa
find /home/xilinx/jupyter_notebooks/mlp_iris \
  -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.txt' -o -name '*.json' \) \
  -exec cp -a {} "$COLLECT/root_files/" \;

uname -a > "$COLLECT/environment/uname.txt"
cat /etc/os-release > "$COLLECT/environment/os-release.txt"
hostname > "$COLLECT/environment/hostname.txt"
hostname -I > "$COLLECT/environment/ip-addresses.txt"
date -Is > "$COLLECT/environment/date-board.txt"

python3 --version \
  > "$COLLECT/environment/python-version.txt" 2>&1

python3 -m pip freeze \
  > "$COLLECT/environment/pip-freeze.txt" 2>&1

python3 - <<'PY' > "$COLLECT/environment/pynq-and-rails.txt" 2>&1
import sys
import pynq
from pynq import get_rails

print("Python:", sys.version)
print("PYNQ:", getattr(pynq, "__version__", "unknown"))
print()
print("RAILS:")
for name, rail in get_rails().items():
    print(name, rail)
PY

{
  echo '=== CPU INFO ==='
  lscpu
  echo
  echo '=== MEMORY ==='
  free -h
  echo
  echo '=== DISK ==='
  df -h
  echo
  echo '=== CPUFREQ ==='
  grep -R . /sys/devices/system/cpu/cpufreq/policy*/scaling_governor 2>/dev/null
  grep -R . /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq 2>/dev/null
  grep -R . /sys/devices/system/cpu/cpufreq/policy*/scaling_min_freq 2>/dev/null
  grep -R . /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq 2>/dev/null
} > "$COLLECT/environment/system-info.txt"

# Primeiro gera o inventário.
find "$COLLECT" -type f \
  | sort \
  > "$COLLECT/INVENTARIO.txt"

# Depois gera hashes, excluindo o próprio arquivo de hashes.
find "$COLLECT" -type f \
  ! -name 'SHA256SUMS_CONTENT.txt' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$COLLECT/SHA256SUMS_CONTENT.txt"

tar -czf "${COLLECT}.tar.gz" "$COLLECT"

sha256sum "${COLLECT}.tar.gz" \
  > "${COLLECT}.tar.gz.sha256"

ls -lh "${COLLECT}.tar.gz" "${COLLECT}.tar.gz.sha256"

echo
echo 'IP(s) da ZCU104:'
hostname -I
```

---

# 32. Copiar a coleta para o PC

No PC Ubuntu:

Primeiro definir o IP retornado por `hostname -I` na placa.

Exemplo:

```bash
BOARD_IP="<IP_DA_ZCU104>"
BOARD_USER="xilinx"
```

Destino:

```bash
BASE="$HOME/Downloads/Plano testes TCC/MLP/hls4ml"
DEST="$BASE/coletas_zcu104_mlp_iris/2026-09-10_run_20250504_150558"

mkdir -p "$DEST"
```

Copiar:

```bash
scp \
  "${BOARD_USER}@${BOARD_IP}:/home/xilinx/jupyter_notebooks/mlp_iris/coleta_zcu104_mlp_iris_2026-09-10_run_20250504_150558.tar.gz" \
  "$DEST/"

scp \
  "${BOARD_USER}@${BOARD_IP}:/home/xilinx/jupyter_notebooks/mlp_iris/coleta_zcu104_mlp_iris_2026-09-10_run_20250504_150558.tar.gz.sha256" \
  "$DEST/"
```

Verificar o arquivo:

```bash
cd "$DEST"

sha256sum -c \
  coleta_zcu104_mlp_iris_2026-09-10_run_20250504_150558.tar.gz.sha256
```

Resultado esperado:

```text
OK
```

Extrair:

```bash
tar -xzf \
  coleta_zcu104_mlp_iris_2026-09-10_run_20250504_150558.tar.gz
```

---

# 33. Snapshot do projeto local do PC

Depois de baixar os dados da placa, ainda no PC:

```bash
BASE="$HOME/Downloads/Plano testes TCC/MLP/hls4ml"
DEST="$BASE/coletas_zcu104_mlp_iris/2026-09-10_run_20250504_150558"

mkdir -p "$DEST/pc_project_snapshot"
```

Copiar os artefatos de projeto relevantes:

```bash
cd "$BASE"

cp -a COMPARACAO_RECURSOS_VITIS_VIVADO.md \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a RESULTADOS_IP.md \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a README_CHATGPT_WEB_ZCU104_IP.md \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a vivado_ooc_post_route.tcl \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a manifests \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a docs \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a golden \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a logs \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a scripts \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a mlp_iris_apfixed16_6_rf1_100mhz \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true

cp -a vivado_zcu104_mlp \
  "$DEST/pc_project_snapshot/" 2>/dev/null || true
```

Copiar este README para a pasta final:

```bash
cp "$BASE/README_MLP_IRIS_ZCU104_HLS4ML.md" "$DEST/"
cp "$BASE/README_MLP_IRIS_ZCU104_HLS4ML.txt" "$DEST/"
```

---

# 34. Gerar manifesto final no PC

Depois que tudo estiver copiado:

```bash
cd "$DEST"

find . -type f \
  ! -name 'SHA256SUMS_FINAL.txt' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > SHA256SUMS_FINAL.txt

find . -type f \
  | sort \
  > INVENTARIO_FINAL.txt
```

Tamanho total:

```bash
du -sh .
```

Quantidade de arquivos:

```bash
find . -type f | wc -l
```

Revisar arquivos principais:

```bash
find . -type f | sort | less
```

---

# 35. Criar um ZIP adicional para arquivamento

Opcionalmente:

```bash
cd "$HOME/Downloads/Plano testes TCC/MLP/hls4ml/coletas_zcu104_mlp_iris"

zip -r \
  2026-09-10_run_20250504_150558.zip \
  2026-09-10_run_20250504_150558
```

Gerar hash:

```bash
sha256sum \
  2026-09-10_run_20250504_150558.zip \
  > 2026-09-10_run_20250504_150558.zip.sha256
```

---

# 36. Arquivos finais recomendados como fonte oficial

Para análise futura, a ordem recomendada é:

## Fonte consolidada principal

```text
FINAL_RESULTS.json
```

## Tabela simples

```text
FINAL_RESULTS_TABLE.csv
```

## Throughput

```text
THROUGHPUT_FINAL.json
```

## Energia

```text
ENERGY_FINAL_AGGREGATE.json
```

## Telemetria

```text
TELEMETRY_VALIDATION.json
```

## Ambiente

```text
METADATA.json
environment/
```

## Dados brutos

Todos os CSV/NPY/NPZ e `SUMMARY.json` das subpastas.

Os arquivos brutos devem ser mantidos mesmo quando existe um JSON consolidado.

---

# 37. Valores recomendados para o TCC

## Acurácia

```text
96,67 %
```

## Latência do kernel físico

```text
6 ciclos
≈ 60 ns
```

## Latência inference-only software-visible

```text
média = 0.05250 ms
p95   = 0.05291 ms
```

## Throughput batch-1

```text
7251.48 inf/s
IC95% [7217.40 ; 7285.57]
```

## End-to-end

```text
latência média = 0.22472 ms
throughput = 4558.03 inf/s
IC95% throughput [4543.69 ; 4572.36]
```

## Saturated

```text
17507.64 inf/s
IC95% [17449.29 ; 17566.00]
```

## Potência ativa

```text
batch-1   = 10.1522 W
E2E       = 10.1530 W
saturated = 10.1327 W
```

## Potência dinâmica

```text
batch-1   = 0.1974 W
E2E       = 0.2017 W
saturated = 0.1784 W
```

## Energia total

```text
batch-1   = 1.41615 mJ/inf
E2E       = 2.24633 mJ/inf
saturated = 0.55261 mJ/inf
```

## Energia dinâmica

```text
batch-1   = 0.02754 mJ/inf
E2E       = 0.04462 mJ/inf
saturated = 0.009735 mJ/inf
```

---

# 38. Cuidados de interpretação no artigo

1. Não chamar 60 ns de latência end-to-end.
2. Não chamar `1 / latência média` de throughput medido.
3. Não substituir batch-1 pelo saturado.
4. Não usar o throughput da campanha energética como throughput principal; ele existe para o cálculo de energia na mesma janela.
5. Não tratar as 3000 repetições determinísticas como novas observações de acurácia.
6. Não afirmar que `12V_power` mede apenas o kernel.
7. Não comparar o E2E desta MLP com um E2E iniciado em dados brutos sem explicar que aqui a entrada já está normalizada.
8. Não omitir que a Softmax está no hardware.
9. Não remover os máximos de latência.
10. Não escolher somente o melhor bloco saturado.
11. Não usar os valores instrumentados de 6427/4068 inf/s como throughput principal final.
12. Preservar os valores instrumentados como evidência histórica e de latência.
13. Registrar que o sampler final usa processo separado.
14. Registrar o erro de relógio da placa.
15. Ao comparar com Vitis AI, usar fronteiras equivalentes.

---

# 39. Conclusão técnica

A implementação física da MLP Iris com hls4ml na ZCU104 apresentou funcionamento determinístico e equivalência exata de classes com a referência HLS para as 30 amostras do holdout, mantendo acurácia de 96,67%.

O kernel físico apresenta latência de aproximadamente 60 ns, porém a latência software-visible é da ordem de 52,5 µs, evidenciando que, para uma rede de apenas 139 parâmetros, o custo de comunicação e orquestração AXI-Lite/ARM/Python é muito maior que o custo computacional interno do kernel.

O throughput batch-1 final medido sem instrumentação individual foi aproximadamente 7,25 mil inferências por segundo. O cenário end-to-end normalizado-float-to-class atingiu aproximadamente 4,56 mil inferências por segundo, enquanto o caminho saturado agrupado atingiu aproximadamente 17,51 mil inferências por segundo.

A potência total da placa permaneceu próxima de 10 W. A potência dinâmica incremental observada foi pequena, aproximadamente 0,18–0,20 W, coerente com a pequena complexidade da MLP frente ao consumo base da plataforma. A energia total por inferência variou de aproximadamente 0,553 mJ no saturado a 2,246 mJ no cenário end-to-end.

A campanha energética final utilizou aquisição PMBus a 1 Hz em processo separado, após demonstrar que a versão baseada em thread perdia amostras sob carga. Todas as 15 janelas finais mantiveram o número esperado de amostras, zero mismatches e contadores de invocação corretos.

A metodologia final preserva separadamente:

```text
acurácia
latência física do kernel
latência software-visible
throughput batch-1
end-to-end
throughput saturado
potência total
potência dinâmica
energia total
energia dinâmica
telemetria
integridade
reprodutibilidade
```

Essas separações devem ser mantidas em qualquer comparação futura com Vitis AI, CPU ou GPU.

---

# 40. Checklist antes de encerrar o experimento

- [x] Hardware validado contra HLS
- [x] 30/30 classes hardware = HLS
- [x] Acurácia calculada em amostras únicas
- [x] 3000 execuções de determinismo sem erro
- [x] Kernel físico medido em 6 ciclos
- [x] Latência individual coletada
- [x] Throughput batch-1 final coletado sem timing individual
- [x] E2E final coletado sem timing individual
- [x] Saturated medido em dois blocos
- [x] 16 janelas saturated preservadas
- [x] CPUFreq verificado
- [x] Telemetria em thread diagnosticada
- [x] Telemetria migrada para multiprocessing
- [x] Impacto da telemetria caracterizado
- [x] 15 janelas energéticas válidas
- [x] Integração energética usada como verificação
- [x] Nenhum outlier removido
- [x] JSON final gerado
- [x] CSV final gerado
- [x] Data real registrada apesar do relógio incorreto
- [x] Hashes dos artefatos principais registrados
- [ ] Copiar coleta completa da ZCU104 para o PC
- [ ] Gerar SHA256SUMS_FINAL.txt no PC
- [ ] Criar snapshot do projeto Vivado/hls4ml local
- [ ] Fazer backup adicional externo/Git/Drive conforme desejado

---

Documento referente à campanha física da MLP Iris / hls4ml / ZCU104 realizada em 10 de setembro de 2026.
