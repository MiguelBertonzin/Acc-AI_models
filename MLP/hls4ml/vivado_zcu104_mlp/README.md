# MLP Iris — hls4ml + Vivado + ZCU104

Implementação completa de uma **MLP para classificação do dataset Iris** em hardware customizado utilizando **hls4ml**, **Vitis HLS 2024.2** e **Vivado 2024.2**, com integração ao Processing System da **AMD/Xilinx ZCU104** por **AXI4-Lite**.

Este diretório reúne o wrapper RTL, relatórios de síntese/implementação, artefatos finais (`.bit`, `.hwh`, `.xsa`), script Tcl de reconstrução e arquivos de rastreabilidade usados no TCC.

## 1. Objetivo

Esta implementação faz parte do TCC que compara dois fluxos de implantação de redes neurais na ZCU104:

- **hls4ml**: acelerador customizado sintetizado especificamente para a rede;
- **Vitis AI / DPU**: execução da rede sobre a arquitetura DPU.

A MLP Iris funciona como um caso pequeno e controlado para validar conversão do modelo, geração do IP em HLS, integração PS–PL no Vivado, acesso AXI, geração do bitstream, validação física e medição de acurácia, latência, vazão, potência e energia.

## 2. Estado atual

### Concluído

- [x] treinamento e validação do modelo;
- [x] conversão para hls4ml;
- [x] geração e exportação do IP no Vitis HLS;
- [x] importação do IP no Vivado;
- [x] criação do wrapper AXI4-Lite;
- [x] criação e validação do Block Design;
- [x] configuração do Zynq UltraScale+ MPSoC;
- [x] clock PL de 100 MHz;
- [x] reset e SmartConnect;
- [x] endereço AXI atribuído;
- [x] síntese;
- [x] implementation / place / route;
- [x] timing pós-route aprovado;
- [x] DRC;
- [x] bitstream;
- [x] HWH;
- [x] XSA com bitstream incluído;
- [x] projeto Vivado renomeado para `MLP-Iris-tcc`;
- [x] Tcl de reconstrução salvo;
- [x] hashes SHA-256 dos artefatos finais.

### Próximas etapas

- [ ] carregar o overlay na ZCU104;
- [ ] smoke test de uma inferência;
- [ ] validar as 30 amostras Iris;
- [ ] medir latência;
- [ ] medir throughput;
- [ ] medir potência;
- [ ] calcular energia por inferência;
- [ ] comparar com CPU, GPU e Vitis AI/DPU.

## 3. Modelo

Arquitetura:

```text
4 entradas
   |
Dense(8)
   |
ativação
   |
Dense(8)
   |
ativação
   |
Dense(3)
   |
Softmax
   |
3 classes
```

| Parâmetro | Valor |
|---|---|
| Dataset | Iris |
| Arquitetura | `4 -> 8 -> 8 -> 3` |
| Parâmetros treináveis | 139 |
| Saídas | 3 classes |
| Acurácia de referência | 29/30 = 96,67% |
| Softmax | mantida no hardware |
| Precisão | `ap_fixed<16,6>` |
| Bits fracionários | 10 |
| Reuse Factor | 1 |
| Strategy | `Latency` |
| IOType | `io_parallel` |
| Clock | 100 MHz |
| Período | 10 ns |

## 4. Ambiente

| Item | Configuração |
|---|---|
| hls4ml | 1.3.0 |
| Vitis HLS | 2024.2 |
| Vivado | 2024.2 |
| FPGA | `xczu7ev-ffvc1156-2-e` |
| Board | `xilinx.com:zcu104:part0:1.1` |
| Plataforma | AMD/Xilinx ZCU104 |

## 5. Projeto Vivado oficial

O projeto foi inicialmente criado com o nome incorreto `LeNetMNIST-tcc`, embora o hardware implementado sempre tenha sido a **MLP Iris**.

O projeto oficial atual é:

```text
/home/miguel/Desktop/MLP-Iris-tcc/MLP-Iris-tcc.xpr
```

Top:

```text
mlp_iris_bd_wrapper
```

Block Design:

```text
mlp_iris_bd
```

IP:

```text
mlp_iris_core
```

Wrapper:

```text
mlp_iris_axi_wrapper
```

O projeto antigo `LeNetMNIST-tcc` deve ser tratado apenas como backup histórico.

## 6. IP hls4ml

VLNV:

```text
xilinx.com:hls:mlp_iris:1.0
```

Repositório do IP:

```text
/home/miguel/Downloads/Plano testes TCC/MLP/hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/mlp_iris_prj/solution1/impl/ip
```

SHA-256 validado do pacote original do IP:

```text
88a11380ea472636af4b4b72348aeda9e0b3b2421456b71b0b30222b2a774ba3
```

O IP original deve permanecer inalterado.

## 7. Interface original do IP

O IP foi gerado com `io_parallel` e expõe:

```text
ap_clk
ap_rst
ap_start
ap_done
ap_idle
ap_ready

features[63:0]
features_ap_vld

layer7_out_0[15:0]
layer7_out_0_ap_vld
layer7_out_1[15:0]
layer7_out_1_ap_vld
layer7_out_2[15:0]
layer7_out_2_ap_vld
```

O IP não possui AXI nativo:

```text
IS_AXI = 0
```

## 8. Wrapper AXI4-Lite

Arquivo:

```text
src/mlp_iris_axi_wrapper.v
```

O wrapper converte transações AXI4-Lite em sinais do protocolo HLS.

Funções:

- registrar as quatro entradas;
- gerar `features_ap_vld`;
- controlar `ap_start`;
- monitorar `ap_done`, `ap_idle` e `ap_ready`;
- armazenar as três saídas;
- armazenar os sinais `ap_vld`;
- manter `DONE_STICKY`;
- contabilizar ciclos;
- contabilizar invocações.

### Diferença em relação à ResNet8

A ResNet8 usa `io_stream`, AXI DMA e AXI4-Stream. Seu wrapper faz adaptação de largura de stream e tratamento de sinais como `TLAST`.

A MLP usa `io_parallel` e poucos dados, portanto AXI4-Lite/MMIO é mais simples e adequado.

## 9. Arquitetura do Block Design

```text
+---------------------------+
| Zynq UltraScale+ MPSoC    |
| ARM Cortex-A53            |
+-------------+-------------+
              |
       M_AXI_HPM0_FPD
              |
              v
      +---------------+
      | SmartConnect  |
      |      1 x 1    |
      +-------+-------+
              |
          AXI4-Lite
              |
              v
+----------------------------+
| mlp_iris_axi_wrapper       |
+-------------+--------------+
              |
              v
      +---------------+
      | mlp_iris_core |
      |     hls4ml    |
      +---------------+
```

Blocos principais:

```text
zynq_ultra_ps_e_0
axi_smc
rst_ps8_0_100M
mlp_iris_axi_wrapper_0
```

## 10. Clock e reset

Clock:

```text
clk_pl_0 = 100 MHz
Period   = 10.000 ns
```

Reset AXI:

```text
s_axi_aresetn
```

ativo em nível baixo.

Reset do IP HLS:

```text
ap_rst
```

ativo em nível alto.

Conversão:

```verilog
.ap_rst(~s_axi_aresetn)
```

## 11. Endereço AXI

Base:

```text
0xA0000000
```

Range:

```text
4 KiB
```

Caminho:

```text
zynq_ultra_ps_e_0/Data
 -> M_AXI_HPM0_FPD
 -> axi_smc
 -> mlp_iris_axi_wrapper_0/S_AXI
```

## 12. Mapa de registradores

| Offset | Acesso | Função |
|---:|:---:|---|
| `0x00` | R/W | controle/status |
| `0x10` | R/W | `features[31:0]` |
| `0x14` | R/W | `features[63:32]` |
| `0x20` | R | saída 0 |
| `0x24` | R | saída 1 |
| `0x28` | R | saída 2 |
| `0x2C` | R | valid bits |
| `0x30` | R | ciclos da última inferência |
| `0x34` | R | contador de invocações |

### `0x00`

Escrita:

```text
bit 0 -> START
bit 1 -> CLEAR_DONE
```

Leitura:

```text
bit 1 -> DONE_STICKY
bit 2 -> AP_IDLE
bit 3 -> AP_READY
bit 4 -> BUSY
```

## 13. Formato numérico

```text
ap_fixed<16,6>
```

Bits fracionários:

```text
10
```

Escala:

```text
2^10 = 1024
```

Conversão:

```text
raw   = round(valor * 1024)
valor = signed(raw) / 1024
```

## 14. Packing das entradas

```text
features[15:0]   = x0
features[31:16]  = x1
features[47:32]  = x2
features[63:48]  = x3
```

Exemplo:

```python
def float_to_fixed16(x):
    raw = int(round(x * 1024.0))
    return raw & 0xFFFF

x0 = float_to_fixed16(features[0])
x1 = float_to_fixed16(features[1])
x2 = float_to_fixed16(features[2])
x3 = float_to_fixed16(features[3])

low  = x0 | (x1 << 16)
high = x2 | (x3 << 16)

ip.write(0x10, low)
ip.write(0x14, high)
```

## 15. Decodificação das saídas

```python
def fixed16_to_float(raw):
    raw &= 0xFFFF
    if raw & 0x8000:
        raw -= 0x10000
    return raw / 1024.0
```

Uso:

```python
y0 = fixed16_to_float(ip.read(0x20))
y1 = fixed16_to_float(ip.read(0x24))
y2 = fixed16_to_float(ip.read(0x28))

scores = [y0, y1, y2]
pred = max(range(3), key=lambda i: scores[i])
```

## 16. Sequência de inferência

```python
ip.write(0x10, low)
ip.write(0x14, high)

ip.write(0x00, 0x2)  # CLEAR_DONE
ip.write(0x00, 0x1)  # START

while True:
    status = ip.read(0x00)
    if (status >> 1) & 1:
        break

valid = ip.read(0x2C)

y0_raw = ip.read(0x20)
y1_raw = ip.read(0x24)
y2_raw = ip.read(0x28)

cycles = ip.read(0x30)
```

Em benchmark, utilizar timeout.

## 17. Síntese integrada

Resultado do design integrado após síntese:

| Recurso | Uso |
|---|---:|
| LUT | 3122 |
| FF | 1436 |
| DSP48E2 | 108 |
| RAMB18 | 3 |
| URAM | 0 |

Arquivos:

```text
reports/utilization_post_synth.rpt
reports/utilization_hierarchical_post_synth.rpt
```

Os relatórios pós-route devem ser usados como referência final quando a utilização física for necessária.

## 18. Timing final pós-route

| Métrica | Valor |
|---|---:|
| WNS | +3.098 ns |
| TNS | 0.000 ns |
| Setup failing endpoints | 0 |
| WHS | +0.019 ns |
| THS | 0.000 ns |
| Hold failing endpoints | 0 |
| Pulse Width Slack | +3.500 ns |

Vivado:

```text
All user specified timing constraints are met.
```

Verificações:

```text
no_clock = 0
constant_clock = 0
unconstrained_internal_endpoints = 0
multiple_clock = 0
generated_clocks = 0
combinational loops = 0
latch loops = 0
```

## 19. Routing

```text
Failed Nets           = 0
Unrouted Nets         = 0
Partially Routed Nets = 0
Node Overlaps         = 0
```

## 20. Estrutura atual do diretório

```text
vivado_zcu104_mlp/
├── README.md
├── src/
│   └── mlp_iris_axi_wrapper.v
├── tcl/
│   └── rebuild_mlp_iris_zcu104.tcl
├── reports/
│   ├── check_timing_post_route.rpt
│   ├── clocks_post_synth.rpt
│   ├── drc_post_route.rpt
│   ├── route_status_post_route.rpt
│   ├── timing_post_route.rpt
│   ├── timing_post_synth.rpt
│   ├── utilization_hierarchical_post_route.rpt
│   ├── utilization_hierarchical_post_synth.rpt
│   ├── utilization_post_route.rpt
│   └── utilization_post_synth.rpt
├── output/
│   ├── mlp_iris_zcu104.bit
│   ├── mlp_iris_zcu104.hwh
│   └── mlp_iris_zcu104.xsa
├── manifests/
│   └── sha256_final.txt
├── pynq/
├── results_zcu104/
├── logs/
└── tb/
```

## 21. Artefatos finais e tamanhos

| Arquivo | Tamanho aproximado |
|---|---:|
| `output/mlp_iris_zcu104.bit` | 19 MB |
| `output/mlp_iris_zcu104.hwh` | 172 KB |
| `output/mlp_iris_zcu104.xsa` | 1,1 MB |
| `tcl/rebuild_mlp_iris_zcu104.tcl` | 53 KB |
| `src/mlp_iris_axi_wrapper.v` | 13 KB |

## 22. SHA-256 dos artefatos finais

Arquivo:

```text
manifests/sha256_final.txt
```

Conteúdo:

```text
289d1e1d63e383cb379cc7fd30af548ebcf80741b7a79cbbe7d6ae3f58458e46  output/mlp_iris_zcu104.bit
2b7b7619cca9de08627fcda18956cefbbc93417a37519a09fe04ace6906b492c  output/mlp_iris_zcu104.hwh
75ac3240ded744ccf3e0f3628ac271e3c740876d81013a663f2d4a761c8d2812  output/mlp_iris_zcu104.xsa
ab91e4baac865f84f2716a06b682bdab0e3b12f058f4fc92289b0f548454e049  src/mlp_iris_axi_wrapper.v
```

Esses hashes identificam exatamente o hardware usado nos testes.

## 23. XSA

A XSA final foi gerada diretamente no repositório:

```text
output/mlp_iris_zcu104.xsa
```

Com bitstream incluído:

```tcl
write_hw_platform     -fixed     -include_bit     -force
```

## 24. Tcl de reconstrução

Arquivo:

```text
tcl/rebuild_mlp_iris_zcu104.tcl
```

Gerado com:

```tcl
write_project_tcl -force ...
```

### Atenção à module reference

O projeto contém:

```text
mlp_iris_axi_wrapper
```

como módulo RTL externo.

O script depende de:

```text
src/mlp_iris_axi_wrapper.v
```

O Tcl também utiliza `origin_dir`; portanto, para preservar reprodutibilidade, mantenha a estrutura deste diretório consistente.

## 25. Carregamento no PYNQ

Arquivos essenciais:

```text
mlp_iris_zcu104.bit
mlp_iris_zcu104.hwh
```

Exemplo:

```python
from pynq import Overlay

overlay = Overlay("mlp_iris_zcu104.bit")
print(overlay.ip_dict)
```

## 26. Acesso via MMIO

```python
from pynq import MMIO

BASE_ADDR = 0xA0000000
ADDR_RANGE = 0x1000

ip = MMIO(BASE_ADDR, ADDR_RANGE)
```

Antes de benchmarks definitivos, confirmar o endereço com:

```python
overlay.ip_dict
```

## 27. Primeiro teste físico

Ordem recomendada:

1. carregar o overlay;
2. imprimir `overlay.ip_dict`;
3. ler `0x00`;
4. verificar `AP_IDLE`;
5. carregar uma entrada conhecida;
6. limpar `DONE`;
7. emitir `START`;
8. aguardar `DONE_STICKY`;
9. verificar `0x2C`;
10. ler as três saídas;
11. decodificar `ap_fixed<16,6>`;
12. comparar com a referência.

## 28. Validação de acurácia

Usar as mesmas 30 amostras do conjunto de teste de referência.

Registrar:

```text
sample
classe esperada
classe prevista
score0
score1
score2
acerto/erro
```

Referência atual:

```text
29/30 = 96,67%
```

## 29. Latência

O wrapper disponibiliza:

```text
0x30 = ciclos da última inferência
```

Com 100 MHz:

```text
1 ciclo = 10 ns
```

Logo:

```python
latency_hw_ns = cycles * 10
latency_hw_us = cycles * 0.01
```

Essa medida representa o intervalo observado pelo wrapper e não deve ser confundida com a latência end-to-end.

## 30. Latência end-to-end

```python
from time import perf_counter_ns

t0 = perf_counter_ns()

# writes
# start
# polling
# reads

t1 = perf_counter_ns()

latency_e2e_ns = t1 - t0
```

Inclui Python, MMIO, AXI, polling e acelerador.

## 31. Throughput

Após warm-up:

```python
throughput = num_inferences / elapsed_seconds
```

Registrar número de inferências, tempo total, média, mediana, desvio padrão, throughput, batch e repetições.

## 32. Potência e energia

Manter separadas:

```text
P_idle
P_load
P_dynamic
```

```text
P_dynamic = P_load - P_idle
```

Energia dinâmica por inferência:

```text
E_dynamic/inference = P_dynamic / throughput
```

Manter a mesma metodologia das demais plataformas do TCC.

## 33. Reprodutibilidade experimental

Registrar em cada execução:

- SHA-256 do `.bit`;
- SHA-256 do `.hwh`;
- SHA-256 do `.xsa`;
- SHA-256 do wrapper;
- Vivado;
- Vitis HLS;
- hls4ml;
- precisão;
- RF;
- Strategy;
- IOType;
- clock;
- dataset;
- preprocessing;
- amostras;
- batch;
- warm-up;
- repetições;
- método de medição de potência.

Não misturar resultados obtidos com artefatos de hashes diferentes.

## 34. Resumo final

```text
Modelo:               MLP Iris
Arquitetura:          4 -> 8 -> 8 -> 3
Parâmetros:           139
Acurácia referência:  96,67%

hls4ml:               1.3.0
Vitis HLS:            2024.2
Vivado:               2024.2

Board:                ZCU104
FPGA:                 xczu7ev-ffvc1156-2-e

Precisão:             ap_fixed<16,6>
Reuse Factor:         1
Strategy:             Latency
IOType:               io_parallel
Clock:                100 MHz

Base AXI:             0xA0000000

Síntese integrada:
LUT:                  3122
FF:                   1436
DSP:                  108
RAMB18:               3
URAM:                 0

Timing pós-route:
WNS:                  +3.098 ns
TNS:                  0.000 ns
WHS:                  +0.019 ns
THS:                  0.000 ns

Routing:
Failed Nets:          0
Unrouted Nets:        0
Partially Routed:     0

Projeto Vivado:
MLP-Iris-tcc

Artefatos finais:
output/mlp_iris_zcu104.bit
output/mlp_iris_zcu104.hwh
output/mlp_iris_zcu104.xsa
```

## 35. Próxima etapa

```text
.bit + .hwh
      |
      v
ZCU104 / PYNQ
      |
      v
smoke test
      |
      v
30 amostras Iris
      |
      v
acurácia
      |
      v
latência
      |
      v
throughput
      |
      v
potência
      |
      v
energia por inferência
      |
      v
comparação com Vitis AI / DPU
```

## Observação final

O hardware descrito neste README corresponde aos artefatos identificados por:

```text
manifests/sha256_final.txt
```

Qualquer nova síntese, implementation ou geração de bitstream deve produzir novos hashes e ser tratada como uma nova configuração experimental.
