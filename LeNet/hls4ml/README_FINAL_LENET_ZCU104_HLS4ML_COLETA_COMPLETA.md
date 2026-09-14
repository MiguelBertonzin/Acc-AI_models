# LeNet / MNIST — hls4ml → Vitis HLS → Vivado → ZCU104 → PYNQ
## Registro técnico completo da implementação, validação física e benchmark

**Data real da campanha física:** 10/09/2026  
**Placa:** AMD/Xilinx ZCU104 — Zynq UltraScale+ XCZU7EV  
**Fluxo:** Keras → hls4ml → Vitis HLS → IP RTL → Vivado → AXI DMA → PYNQ → validação e benchmark físicos  
**Identidade recomendada da configuração:** `LeNet_MNIST_hls4ml_Q22.12_RF5-50-64-60-42_Resource_LineBuffer_100MHz_ZCU104`  
**Abreviação:** `LENET_Q22_12_RF_CAP64_100M`

> IMPORTANTE SOBRE A DATA: o relógio da ZCU104 estava incorreto durante a campanha e gerou o `RUN_ID=20250504_164021`. A campanha física ocorreu de fato em **10/09/2026**. O diretório com o nome antigo deve ser preservado para rastreabilidade, mas qualquer pacote/arquivo final criado no PC deve usar a data real de 2026-09-10.

---

# 1. Objetivo deste documento

Este documento registra, de forma consolidada, a campanha completa de implementação e avaliação física da LeNet/MNIST usando hls4ml na ZCU104. Ele deve ser usado posteriormente como referência para:

- escrever a metodologia e a seção de resultados do TCC;
- reproduzir o experimento;
- conferir exatamente quais limites foram usados para latência, throughput, potência e energia;
- localizar os arquivos físicos e os resultados brutos;
- entender problemas encontrados durante HLS, integração Vivado, DMA, PYNQ e telemetria;
- evitar repetir decisões ou erros já resolvidos;
- comparar posteriormente esta implementação hls4ml com Vitis AI/DPU;
- auditar resultados por meio dos hashes SHA256 registrados.

O princípio adotado durante toda a campanha foi **não misturar métricas de fronteiras diferentes**. Em particular, foram tratados separadamente:

1. validação de acurácia;
2. determinismo/integridade;
3. latência DMA+FPGA / inference-only;
4. latência end-to-end;
5. throughput DMA+FPGA;
6. throughput end-to-end;
7. carga máxima serial sustentada;
8. potência idle e ativa;
9. potência dinâmica;
10. energia total por inferência;
11. energia dinâmica por inferência.

Nenhum outlier foi removido das campanhas finais.

---

# 2. Workspace e caminhos

## 2.1 PC

Workspace principal do hls4ml:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml
```

Modelo Keras:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/lenet_mnist_final.h5
```

Projeto Vivado original:

```text
/home/miguel/LeNet-tcc3-08_09.xpr
/home/miguel/LeNet-tcc3-08_09.srcs
/home/miguel/LeNet-tcc3-08_09.gen
/home/miguel/LeNet-tcc3-08_09.runs
```

Block Design:

```text
/home/miguel/LeNet-tcc3-08_09.srcs/sources_1/bd/system/system.bd
```

Bitstream original pós-implementação:

```text
/home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper.bit
```

HWH original:

```text
/home/miguel/LeNet-tcc3-08_09.gen/sources_1/bd/system/hw_handoff/system.hwh
```

Deployment recomendado no workspace do TCC:

```text
hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/
```

Nomes usados na ZCU104:

```text
lenet_zcu104.bit
lenet_zcu104.hwh
```

## 2.2 ZCU104

Usuário:

```text
xilinx
```

IP utilizado durante esta campanha:

```text
192.168.2.99
```

Workspace completo na placa:

```text
/home/xilinx/jupyter_notebooks/lenet_mnist
```

Estrutura lógica importante:

```text
/home/xilinx/jupyter_notebooks/lenet_mnist/
├── overlay/
├── golden/
└── results_zcu104/
```

Diretório da campanha principal:

```text
/home/xilinx/jupyter_notebooks/lenet_mnist/results_zcu104/benchmark_complete/run_20250504_164021
```

Diretório de validação:

```text
/home/xilinx/jupyter_notebooks/lenet_mnist/results_zcu104/validation
```

Novamente: `20250504_164021` veio do relógio incorreto da placa. A data correta da campanha é **2026-09-10**.

---

# 3. Hardware e ambiente de execução

Placa:

```text
AMD/Xilinx ZCU104
Zynq UltraScale+ MPSoC
FPGA part = xczu7ev-ffvc1156-2-e
Board part = xilinx.com:zcu104:part0:1.1
```

Ambiente hls4ml/Vivado:

```text
hls4ml    = 1.3.0
Backend   = Vitis
Vitis HLS = 2024.2
Vivado    = 2024.2
```

Ambiente de execução da placa:

```text
PYNQ = 3.1.1
Kernel observado = 6.6.10-xilinx-v2024.1-gfc1c20f0a1a9 aarch64
```

Clock PL usado pelo acelerador:

```text
FCLK0 ≈ 99.999 MHz
alvo nominal = 100 MHz
período = 10 ns
```

Durante os benchmarks foi conferido o estado do ARM A53:

```text
scaling_governor = userspace
scaling_cur_freq  = 1199999 kHz
scaling_min_freq  = 299999 kHz
scaling_max_freq  = 1199999 kHz
cpuinfo_min_freq  = 299999 kHz
cpuinfo_max_freq  = 1199999 kHz
scaling_setspeed  = 1199999 kHz
```

Portanto, para as medições que envolvem software no ARM, o processador permaneceu aproximadamente em 1,2 GHz e não ficou variando livremente por DVFS.

---

# 4. Modelo LeNet congelado

Entrada:

```text
28 × 28 × 1
```

Arquitetura:

```text
Entrada 28×28×1
  ↓
Conv2D: 6 filtros 5×5, valid
  ↓
24×24×6
  ↓
ReLU
  ↓
MaxPooling2D 2×2
  ↓
12×12×6
  ↓
Conv2D: 16 filtros 5×5, valid
  ↓
8×8×16
  ↓
ReLU
  ↓
MaxPooling2D 2×2
  ↓
4×4×16
  ↓
Flatten = 256
  ↓
Dense 120
  ↓
ReLU
  ↓
Dense 84
  ↓
ReLU
  ↓
Dense 10 linear
  ↓
10 logits
```

Parâmetros:

| Camada | Parâmetros |
|---|---:|
| Conv1 | 156 |
| Conv2 | 2.416 |
| Dense1 | 30.840 |
| Dense2 | 10.164 |
| Dense saída | 850 |
| **Total** | **44.426** |

O modelo final **não possui Softmax**. A decisão de classe é:

```python
pred = np.argmax(logits)
```

Essa decisão foi mantida para não introduzir Softmax no hardware apenas para classificação e para manter uma fronteira comparável com outros aceleradores.

---

# 5. Configuração hls4ml congelada

```text
Backend            = Vitis
Part               = xczu7ev-ffvc1156-2-e
ClockPeriod        = 10 ns
ClockFrequency     = 100 MHz
IOType             = io_stream
Strategy           = Resource
ConvImplementation = LineBuffer
Batch              = 1
Softmax            = não
Saída              = logits
```

Precisão:

```text
ap_fixed<22,12,AP_RND_CONV,AP_SAT>
```

Formato:

```text
22 bits totais
12 bits inteiros, incluindo sinal
10 bits fracionários
escala = 2^10 = 1024
```

Reuse Factors:

```text
Conv1   = 5
Conv2   = 50
Dense1  = 64
Dense2  = 60
Output  = 42
```

Essa configuração foi congelada para toda a campanha. Alterar precisão, RF, estratégia, clock, modelo ou pesos produziria outro experimento.

---

# 6. Por que essa configuração foi escolhida

Durante o estudo de Reuse Factor foi constatado que um `ReuseFactor=64` global não era uma boa definição. O valor 64 é estruturalmente válido diretamente para algumas camadas, mas o backend precisa adaptar outras, e a primeira convolução ficaria desnecessariamente lenta.

Foram estudados candidatos, entre eles:

```text
Refinado conservador:
RF = 5 / 150 / 128 / 120 / 84

RF<=64 balanceado, usado no bitstream final:
RF = 5 / 50 / 64 / 60 / 42
```

O candidato final trocou mais recursos por maior desempenho analítico.

Escolhas principais:

- `io_stream`: adequado para CNN e para fluxo AXI4-Stream;
- `LineBuffer`: implementação apropriada para convoluções;
- `Resource`: evitou paralelismo inviável que `Latency`/RF=1 causaria;
- 100 MHz: ponto conservador e reproduzível para integração DMA/PS;
- Q22.12: 10 bits fracionários e grande faixa inteira;
- `AP_RND_CONV`: arredondamento;
- `AP_SAT`: saturação em vez de wrap-around silencioso;
- saída em logits: classificação por `argmax`.

---

# 7. Referência software antes da FPGA

A validação C++ do hls4ml foi feita nas **10.000 imagens do conjunto de teste MNIST** antes da implementação física.

Resultados de referência:

```text
Keras accuracy       = 98,98%
hls4ml C++ accuracy  = 98,99%
Keras/HLS agreement  = 99,99%
```

Comparação de logits Keras × hls4ml C++:

```text
MAE                    = 0.0156489812
Max absolute logit err = 0.0985636711
```

Houve exatamente uma divergência de top-1 entre Keras e hls4ml C++ no conjunto de 10.000 imagens.

A validação física posteriormente mostrou que a FPGA reproduziu **bit a bit** a referência fixa do hls4ml C++.

---

# 8. Estimativa de latência do HLS

O Vitis HLS estimou:

```text
9412 – 9460 ciclos
```

A 100 MHz:

```text
94,12 – 94,60 µs
```

Esse número é uma **estimativa do HLS para o núcleo**, não uma medição física do caminho software/DMA/FPGA.

Ele não deve ser comparado diretamente com a latência E2E sem explicar que as fronteiras são diferentes.

---

# 9. Integração Vivado

Fluxo físico:

```text
ARM / DDR
   ↕
AXI DMA
   ↕
AXI4-Stream
   ↕
lenet_dma_wrapper
   ↕
LeNet hls4ml
```

Blocos principais:

```text
Zynq UltraScale+ PS
AXI DMA
SmartConnect de controle
SmartConnect de memória
proc_sys_reset
lenet_dma_wrapper
IP LeNet hls4ml
```

DDR é acessada através de:

```text
S_AXI_HPC0_FPD
```

O IP LeNet **não possui AXI4-Lite de controle**. Isso é importante porque, no PYNQ, o `ip_dict` mostrou essencialmente:

```text
axi_dma_0
zynq_ultra_ps_e_0
```

Isso é esperado e não significa que a LeNet desapareceu do bitstream.

O `ap_start` da LeNet foi amarrado a `1` no wrapper:

```text
accel_ap_start <= '1'
```

Assim, a execução do software é controlada pelas transações DMA, e não por registradores AXI-Lite da rede.

---

# 10. AXI DMA congelado

Instância:

```text
axi_dma_0
```

Configuração geral:

```text
Scatter Gather       = OFF
Micro DMA            = OFF
Address width        = 32 bits
Buffer length width  = 26 bits
```

Controle AXI-Lite:

```text
Base  = 0xA0000000
Range = 0x00010000
```

MM2S:

```text
Memory-map width = 128 bits
AXIS width       = 32 bits
Max burst        = 64
DRE              = ON
```

S2MM:

```text
Memory-map width = 512 bits
AXIS width       = 512 bits
Max burst        = 64
DRE              = ON
```

Somente a região DDR baixa é necessária para os acessos de memória DMA.

---

# 11. Formato de entrada e saída física

## 11.1 Entrada

MNIST possui:

```text
784 pixels
```

O buffer do PYNQ:

```text
shape = (784,)
dtype = uint32
```

Tamanho:

```text
784 × 4 = 3136 bytes
```

Cada pixel é normalizado:

```python
pixels = image.astype(np.float32) / 255.0
pixels = pixels.reshape(-1)
```

Ordem:

```text
row-major
```

Depois é convertido para Q22.12 e enviado em words de 32 bits, com os 22 bits úteis na região inferior.

## 11.2 Saída

A LeNet produz:

```text
10 logits × 32 bits = 320 bits
```

O wrapper adapta a saída para a interface DMA S2MM de 512 bits:

```text
512 bits = 16 × uint32 = 64 bytes
```

Mapeamento:

```text
words 0–9   = 10 logits
words 10–15 = padding zero
```

TLAST:

```text
m_axis_tlast <= accel_output_valid
```

Há exatamente **um beat de saída por inferência**.

---

# 12. Regra crítica do DMA

A ordem obrigatória usada na campanha foi:

```text
1. preencher input_buffer
2. limpar output_buffer
3. flush
4. armar recvchannel / S2MM
5. iniciar sendchannel / MM2S
6. wait send
7. wait recv
8. invalidate output
9. conferir padding
10. decodificar Q22.12
11. argmax
```

Regra principal:

```text
RECV antes de SEND
```

Motivo: a LeNet pode produzir o resultado antes de o S2MM estar pronto. Armar o S2MM primeiro evita perder a saída ou travar a transação.

---

# 13. Síntese, recursos e timing

## 13.1 Recurso do IP isolado

Resultados anteriores do IP LeNet isolado:

| Recurso | Usado | Disponível | Uso |
|---|---:|---:|---:|
| LUT | 141.417 | 230.400 | 61,38% |
| FF | 81.051 | 460.800 | 17,59% |
| DSP48E2 | 746 | 1.728 | 43,17% |
| BRAM Tile | 114 | 312 | 36,54% |
| URAM | 0 | 96 | 0% |
| CARRY8 | 15.933 | 28.800 | 55,32% |

## 13.2 Sistema completo pós-route

Sistema completo inclui PS + DMA + SmartConnect + reset + wrapper + LeNet:

| Recurso | Usado | Disponível | Uso |
|---|---:|---:|---:|
| CLB LUTs | 154.485 | 230.400 | 67,05% |
| LUT como lógica | 151.780 | 230.400 | 65,88% |
| LUT como memória | 2.705 | 101.760 | 2,66% |
| CLB Registers | 96.655 | 460.800 | 20,98% |
| CARRY8 | 16.075 | 28.800 | 55,82% |
| DSP48E2 | 746 | 1.728 | 43,17% |
| RAMB36E2 | 122 | — | — |
| RAMB18E2 | 6 | — | — |
| BRAM equivalentes | 125 | 312 | 40,06% |
| URAM | 0 | 96 | 0% |

Ocupação física de CLBs:

```text
CLBs usados       = 28.154
CLBs disponíveis  = 28.800
ocupação          = 97,76%
```

Esse número é extremamente importante. Apesar de as LUTs aparecerem como ~67%, o design está quase no limite físico de CLBs.

## 13.3 Congestionamento

Durante o route:

```text
congestionamento estimado = nível 6
```

O router inicialmente indicou overlaps, mas o rip-up/reroute convergiu.

Final:

```text
Failed nets           = 0
Unrouted nets         = 0
Partially routed nets = 0
Node overlaps         = 0

Routable nets         = 280.922
Fully routed          = 280.922
Routing errors        = 0
```

## 13.4 Timing

Alvo:

```text
100 MHz
10 ns
```

Signoff pós-route:

```text
WNS setup = +0.033 ns
WHS hold  = +0.010 ns
```

Também foi observado:

```text
WNS = +0.030 ns
TNS = 0.000 ns
WHS = +0.010 ns
THS = 0.000 ns
```

Resultado:

```text
setup timing = PASS
hold timing  = PASS
```

DRC:

```text
Errors            = 0
Critical Warnings = 0
```

---

# 14. Problemas e decisões importantes no Vivado

## 14.1 Estimativa de recursos do HLS versus Vivado

Em etapa anterior o HLS chegou a estimar aproximadamente **121% de LUT**, sugerindo que a configuração talvez não coubesse. No entanto, a síntese Vivado do IP isolado mediu **61,38% de LUT**, e o sistema completo pós-route ficou em **67,05% de LUT**.

Conclusão: as estimativas de recurso do HLS não devem ser interpretadas isoladamente como utilização física final.

Ao mesmo tempo, a ocupação física de CLB chegou a **97,76%**, mostrando que olhar apenas LUT também é insuficiente.

## 14.2 Warnings HPC0

Foram vistos warnings de:

```text
AWUSER_WIDTH mismatch
ARUSER_WIDTH mismatch
```

entre:

```text
zynq_ultra_ps_e_0/S_AXI_HPC0_FPD
axi_smc_mem/M00_AXI
```

Eles não impediram:

```text
Validate
Synthesis
Placement
Routing
Timing closure
Bitstream
```

Portanto, não houve motivo para modificar a arquitetura apenas para eliminar esses warnings.

## 14.3 Segmentos DDR

QSPI e OCM foram deliberadamente excluídos das regiões que o DMA poderia acessar:

```text
HPC0_QSPI
HPC0_LPS_OCM
```

Foi utilizada a região:

```text
HPC0_DDR_LOW
```

Warnings sobre regiões excluídas eram esperados.

## 14.4 Reset

O reset do PS:

```text
pl_resetn0
```

é ativo em nível baixo.

A conexão correta foi direta ao:

```text
rst_ps8_0_100M/ext_reset_in
```

O Vivado propagou:

```text
pl_resetn0 POLARITY = ACTIVE_LOW
ext_reset_in        = ACTIVE_LOW
C_EXT_RESET_HIGH    = 0
```

Portanto, **não foi utilizado inversor externo**.

## 14.5 Wrapper necessário

A rede hls4ml não podia ser simplesmente conectada ao DMA sem adaptação. Foi necessário um wrapper específico para:

- adaptar sinais AXI4-Stream;
- adaptar saída de 320 bits para 512 bits;
- inserir padding;
- gerar TLAST;
- manter `ap_start=1`;
- conectar TVALID/TREADY corretamente.

---

# 15. Artefatos do overlay e hashes

Bitstream:

```text
SHA256 lenet_zcu104.bit
a48000cf00fed7387ac832febcab08a98fc0b8833d216db1481af29a3717463c
```

HWH:

```text
SHA256 lenet_zcu104.hwh
d6c4110c5db2d8e6c3d534fe129f66af84e1d613e7a8541ac7ee7a5543e27611
```

Esses hashes identificam exatamente o hardware usado na campanha.

---

# 16. Arquivos golden e hashes

Arquivos levados para a placa:

```text
mnist_test_uint8.npz
cpp_predictions_10000.npz
cpp_validation.json
hls_config.json
build_manifest.json
```

Hashes:

```text
mnist_test_uint8.npz
bea672a7825e79be670e076977f43a8679a382cab2b84ef2a0a15cddda799d81

cpp_predictions_10000.npz
0fbf3efd5a909e74ffd088d931e4905eb37c72ee85ddc5adbf323482d4b4db27

cpp_validation.json
06ba632205e7f9997e4a5fa791baac66cbcf8e2984a394a2f567cde16da21314

hls_config.json
374bba0548a7ead18cf3b7939d982d5ddc97fcdbe98b1ea3d740243cea3bbe4b

build_manifest.json
e323f8ab756f38341d33055d691ab4e54af9ad866e700f592858c8ff804b34bd
```

Dataset carregado na placa:

```text
IMAGES = (10000, 28, 28), uint8
LABELS = (10000,)
TEST_INDICES = 0..9999
```

---

# 17. Smoke test físico

O primeiro teste físico foi feito com a imagem de índice:

```text
index = 0
true label = 7
```

Resultados:

```text
Keras = 7
HLS   = 7
FPGA  = 7
```

Logits da FPGA:

```text
0  = -4.422851562
1  = -1.685546875
2  = +0.302734375
3  = +0.307617188
4  = -8.172851562
5  = -3.560546875
6  = -14.952148438
7  = +11.027343750
8  = -1.573242188
9  = +0.274414062
```

Padding:

```text
[0, 0, 0, 0, 0, 0]
```

Comparação FPGA × hls4ml C++:

```text
MAE     = 0.0
Max abs = 0.0
```

Os 10 logits físicos foram exatamente iguais à referência HLS fixa.

Status DMA sem erros.

Resultado:

```text
SMOKE PASS = True
```

Arquivo:

```text
results_zcu104/validation/smoke_test/smoke_index_00000.json
```

---

# 18. Validação intermediária de 100 imagens

Resultados:

```text
FPGA accuracy        = 99,0000%
HLS accuracy         = 99,0000%
Keras accuracy       = 99,0000%

FPGA/HLS agreement   = 100,0000%
FPGA/Keras agreement = 100,0000%

FPGA × HLS logit MAE = 0.0
FPGA × HLS max abs   = 0.0

10 logits exatamente iguais = 100/100
padding errors               = 0
DMA errors                   = 0

FPGA != HLS indices   = []
FPGA != Keras indices = []
FPGA incorretas       = [18]
```

Hashes:

```text
validation_100.csv
811a54fba6fb7ea663db594333420269b6468c202f64bc7694972f816918e498

validation_100_full.npz
213fb1b0cabcc9f03a72ec1fb4acf0d23b8324bffc9ae4e4b3b860c4a7b000d2

validation_100_summary.json
76a7ffa85d9ca8da069c1ab6e04b044613595fa9984ce747e8c1b18068ce3106
```

---

# 19. Validação oficial de acurácia — 10.000 imagens

Foi usado o conjunto completo de teste do MNIST:

```text
n_unique_images = 10000
batch           = 1
preprocessing   = uint8 → float32 / 255
ordem           = a mesma usada na referência C++
```

Nenhuma repetição foi contada como uma nova observação de acurácia.

Resultados:

```text
FPGA correct = 9899 / 10000
FPGA accuracy = 98,9900%
HLS accuracy  = 98,9900%
Keras accuracy = 98,9800%
```

IC95% Wilson para acurácia FPGA:

```text
[98,7743478501%; 99,1680279896%]
```

Concordância:

```text
FPGA/HLS top-1   = 100,0000%
FPGA/Keras top-1 = 99,9900%
```

Divergências:

```text
FPGA != HLS   = 0
FPGA != Keras = 1
índice FPGA != Keras = 9530

erros de classificação FPGA = 101
```

Integridade numérica FPGA × hls4ml C++:

```text
MAE dos logits                   = 0.0
Max absolute error               = 0.0
10 logits exatamente iguais      = 10000 / 10000
RAW Q22.12 exatamente igual      = 10000 / 10000 imagens
RAW word mismatches              = 0 / 100000 words
padding errors                   = 0
```

Essa é uma das conclusões mais fortes da campanha: **a FPGA reproduziu exatamente a referência fixa do hls4ml C++ nas 10.000 imagens**.

## 19.1 Métricas de classificação

```text
Macro precision = 0.9898047147900856
Macro recall    = 0.9898530752010318
Macro F1        = 0.9898171454416245
Cohen Kappa     = 0.9887734745258243
MCC             = 0.9887761452506671
```

Por classe:

| Classe | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 | 0.989827 | 0.992857 | 0.991340 |
| 1 | 0.993849 | 0.996476 | 0.995161 |
| 2 | 0.990338 | 0.993217 | 0.991776 |
| 3 | 0.993056 | 0.991089 | 0.992071 |
| 4 | 0.990891 | 0.996945 | 0.993909 |
| 5 | 0.985491 | 0.989910 | 0.987696 |
| 6 | 0.990615 | 0.991649 | 0.991132 |
| 7 | 0.995045 | 0.976654 | 0.985763 |
| 8 | 0.988660 | 0.984600 | 0.986626 |
| 9 | 0.980276 | 0.985134 | 0.982699 |

Matriz de confusão, linhas = classe verdadeira, colunas = classe prevista:

```text
[[ 973    0    0    0    1    0    2    0    3    1]
 [   0 1131    0    1    0    1    1    1    0    0]
 [   1    1 1025    0    1    0    2    1    0    1]
 [   1    0    1 1001    0    3    0    0    4    0]
 [   0    0    0    0  979    0    2    0    0    1]
 [   2    0    0    4    0  883    1    0    1    1]
 [   2    1    0    0    1    3  950    0    1    0]
 [   0    4    7    0    0    0    0 1004    1   12]
 [   3    0    2    1    0    3    0    2  959    4]
 [   1    1    0    1    6    3    1    1    1  994]]
```

Soma = 10000.

Arquivos e hashes:

```text
validation_10000.csv
06b528406fbf27c472c4003b89403a44045f4f7d14396cdce589762a6199de80

validation_10000_full.npz
2955660805553d4122475c4457fbdd408eb6621b4cf98a848440534ee31f30a9

validation_10000_summary.json
0ebc1f597b8aab5c44ad2e7b499272b33e997379f012cf61890d39adfbdc8226
```

---

# 20. Determinismo / integridade

O teste de determinismo foi separado da acurácia.

Protocolo:

```text
100 imagens únicas
30 repetições por imagem
total = 3000 inferências físicas
```

As 3000 execuções **não adicionaram 3000 observações de acurácia**. Serviram exclusivamente para verificar determinismo/integridade.

Resultado:

```text
execution failures = 0
RAW mismatches     = 0
class mismatches   = 0
padding errors     = 0
```

Arquivo:

```text
determinism_3000.json
```

SHA256:

```text
275b6fa8bf2f552fc18fcc5e52da00ac1b70a6669cd296be08523aa345115273
```

---

# 21. Latência inference-only / DMA+FPGA

## 21.1 Fronteira

Buffers preparados antes da região cronometrada.

Incluído:

```text
PYNQ DMA transfer setup
recv transfer / S2MM
send transfer / MM2S
FPGA LeNet
wait send
wait recv
```

Excluído:

```text
uint8 → float
/255
conversão Q22.12
cópia para input_buffer
input flush
output invalidate
decode dos logits
argmax
```

Portanto, essa métrica deve ser chamada:

```text
latência DMA+FPGA
ou
latência inference-only software-visible
```

Ela **não deve** ser chamada de “latência pura do núcleo HLS”.

## 21.2 Protocolo

```text
warm-ups = 200
campanhas = 5
medições/campanha = 20000
total = 100000 latências individuais
outliers removidos = não
```

## 21.3 Resultado global

```text
mean       = 0.3699561455 ms
median     = 0.3701600000 ms
std        = 0.0064110471 ms
CV         = 1.7329208%
min        = 0.3432300 ms
max        = 0.6692300 ms
p90        = 0.3718700 ms
p95        = 0.3818400 ms
p99        = 0.3857901 ms
```

O máximo de 0,66923 ms foi preservado; não houve remoção pós-hoc.

## 21.4 Estatística das médias das cinco campanhas

```text
mean       = 0.3699561455 ms
median     = 0.3700818300 ms
std        = 0.0003753357 ms
CV         = 0.1014541%
min        = 0.3694938005 ms
max        = 0.3703592825 ms

IC95% t         = [0.3694901046; 0.3704221864] ms
IC95% bootstrap = [0.3696654432; 0.3702455928] ms
```

Comparação apenas de referência com a previsão HLS:

```text
HLS estimado = 94,12–94,60 µs
medido DMA+FPGA ≈ 369,96 µs
```

A diferença não deve ser chamada integralmente de “overhead da CPU”, pois as fronteiras e a natureza das medições são diferentes.

Arquivos:

```text
latency_dma_fpga/LATENCY_DMA_FINAL.json
SHA256 = a3deeaaf09f01786ea6daf7b9190b3fdff77e872da9c04271dc67dfd757565dd

latency_dma_fpga/latency_all_100000_ns.npy
SHA256 = eace461da29788f87421b295b60078b969dad1b30ec836f8848fef8f7c2f8438
```

---

# 22. Latência End-to-End

## 22.1 Fronteira

Começa com a imagem MNIST `uint8` já na RAM e termina na classe prevista:

```text
MNIST uint8 em RAM
  ↓
float32
  ↓
/255
  ↓
Q22.12
  ↓
cópia PynqBuffer
  ↓
flush
  ↓
DMA MM2S
  ↓
FPGA LeNet
  ↓
DMA S2MM
  ↓
invalidate
  ↓
decode Q22.12
  ↓
argmax
  ↓
classe
```

Não inclui leitura do dataset a partir do disco.

## 22.2 Protocolo

```text
warm-ups = 200
campanhas = 5
medições/campanha = 20000
total = 100000
class mismatches = 0
outliers removidos = não
```

## 22.3 Resultado global

```text
mean       = 1.0969730981 ms
median     = 1.0945800000 ms
std        = 0.0282690469 ms
CV         = 2.5770046%
min        = 1.0640800 ms
max        = 5.1926000 ms
p90        = 1.1096900 ms
p95        = 1.1159905 ms
p99        = 1.1701901 ms
```

O máximo de 5,1926 ms foi mantido no conjunto.

## 22.4 Estatística das médias de campanha

```text
mean       = 1.0969730981 ms
median     = 1.0998920460 ms
std        = 0.0080097988 ms
CV         = 0.7301728%
min        = 1.0829386435 ms
max        = 1.1029351120 ms

IC95% t         = [1.0870276192; 1.1069185770] ms
IC95% bootstrap = [1.0898668075; 1.1015496814] ms
```

Comparação E2E vs DMA+FPGA:

```text
diferença média = 0.7270169526 ms
razão E2E / DMA+FPGA = 2.9651436×
```

Arquivos:

```text
latency_end_to_end/LATENCY_E2E_FINAL.json
SHA256 = 246225c95f2383e24795f0ddbcf1d09548aad94529d34ba54e43b6b659915651

latency_end_to_end/latency_e2e_all_100000_ns.npy
SHA256 = cddb63923af9639a660615a640065d629d1212e9b78ad3a16bff4a7d6132555d
```

---

# 23. Throughput DMA+FPGA / inference-only

O throughput foi deliberadamente medido com **um único cronômetro ao redor da janela inteira**, sem chamar `perf_counter()` dentro de cada inferência.

Protocolo:

```text
warm-ups = 200
campanhas = 5
inferências/campanha = 100000
total = 500000 inferências temporizadas
batch = 1
```

FPS das campanhas:

```text
2781.379724784601
2779.643267409865
2789.290401127387
2818.284139625656
2789.495400344129
```

Estatísticas:

```text
mean       = 2791.618586658327 FPS
median     = 2789.290401127387 FPS
std        = 15.566170423522 FPS
CV         = 0.5576037679%
min        = 2779.643267409865 FPS
max        = 2818.284139625656 FPS
p90        = 2806.768643913045 FPS
p95        = 2812.526391769351 FPS
p99        = 2817.132590054395 FPS

IC95% t         = [2772.290633253536; 2810.946540063119] FPS
IC95% bootstrap = [2782.267277103264; 2805.145508801139] FPS
```

Outliers removidos:

```text
não
```

Arquivo:

```text
throughput_dma_fpga/THROUGHPUT_DMA_FINAL.json
SHA256 = b6ba336a23e717f34ece23fc3f075889c66f0109f034fa7296d6139611a9a7e9
```

---

# 24. Throughput End-to-End

Fronteira igual à latência E2E.

Protocolo:

```text
warm-ups = 200
campanhas = 5
inferências/campanha = 20000
total = 100000
cronômetro = um único timer por campanha
```

FPS:

```text
909.407333652351
905.868354477565
903.172196305999
914.004548937574
903.304621761304
```

Estatísticas:

```text
mean       = 907.151411026959 FPS
median     = 905.868354477565 FPS
std        = 4.591646155587 FPS
CV         = 0.5061609451%
min        = 903.172196305999 FPS
max        = 914.004548937574 FPS
p90        = 912.165662823485 FPS
p95        = 913.085105880529 FPS
p99        = 913.820660326165 FPS

IC95% t         = [901.450129303076; 912.852692750842] FPS
IC95% bootstrap = [903.764398122434; 910.918635354214] FPS
```

Arquivo:

```text
throughput_end_to_end/THROUGHPUT_E2E_FINAL.json
SHA256 = a0c01fa4ead68b4f0993e99664220cb75fd715ed803b8d1e12cc043af7936666
```

---

# 25. Throughput sustentado de aproximadamente 10 s

Antes da potência, foram feitas janelas sustentadas para conferir estabilidade.

## 25.1 DMA+FPGA

```text
8 janelas
28000 inferências/janela
≈ 10 s/janela
```

FPS:

```text
2776.744
2804.149
2775.730
2776.644
2775.339
2792.208
2774.370
2776.897
```

Estatísticas:

```text
mean   = 2781.510097799254 FPS
median = 2776.693814433361 FPS
std    = 10.804174308116 FPS
CV     = 0.3884283691%
min    = 2774.369918470869 FPS
max    = 2804.149178961695 FPS
```

Shift em relação ao throughput principal:

```text
-0.3621013597%
```

SHA256:

```text
SUSTAINED_DMA_FINAL.json
21ffad817ee2bec0c3642b10a4abcd40b989a8955f43b877a3e25cc044f262f4
```

## 25.2 E2E

```text
8 janelas
9000 inferências/janela
≈ 10 s/janela
```

FPS:

```text
904.294
903.946
903.491
903.453
903.348
903.374
904.010
901.311
```

Estatísticas:

```text
mean   = 903.403386635116 FPS
median = 903.472343659624 FPS
std    = 0.914070271440 FPS
CV     = 0.1011807444%
min    = 901.310644478912 FPS
max    = 904.294024803782 FPS
```

Shift em relação ao throughput principal:

```text
-0.4131641473%
```

SHA256:

```text
SUSTAINED_E2E_FINAL.json
6f22a9031e299fdc3d542b308439725cd1d2927b536a09154d72d0e37ea0c872
```

Conclusão: o desempenho permaneceu estável em janelas sustentadas.

---

# 26. Sensor físico de potência

Foi usado:

```python
from pynq import get_rails
RAILS = get_rails()
RAILS["12V"].power.value
```

A grandeza experimental principal é:

```text
rail 12V → power.value
```

IMPORTANTE: o atributo de `voltage` do objeto `12V` mostrou valor estranho (`0.004`) e não foi usado para reconstruir potência por `V × I`. O próprio sensor de potência do rail foi utilizado diretamente.

Exemplo observado:

```text
Rail: 12V
power = 10.787 W
```

Teste simples idle, 10 amostras a ~1 Hz:

```text
10.837
10.312
10.312
10.325
10.300
10.312
10.312
10.300
10.300
10.300 W
```

Resumo:

```text
mean = 10.361 W
min  = 10.300 W
max  = 10.837 W
mean sampling interval ≈ 1.001788 s
```

Essa coleta foi **somente calibração** e não entrou na energia oficial.

A primeira amostra mais alta não foi apagada; simplesmente não pertence à coleta oficial.

---

# 27. Problema da telemetria e como foi resolvido

Esse foi um dos pontos metodológicos mais importantes.

## 27.1 Sampler em processo separado

Foi usado `multiprocessing` com `fork`, para que leitura PMBus e escrita CSV ocorressem em processo separado do workload principal.

Teste do sampler a 1 Hz:

```text
7 amostras
power mean = 10.3515714286 W
interval mean = 0.9999837617 s
interval min  = 0.9999070400 s
interval max  = 1.0000639700 s
```

Ou seja, a periodicidade do sampler foi excelente.

## 27.2 Primeiro teste: sem telemetria vs telemetria a 1 Hz

DMA+FPGA:

```text
delta médio = +0.6689616%
IC95% = [+0.1974883%; +1.1404350%]
inclui zero? não
```

E2E:

```text
delta médio = -0.1821291%
IC95% = [-0.4845432%; +0.1202851%]
inclui zero? sim
```

No DMA, o sampler parecia até **aumentar** o throughput, o que sugeria que o teste estava capturando efeito de escalonamento/processo e não apenas custo PMBus.

Arquivo:

```text
TELEMETRY_VALIDATION.json
SHA256 = 701153a3756cd49e0b75fcbee58da07ec673087fac3b3fa4630750004f1ac68c
```

## 27.3 Segundo teste: reduzir para 0,5 Hz

Sampler a cada 2 s:

```text
interval mean = 1.9999802760 s
min = 1.9999504300 s
max = 2.0000246100 s
```

Mesmo assim:

DMA:

```text
delta médio = +0.8823696%
IC95% = [+0.3461323%; +1.4186069%]
inclui zero? não
```

E2E:

```text
delta médio = +0.3756139%
IC95% = [+0.1974034%; +0.5538244%]
inclui zero? não
```

Portanto, simplesmente reduzir a frequência de leitura não eliminou o aparente efeito.

Arquivo:

```text
TELEMETRY_VALIDATION_2S.json
SHA256 = 7e16254c5640aeab4faa71c9a52dcd419243a139e20c93b3dbfebdb1813d793f
```

## 27.4 Teste-controle final: processo filho controle vs PMBus real

Foi criado um processo filho controle com:

- mesmo `multiprocessing`;
- mesmo ciclo temporal;
- escrita CSV;
- `flush()`;
- mesma frequência de 2 s;
- **sem chamar `get_rails()`**.

A comparação passou a ser:

```text
processo controle
vs
processo controle + leitura real PMBus
```

DMA+FPGA:

```text
mean delta = -0.0498906%
std        = 0.3998403%
IC95%      = [-0.5463578%; +0.4465767%]
inclui zero = sim
max |delta de um par| = 0.7553416%
```

E2E:

```text
mean delta = -0.1539077%
std        = 0.2606754%
IC95%      = [-0.4775789%; +0.1697635%]
inclui zero = sim
max |delta de um par| = 0.4925335%
```

Resultado:

```text
DMA+FPGA: True
E2E: True
LEITURA PMBus APROVADA: True
```

Conclusão: o efeito específico da leitura PMBus foi compatível com zero. O estranho comportamento dos testes anteriores estava associado à comparação “sem processo” versus “com processo”, e não necessariamente à leitura PMBus.

Arquivo:

```text
PMBUS_VS_CONTROL.json
SHA256 = 24b59585e220ee6157fbd2e9ca0d987d1afa07f4712480302b1cf8b8a990833b
```

---

# 28. Metodologia oficial de potência e energia

Rail:

```text
12V total board input power
```

Sampler:

```text
intervalo = 2.0 s
frequência = 0.5 Hz
processo separado
```

Por cenário:

```text
5 repetições
idle pareado antes da carga
idle ≈ 10 s
carga ≈ 30 s para inference-only e E2E
```

Cálculo primário de potência:

```text
média aritmética das amostras PMBus
```

Cálculo suplementar:

```text
média por integração trapezoidal
```

Potência dinâmica:

```text
P_dynamic = P_active - P_idle
```

Sem clipping de valores.

Energia total por inferência:

```text
E_total = P_active / FPS_mesma_janela
```

Energia dinâmica:

```text
E_dynamic = P_dynamic / FPS_mesma_janela
```

O FPS usado para energia é o **FPS medido dentro da mesma janela ativa da telemetria**, não o throughput de outra campanha.

Isso evita misturar condições experimentais.

---

# 29. Potência e energia oficiais — inference-only

Protocolo:

```text
5 repetições
84000 inferências por janela ativa
~30 s por carga
idle pareado ~10 s
```

Resultados individuais resumidos:

```text
Rep 1:
idle   = 10.3472 W
active = 10.5980 W
FPS    = 2782.804
Pdyn   = 0.2508 W
Etotal = 3.8084 mJ/inf
Edyn   = 0.0901 mJ/inf

Rep 2:
idle   = 10.3048 W
active = 10.6148 W
FPS    = 2793.433
Pdyn   = 0.3100 W
Etotal = 3.7999 mJ/inf
Edyn   = 0.1110 mJ/inf

Rep 3:
idle   = 10.3072 W
active = 10.5989 W
FPS    = 2784.447
Pdyn   = 0.2917 W
Etotal = 3.8065 mJ/inf
Edyn   = 0.1047 mJ/inf

Rep 4:
idle   = 10.3048 W
active = 10.6048 W
FPS    = 2769.416
Pdyn   = 0.3000 W
Etotal = 3.8293 mJ/inf
Edyn   = 0.1083 mJ/inf

Rep 5:
idle   = 10.3672 W
active = 10.5997 W
FPS    = 2798.945
Pdyn   = 0.2325 W
Etotal = 3.7870 mJ/inf
Edyn   = 0.0831 mJ/inf
```

Resultado consolidado:

```text
Idle power mean             = 10.326240 W
Active total power mean     = 10.603240 W
Dynamic power mean          = 0.277000 W
Same-window throughput mean = 2785.808910 FPS
Total energy mean           = 3.806211 mJ/inference
Dynamic energy mean         = 0.099451 mJ/inference

IC95% throughput =
[2771.784358; 2799.833462] FPS

IC95% dynamic power =
[0.235367969; 0.318632031] W

IC95% dynamic energy =
[0.084311410; 0.114589799] mJ/inference
```

---

# 30. Potência e energia oficiais — End-to-End

Protocolo:

```text
5 repetições
27000 inferências por janela
~30 s
idle pareado ~10 s
```

Resultados individuais:

```text
Rep 1:
idle   = 10.3122 W
active = 10.5434 W
FPS    = 911.265
Pdyn   = 0.2312 W
Etotal = 11.5701 mJ/inf
Edyn   = 0.2537 mJ/inf

Rep 2:
idle   = 10.3048 W
active = 10.5417 W
FPS    = 911.973
Pdyn   = 0.2369 W
Etotal = 11.5592 mJ/inf
Edyn   = 0.2598 mJ/inf

Rep 3:
idle   = 10.3022 W
active = 10.5399 W
FPS    = 915.212
Pdyn   = 0.2377 W
Etotal = 11.5163 mJ/inf
Edyn   = 0.2597 mJ/inf

Rep 4:
idle   = 10.3148 W
active = 10.5400 W
FPS    = 914.608
Pdyn   = 0.2252 W
Etotal = 11.5241 mJ/inf
Edyn   = 0.2462 mJ/inf

Rep 5:
idle   = 10.3022 W
active = 10.5559 W
FPS    = 913.408
Pdyn   = 0.2537 W
Etotal = 11.5566 mJ/inf
Edyn   = 0.2778 mJ/inf
```

Consolidado:

```text
Idle power mean             = 10.307240 W
Active total power mean     = 10.544186 W
Dynamic power mean          = 0.236946 W
Same-window throughput mean = 913.293127 FPS
Total energy mean           = 11.545270 mJ/inference
Dynamic energy mean         = 0.259442 mJ/inference

IC95% throughput =
[911.208228; 915.378025] FPS

IC95% dynamic power =
[0.223734693; 0.250156735] W

IC95% dynamic energy =
[0.244966067; 0.273917635] mJ/inference
```

---

# 31. Teste separado de carga máxima sustentada

Foi feita uma terceira campanha física independente para responder à pergunta:

```text
inference-only normal
vs
máxima carga sustentada
vs
end-to-end
```

Nome técnico preferido:

```text
maximum sustained serial load
ou
saturated sustained serial
```

Não afirmar que isso é:

```text
pure HLS kernel saturation
```

Motivo: o deployment atual usa AXI DMA em simple mode e uma inferência/frame por transação controlada pelo host.

## 31.1 Protocolo

```text
5 repetições
idle pareado = 10 s
168000 inferências/carga
duração estimada ≈ 60,3 s
warm-up = 500
sampler = 0,5 Hz
~30 amostras de potência por janela ativa
```

Resultados individuais:

```text
Rep 1:
idle   = 10.3422 W
active = 10.5972 W
FPS    = 2780.974
Pdyn   = 0.2550 W
Etotal = 3.8106 mJ/inf
Edyn   = 0.0917 mJ/inf

Rep 2:
idle   = 10.3046 W
active = 10.6047 W
FPS    = 2799.466
Pdyn   = 0.3001 W
Etotal = 3.7881 mJ/inf
Edyn   = 0.1072 mJ/inf

Rep 3:
idle   = 10.3048 W
active = 10.5963 W
FPS    = 2793.441
Pdyn   = 0.2915 W
Etotal = 3.7933 mJ/inf
Edyn   = 0.1044 mJ/inf

Rep 4:
idle   = 10.3148 W
active = 10.6036 W
FPS    = 2773.225
Pdyn   = 0.2888 W
Etotal = 3.8236 mJ/inf
Edyn   = 0.1041 mJ/inf

Rep 5:
idle   = 10.3048 W
active = 10.6027 W
FPS    = 2776.902
Pdyn   = 0.2979 W
Etotal = 3.8182 mJ/inf
Edyn   = 0.1073 mJ/inf
```

## 31.2 Estatísticas completas do cenário sustentado

Idle:

```text
mean       = 10.314240 W
median     = 10.304800 W
std        = 0.016226768 W
CV         = 0.157324%
min        = 10.304600 W
max        = 10.342200 W
p90        = 10.331240 W
p95        = 10.336720 W
p99        = 10.341104 W
IC95% t    = [10.294091806; 10.334388194] W
bootstrap  = [10.304720; 10.329200] W
```

Potência total ativa:

```text
mean       = 10.600906667 W
median     = 10.602700000 W
std        = 0.003859721 W
CV         = 0.036409%
min        = 10.596333333 W
max        = 10.604733333 W
p90        = 10.604266667 W
p95        = 10.604500000 W
p99        = 10.604686667 W
IC95% t    = [10.596114190; 10.605699143] W
bootstrap  = [10.597953333; 10.603860000] W
```

Potência dinâmica:

```text
mean       = 0.286666667 W
median     = 0.291533333 W
std        = 0.018292151 W
CV         = 6.380983%
min        = 0.255000000 W
max        = 0.300133333 W
p90        = 0.299240000 W
p95        = 0.299686667 W
p99        = 0.300044000 W
IC95% t    = [0.263953962; 0.309379372] W
bootstrap  = [0.270780000; 0.297520000] W
```

Throughput da mesma janela:

```text
mean       = 2784.801482 FPS
median     = 2780.974205 FPS
std        = 11.188866 FPS
CV         = 0.401783%
min        = 2773.224505 FPS
max        = 2799.465644 FPS
p90        = 2797.055891 FPS
p95        = 2798.260767 FPS
p99        = 2799.224668 FPS
IC95% t    = [2770.908670; 2798.694294] FPS
bootstrap  = [2776.245361; 2793.747997] FPS
```

Energia total:

```text
mean       = 3.806750505 mJ/inf
median     = 3.810607082 mJ/inf
std        = 0.015457028
CV         = 0.406043%
min        = 3.788127694
max        = 3.823551482
p90        = 3.821401319
p95        = 3.822476401
p99        = 3.823336465
IC95% t    = [3.787558070; 3.825942940]
bootstrap  = [3.794688570; 3.818812439]
```

Energia dinâmica:

```text
mean       = 0.102934689 mJ/inf
median     = 0.104363509 mJ/inf
std        = 0.006460528
CV         = 6.276337%
min        = 0.091694486
max        = 0.107277830
p90        = 0.107251073
p95        = 0.107264452
p99        = 0.107275155
IC95% t    = [0.094912883; 0.110956494]
bootstrap  = [0.097297594; 0.106668209]
```

Cálculo trapezoidal suplementar:

```text
Pactive mean = 10.600993102 W
Pdynamic mean = 0.284993007 W
Etotal mean = 3.806781540 mJ/inf
Edynamic mean = 0.102332931 mJ/inf
```

Arquivo:

```text
SATURATED_POWER_ENERGY_FINAL.json
SHA256 = 7bbf07d4b77df48b36ce644ca804df87168fc1710e7b0ea5d6b5f85638b2f11b
```

---

# 32. Comparação final dos três cenários

| Métrica | Inference-only | Maximum sustained serial | End-to-End |
|---|---:|---:|---:|
| Latência | 0,369956 ms | — | 1,096973 ms |
| Throughput da coleta energética | 2785,809 FPS | 2784,801 FPS | 913,293 FPS |
| Potência idle | 10,32624 W | 10,31424 W | 10,30724 W |
| Potência ativa total | 10,60324 W | 10,60091 W | 10,54419 W |
| Potência dinâmica | 0,27700 W | 0,28667 W | 0,23695 W |
| Energia total | 3,80621 mJ/inf | 3,80675 mJ/inf | 11,54527 mJ/inf |
| Energia dinâmica | 0,09945 mJ/inf | 0,10293 mJ/inf | 0,25944 mJ/inf |

Sustentado vs inference-only:

```text
throughput        = -0.0361628%
potência total    = -0.0220059%
potência dinâmica = +3.4897714%
energia dinâmica  = +3.5033319%
```

Os intervalos de confiança de potência/energia dinâmica se sobrepõem fortemente. Portanto, a leitura mais defensável é que **inference-only e carga máxima serial sustentada apresentaram comportamento praticamente equivalente**, especialmente em throughput e potência total.

Isso mostra que o workload inference-only já mantinha o deployment muito próximo do regime sustentado máximo alcançável com o protocolo atual.

## 32.1 End-to-End vs inference-only

A E2E:

- é aproximadamente 2,965× mais lenta em latência;
- possui throughput aproximadamente 67% menor;
- apresenta potência ativa total semelhante;
- apresenta energia total aproximadamente 3× maior;
- apresenta energia dinâmica aproximadamente 2,61× maior.

O ponto importante é que o aumento de energia E2E não ocorre por grande aumento de potência instantânea. O principal efeito é que cada inferência permanece muito mais tempo no sistema porque entram preprocessing, quantização, manipulação de buffers, cache e pós-processamento.

---

# 33. Arquivos finais de potência/energia e hashes

```text
FINAL_POWER_ENERGY.json
d880f56890cc186839803ea9a7bb1a92dfc7ba4b338a86f368bd81b4c2154e09

official_power_energy_replicates.csv
f9eb309dfb5c6a5a3f2a4a989044970a1f80800e651cf2aeebdbc343a69819fe

PMBUS_VS_CONTROL.json
24b59585e220ee6157fbd2e9ca0d987d1afa07f4712480302b1cf8b8a990833b

SATURATED_POWER_ENERGY_FINAL.json
7bbf07d4b77df48b36ce644ca804df87168fc1710e7b0ea5d6b5f85638b2f11b

THREE_SCENARIO_COMPARISON.json
1f7d3eb1f73b732792af088681adbbc588480c8168d904cec8cb5676776d41ed
```

Hash final do `METADATA.json` após inclusão do cenário sustentado:

```text
c415ead633332a05b461b0044744914d1a573e19b1e3b898c7f4125e2faba3b6
```

---

# 34. Resumo canônico para o TCC

Se for necessário consultar rapidamente os valores principais, usar esta tabela:

| Métrica | Resultado |
|---|---:|
| Acurácia física FPGA | **98,99%** |
| Acertos MNIST | **9899/10000** |
| Concordância FPGA × hls4ml C++ | **100,00%** |
| Logits FPGA × HLS MAE | **0,0** |
| RAW mismatches FPGA × HLS | **0 / 100000** |
| Determinismo | **3000/3000 sem falhas/mismatch** |
| Latência DMA+FPGA | **0,369956 ms** |
| IC95% latência DMA+FPGA | **[0,369490; 0,370422] ms** |
| Latência E2E | **1,096973 ms** |
| IC95% latência E2E | **[1,087028; 1,106919] ms** |
| Throughput DMA+FPGA principal | **2791,619 FPS** |
| IC95% throughput DMA | **[2772,291; 2810,947] FPS** |
| Throughput E2E principal | **907,151 FPS** |
| IC95% throughput E2E | **[901,450; 912,853] FPS** |
| Throughput inference-only na energia | **2785,809 FPS** |
| Potência ativa inference-only | **10,60324 W** |
| Potência dinâmica inference-only | **0,27700 W** |
| Energia total inference-only | **3,80621 mJ/inf** |
| Energia dinâmica inference-only | **0,09945 mJ/inf** |
| Throughput E2E na energia | **913,293 FPS** |
| Potência ativa E2E | **10,54419 W** |
| Potência dinâmica E2E | **0,23695 W** |
| Energia total E2E | **11,54527 mJ/inf** |
| Energia dinâmica E2E | **0,25944 mJ/inf** |
| Throughput max sustained serial | **2784,801 FPS** |
| Potência ativa max sustained | **10,60091 W** |
| Potência dinâmica max sustained | **0,28667 W** |
| Energia total max sustained | **3,80675 mJ/inf** |
| Energia dinâmica max sustained | **0,10293 mJ/inf** |
| Pós-route LUT | **154.485 / 230.400 = 67,05%** |
| Pós-route FF | **96.655 / 460.800 = 20,98%** |
| DSP | **746 / 1728 = 43,17%** |
| BRAM equivalente | **125 / 312 = 40,06%** |
| URAM | **0** |
| Ocupação física CLB | **28.154 / 28.800 = 97,76%** |
| WNS setup | **+0,033 ns** |
| WHS hold | **+0,010 ns** |
| Routing errors | **0** |
| DRC errors | **0** |
| DRC critical warnings | **0** |

---

# 35. Como interpretar os resultados no texto do TCC

## 35.1 Validação funcional

A implementação física da LeNet na ZCU104 apresentou 98,99% de acurácia no conjunto de 10.000 imagens MNIST, reproduzindo exatamente a acurácia da referência fixa gerada pelo hls4ml. Os logits e as palavras Q22.12 obtidas fisicamente na FPGA foram idênticos à referência C++ para todas as imagens, sem divergências de classe, erros de padding ou palavras RAW divergentes.

Isso é evidência direta de que:

- preprocessing usado na placa está correto;
- conversão Q22.12 está correta;
- packing/unpacking está correto;
- wrapper está correto;
- TLAST está correto;
- DMA está transportando os dados corretamente;
- aritmética física corresponde à referência fixa do hls4ml.

## 35.2 Latência

A latência DMA+FPGA de ~0,370 ms é uma métrica software-visible que inclui as transações PYNQ/DMA e o processamento da rede, mas exclui preprocessing e pós-processamento.

A latência E2E de ~1,097 ms inclui toda a cadeia da imagem uint8 já em RAM até a classe final.

A previsão HLS de 94,12–94,60 µs não é uma medição equivalente às duas anteriores.

## 35.3 Throughput

O throughput DMA+FPGA foi ~2792 FPS em uma campanha sem cronômetro por inferência. O throughput E2E ficou ~907 FPS.

O teste de carga máxima serial de ~60 s produziu ~2785 FPS, praticamente igual à coleta inference-only. Isso mostra que o caminho atual já trabalha próximo do limite sustentado do deployment simples DMA+FPGA.

## 35.4 Potência e energia

A potência total da placa ficou por volta de 10,5–10,6 W em todos os cenários. A energia por inferência E2E é maior porque o tempo necessário para concluir cada inferência aumenta substancialmente.

A energia dinâmica inference-only foi ~0,099 mJ/inferência, enquanto E2E foi ~0,259 mJ/inferência.

---

# 36. Problemas enfrentados — checklist consolidado

1. **RF global 64 não era definição adequada.**  
   Foi necessário estudar RF por camada e congelar `5/50/64/60/42`.

2. **Estimativa HLS de LUT parecia inviável (~121%).**  
   A síntese Vivado mostrou utilização muito menor de LUT; portanto estimativa HLS não podia ser usada como verdade física final.

3. **Mesmo com 67% de LUT, o placement ficou praticamente cheio.**  
   Ocupação física de CLBs = 97,76%, com congestionamento de route nível 6.

4. **Congestionamento e overlaps durante route.**  
   O router convergiu e terminou com 0 failed nets, 0 unrouted nets e 0 overlaps.

5. **Warnings AWUSER/ARUSER na HPC0.**  
   Foram tolerados porque não causaram falha de Validate, síntese, route, timing ou bitstream.

6. **Questão de polaridade do reset.**  
   `pl_resetn0` é active-low e foi ligado diretamente ao `ext_reset_in`; não usar inversor externo.

7. **IP hls4ml sem AXI-Lite de controle.**  
   Isso fez o PYNQ não listar a LeNet como IP controlável. É esperado; `ap_start` está preso em 1.

8. **Necessidade de wrapper DMA.**  
   O wrapper foi essencial para adaptação AXIS, largura 320→512, padding e TLAST.

9. **Risco de travamento/perda de saída DMA.**  
   Foi estabelecida a regra obrigatória `recv/S2MM antes de send/MM2S`.

10. **Relógio da ZCU104 incorreto.**  
    Produziu `run_20250504_164021` apesar de a campanha real ser de 10/09/2026. Não renomear o original sem preservar a relação; registrar a data real nos metadados.

11. **Telemetria parecia alterar o throughput.**  
    Testes diretos em 1 Hz e 0,5 Hz mostraram diferenças estatísticas.

12. **Reduzir a frequência de leitura não resolveu sozinho.**  
    O problema estava na comparação entre “nenhum processo” e “processo de telemetria”.

13. **Foi criado processo-controle.**  
    Com multiprocessing/CSV equivalentes dos dois lados, o custo específico da leitura PMBus passou a ter IC95% incluindo zero para DMA e E2E.

14. **Campo `voltage` do rail `12V` não era utilizável para `V×I`.**  
    Foi usada diretamente a leitura `RAILS["12V"].power.value`.

15. **Report Power do Vivado não foi usado como resultado experimental principal.**  
    Potência física da ZCU104 foi tratada como referência experimental. `report_power` serve apenas como informação suplementar.

16. **Termo “saturado” exige cuidado.**  
    A terceira campanha é máxima carga **serial sustentada do deployment atual**, não prova de saturação intrínseca do kernel HLS.

17. **Nenhum outlier foi removido.**  
    Máximos de latência e dispersões das campanhas foram preservados.

---

# 37. O que deve ser preservado no PC

Para considerar este experimento arquivado, preservar:

```text
[obrigatório]
- workspace inteiro da ZCU104
- notebook(s) executado(s)
- overlay .bit
- overlay .hwh
- golden/
- validation/
- benchmark_complete/
- METADATA.json
- todos os JSON de resumo
- todos os CSV RAW
- todos os NPY/NPZ
- arquivos de telemetria
- hashes
- este README
- versão TXT deste README

[PC/hls4ml]
- modelo Keras .h5
- configs hls4ml
- scripts
- build manifest
- build HLS relevante
- IP promovido
- wrapper VHDL
- relatórios HLS
- relatórios Vivado pós-route
- Tcl do block design
- Tcl do projeto
- bitstream/HWH de deployment

[recomendado]
- snapshot bruto do projeto Vivado LeNet-tcc3-08_09*
```

---

# 38. Cópia da ZCU104 para o PC — método recomendado

Executar **no PC**, não dentro da ZCU104.

Primeiro:

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
DEST="$BASE/coletas_zcu104_lenet/2026-09-10_lenet_mnist_hls4ml_zcu104"
BOARD="xilinx@192.168.2.99"
REMOTE="/home/xilinx/jupyter_notebooks/lenet_mnist"

mkdir -p "$DEST/board_workspace"
```

## 38.1 Registrar ambiente da placa

```bash
{
    echo "ACTUAL_EXPERIMENT_DATE=2026-09-10"
    echo "NOTE=ZCU104 clock was incorrect; RUN_ID 20250504_164021 is not the real experiment date."
    echo
    ssh "$BOARD" '
        echo "===== DATE REPORTED BY BOARD ====="
        date
        echo
        echo "===== UNAME ====="
        uname -a
        echo
        echo "===== OS ====="
        cat /etc/os-release 2>/dev/null || true
        echo
        echo "===== PYTHON ====="
        python3 --version
        echo
        echo "===== CPUFREQ ====="
        for p in /sys/devices/system/cpu/cpufreq/policy*; do
            echo "--- $p"
            for f in scaling_governor scaling_cur_freq scaling_min_freq scaling_max_freq cpuinfo_min_freq cpuinfo_max_freq scaling_setspeed; do
                [ -f "$p/$f" ] && printf "%s=" "$f" && cat "$p/$f"
            done
        done
    '
} > "$DEST/BOARD_ENVIRONMENT.txt"
```

## 38.2 Gerar manifesto SHA256 na placa antes da cópia

```bash
ssh "$BOARD" "
    cd '$REMOTE' &&
    find . -type f -print0 |
    sort -z |
    xargs -0 sha256sum
" > "$DEST/SHA256SUMS_BOARD.txt"
```

## 38.3 Registrar lista dos arquivos da placa

```bash
ssh "$BOARD" "
    cd '$REMOTE' &&
    find . -type f -printf '%p\t%s bytes\n' |
    sort
" > "$DEST/BOARD_FILE_LIST.txt"
```

## 38.4 Copiar workspace inteiro com rsync

```bash
rsync -avh \
    --info=progress2 \
    --partial \
    "$BOARD:$REMOTE/" \
    "$DEST/board_workspace/"
```

Não usar `--delete` neste backup.

## 38.5 Verificar hashes após a cópia

```bash
(
    cd "$DEST/board_workspace" &&
    sha256sum -c "../SHA256SUMS_BOARD.txt"
) | tee "$DEST/SHA256_VERIFY.txt"
```

Verificar no final se não existe `FAILED`:

```bash
grep -n "FAILED" "$DEST/SHA256_VERIFY.txt" || echo "OK: nenhum hash falhou"
```

---

# 39. Alternativa se rsync não estiver disponível

Executar no PC:

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
DEST="$BASE/coletas_zcu104_lenet/2026-09-10_lenet_mnist_hls4ml_zcu104"

mkdir -p "$DEST"

scp -r \
    xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/lenet_mnist \
    "$DEST/board_workspace"
```

O `rsync` é preferível porque é retomável e mostra melhor o progresso.

---

# 40. Snapshot adicional do projeto Vivado no PC

O projeto Vivado original está fora do workspace do TCC. Para não perder nada:

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
DEST="$BASE/coletas_zcu104_lenet/2026-09-10_lenet_mnist_hls4ml_zcu104"

mkdir -p "$DEST/pc_vivado"

cd /home/miguel

tar -czf \
    "$DEST/pc_vivado/LeNet-tcc3-08_09_Vivado_full.tar.gz" \
    LeNet-tcc3-08_09*
```

Depois:

```bash
sha256sum \
    "$DEST/pc_vivado/LeNet-tcc3-08_09_Vivado_full.tar.gz" \
    > "$DEST/pc_vivado/LeNet-tcc3-08_09_Vivado_full.tar.gz.sha256"
```

Esse arquivo pode ser grande, mas é o backup bruto mais seguro.

---

# 41. Snapshot do modelo e deployment local

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
DEST="$BASE/coletas_zcu104_lenet/2026-09-10_lenet_mnist_hls4ml_zcu104"

mkdir -p "$DEST/pc_reference"

cp -av \
    "/home/miguel/Downloads/Plano testes TCC/LeNet/lenet_mnist_final.h5" \
    "$DEST/pc_reference/"

rsync -avh \
    "$BASE/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/" \
    "$DEST/pc_reference/deploy_snapshot/"
```

Gerar hashes:

```bash
(
    cd "$DEST/pc_reference" &&
    find . -type f -print0 |
    sort -z |
    xargs -0 sha256sum
) > "$DEST/PC_REFERENCE_SHA256.txt"
```

---

# 42. Copiar este README/TXT para o workspace

Depois de baixar os dois arquivos entregues pelo ChatGPT, colocá-los em:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/
```

e também, de preferência, dentro do pacote de coleta:

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
DEST="$BASE/coletas_zcu104_lenet/2026-09-10_lenet_mnist_hls4ml_zcu104"

cp -av \
    "$BASE/README_FINAL_LENET_ZCU104_HLS4ML_COLETA_COMPLETA.md" \
    "$DEST/"

cp -av \
    "$BASE/README_FINAL_LENET_ZCU104_HLS4ML_COLETA_COMPLETA.txt" \
    "$DEST/"
```

---

# 43. Criar pacote final comprimido

Depois de conferir os hashes:

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
PARENT="$BASE/coletas_zcu104_lenet"
NAME="2026-09-10_lenet_mnist_hls4ml_zcu104"

cd "$PARENT"

tar -czf \
    "${NAME}.tar.gz" \
    "$NAME"

sha256sum \
    "${NAME}.tar.gz" \
    > "${NAME}.tar.gz.sha256"
```

Conferência:

```bash
sha256sum -c "${NAME}.tar.gz.sha256"
```

---

# 44. Inspeção rápida depois da cópia

```bash
BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
DEST="$BASE/coletas_zcu104_lenet/2026-09-10_lenet_mnist_hls4ml_zcu104"

du -sh "$DEST"

find "$DEST" -maxdepth 4 -type f | sort | less
```

Localizar principais resultados:

```bash
find "$DEST" -type f \
    \( \
       -name "METADATA.json" \
       -o -name "validation_10000_summary.json" \
       -o -name "LATENCY_DMA_FINAL.json" \
       -o -name "LATENCY_E2E_FINAL.json" \
       -o -name "THROUGHPUT_DMA_FINAL.json" \
       -o -name "THROUGHPUT_E2E_FINAL.json" \
       -o -name "FINAL_POWER_ENERGY.json" \
       -o -name "SATURATED_POWER_ENERGY_FINAL.json" \
       -o -name "THREE_SCENARIO_COMPARISON.json" \
       -o -name "*.bit" \
       -o -name "*.hwh" \
    \) -print | sort
```

---

# 45. Comando único recomendado

Foi criado junto com este README o script:

```text
copiar_lenet_zcu104_para_pc.sh
```

Depois de salvá-lo no PC:

```bash
chmod +x copiar_lenet_zcu104_para_pc.sh
./copiar_lenet_zcu104_para_pc.sh
```

Ele executa a cópia do workspace da placa, gera/verifica hashes, coleta informações do ambiente e cria o pacote `.tar.gz`.

---

# 46. Cuidados antes de desligar a placa ou apagar arquivos

Não apagar a pasta da ZCU104 até verificar no PC:

```text
1. rsync/scp terminou;
2. SHA256_VERIFY.txt não possui FAILED;
3. lenet_zcu104.bit está presente;
4. lenet_zcu104.hwh está presente;
5. golden/ está presente;
6. validation_10000_full.npz está presente;
7. latency_all_100000_ns.npy está presente;
8. latency_e2e_all_100000_ns.npy está presente;
9. CSVs de potência estão presentes;
10. FINAL_POWER_ENERGY.json está presente;
11. SATURATED_POWER_ENERGY_FINAL.json está presente;
12. THREE_SCENARIO_COMPARISON.json está presente;
13. METADATA.json está presente;
14. o tar.gz final passou no sha256sum -c.
```

Somente depois considerar seguro limpar arquivos da placa.

---

# 47. Limitações e observações metodológicas para o TCC

1. A latência DMA+FPGA inclui software PYNQ e DMA; não é latência pura do kernel.
2. A latência do núcleo HLS de 94,12–94,60 µs é estimativa de HLS.
3. O E2E parte de imagem uint8 já carregada em RAM; I/O de armazenamento não entra.
4. A potência usada é potência total do rail de entrada 12V da placa.
5. A frequência final de amostragem de potência é 0,5 Hz.
6. Janelas idle de 10 s geraram aproximadamente 5 amostras úteis.
7. Janelas ativas de ~30 s geraram aproximadamente 14–15 amostras.
8. Janelas sustentadas de ~60 s geraram aproximadamente 30 amostras.
9. Houve 5 repetições pareadas para potência/energia.
10. A média aritmética foi adotada como estimador primário; trapezoidal é suplementar.
11. Nenhum outlier foi removido.
12. O throughput usado no cálculo energético foi medido na mesma janela.
13. A carga “saturada” é saturação prática serial do deployment atual, não prova de saturação intrínseca do HLS.
14. A alta ocupação física de CLB (97,76%) deve ser considerada ao interpretar escalabilidade para redes maiores.
15. O relógio da placa estava incorreto; data real da campanha = 10/09/2026.

---

# 48. Conclusões principais desta campanha

A implementação física LeNet/MNIST via hls4ml na ZCU104 foi validada com sucesso.

Pontos centrais:

- acurácia FPGA de **98,99%**;
- concordância física FPGA × referência hls4ml C++ de **100%**;
- 100.000 palavras de logits Q22.12 verificadas sem mismatch;
- 3000 execuções adicionais de determinismo sem falhas;
- latência DMA+FPGA de aproximadamente **0,370 ms**;
- latência E2E de aproximadamente **1,097 ms**;
- throughput DMA+FPGA principal de aproximadamente **2792 FPS**;
- throughput E2E principal de aproximadamente **907 FPS**;
- carga serial sustentada de ~60 s manteve aproximadamente **2785 FPS**;
- potência total de placa durante inferência em torno de **10,6 W**;
- energia dinâmica inference-only de aproximadamente **0,099 mJ/inferência**;
- energia dinâmica E2E de aproximadamente **0,259 mJ/inferência**;
- implementation fechou timing a 100 MHz;
- 0 routing errors;
- 0 DRC errors;
- hardware usou 67,05% das LUTs, 43,17% dos DSPs e 40,06% da BRAM equivalente;
- ocupação física de CLBs chegou a 97,76%.

O experimento mostrou que a carga inference-only normal já se encontra praticamente no regime máximo sustentado do caminho DMA+FPGA atual. O custo E2E é dominado não por aumento expressivo de potência instantânea, mas pelo tempo adicional associado às etapas executadas no ARM e à movimentação/preparação de dados.

Este documento deve ser preservado junto com os dados brutos, os hashes, o overlay e o snapshot do projeto para servir de base auditável à redação final do TCC.
