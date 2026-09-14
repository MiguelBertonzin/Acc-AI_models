# LeNet / MNIST — hls4ml → Vivado → ZCU104 → PYNQ

## Handoff completo da implementação FPGA e guia para validação física

Este documento registra o estado final da implementação da rede LeNet/MNIST utilizando **hls4ml**, **Vitis HLS 2024.2**, **Vivado 2024.2** e a placa **AMD/Xilinx ZCU104**.

O objetivo principal é tornar esta implementação completamente reproduzível e permitir continuar o trabalho diretamente na etapa de validação física com **PYNQ/Jupyter Notebook**, sem precisar reconstruir o raciocínio usado durante a integração no Vivado.

Este README deve ser tratado como a principal referência de handoff entre:

```text
modelo Keras
    ↓
hls4ml
    ↓
Vitis HLS
    ↓
IP RTL
    ↓
Vivado Block Design
    ↓
AXI DMA
    ↓
DDR / ARM
    ↓
bitstream + HWH
    ↓
PYNQ
    ↓
inferência física
    ↓
benchmark
```

---

# 1. Estado atual do projeto

## 1.1 Etapas concluídas

* [x] modelo LeNet treinado em Keras;
* [x] modelo sem Softmax, com saída em logits;
* [x] conversão para hls4ml 1.3.0;
* [x] definição da precisão Q22.12;
* [x] definição dos Reuse Factors por camada;
* [x] validação C++ do hls4ml nas 10.000 imagens MNIST;
* [x] síntese C/C++ no Vitis HLS 2024.2;
* [x] exportação do IP;
* [x] promoção do IP para repositório local;
* [x] síntese isolada do IP no Vivado;
* [x] criação do wrapper específico para AXI DMA;
* [x] criação do projeto Vivado;
* [x] criação do Block Design;
* [x] configuração do Zynq UltraScale+ PS;
* [x] integração AXI DMA;
* [x] integração SmartConnect;
* [x] conexão com DDR através de S_AXI_HPC0_FPD;
* [x] configuração de clock em 100 MHz;
* [x] configuração correta de reset;
* [x] mapa de endereços;
* [x] Validate Design;
* [x] geração do HDL wrapper do Block Design;
* [x] síntese do sistema integrado;
* [x] placement;
* [x] routing;
* [x] fechamento de setup timing;
* [x] fechamento de hold timing;
* [x] DRC sem Errors;
* [x] DRC sem Critical Warnings;
* [x] geração do bitstream;
* [x] geração/localização do HWH.

## 1.2 Próximas etapas

Ainda faltam:

* [ ] copiar `.bit` e `.hwh` para uma pasta permanente dentro deste workspace;
* [ ] copiar os relatórios finais do Vivado;
* [ ] opcionalmente exportar XSA;
* [ ] carregar o overlay na ZCU104 com PYNQ;
* [ ] realizar smoke test com uma imagem;
* [ ] comparar os 10 logits FPGA × hls4ml C++;
* [ ] validar 100 imagens;
* [ ] validar as 10.000 imagens MNIST;
* [ ] medir latência;
* [ ] medir throughput;
* [ ] medir consumo idle;
* [ ] medir consumo durante inferência;
* [ ] calcular energia dinâmica por inferência;
* [ ] comparar com Vitis AI/DPU e demais plataformas do TCC.

---

# 2. Workspace principal

Workspace do experimento:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet
```

Workspace específico do hls4ml:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml
```

Estrutura observada:

```text
hls4ml/
├── builds/
├── configs/
├── data/
├── hardware/
├── ip_repo/
├── reports/
├── scripts/
├── README.md
├── README-blockdesign.md
├── README_CHATGPT_WEB_LENET_ZCU104_VIVADO.md
└── README_FINAL_LENET_ZCU104_VIVADO_PYNQ.md
```

Este novo README representa o estado **posterior à geração bem-sucedida do bitstream**.

---

# 3. Arquivos principais

## 3.1 Modelo Keras

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/lenet_mnist_final.h5
```

O modelo possui saída em **logits**.

Não existe Softmax no final da rede.

A classificação deve ser realizada no software usando:

```python
pred = np.argmax(logits)
```

Não adicionar Softmax no hardware apenas para classificação.

---

# 4. Arquitetura da LeNet

Arquitetura funcional:

```text
Entrada
28 × 28 × 1
    |
    v
Conv2D
6 filtros 5 × 5
valid
    |
24 × 24 × 6
    |
    v
ReLU
    |
    v
MaxPooling2D 2 × 2
    |
12 × 12 × 6
    |
    v
Conv2D
16 filtros 5 × 5
valid
    |
8 × 8 × 16
    |
    v
ReLU
    |
    v
MaxPooling2D 2 × 2
    |
4 × 4 × 16
    |
    v
Flatten
    |
256
    |
    v
Dense 120
    |
ReLU
    |
    v
Dense 84
    |
ReLU
    |
    v
Dense 10
linear
    |
    v
10 logits
```

Número total de parâmetros:

```text
44.426
```

Distribuição:

| Camada      | Parâmetros |
| ----------- | ---------: |
| Conv1       |        156 |
| Conv2       |      2.416 |
| Dense1      |     30.840 |
| Dense2      |     10.164 |
| Dense saída |        850 |
| **Total**   | **44.426** |

---

# 5. Configuração hls4ml congelada

Esta configuração representa o experimento que gerou o bitstream atual.

Não alterar simultaneamente precisão, modelo, Reuse Factor ou estratégia se o objetivo for comparar resultados de forma controlada.

| Parâmetro          | Valor                                |
| ------------------ | ------------------------------------ |
| hls4ml             | 1.3.0                                |
| Backend            | Vitis                                |
| Vitis HLS          | 2024.2                               |
| Vivado             | 2024.2                               |
| FPGA               | XCZU7EV                              |
| Board              | ZCU104                               |
| Clock              | 10 ns                                |
| Frequência         | 100 MHz                              |
| IOType             | `io_stream`                          |
| Strategy           | `Resource`                           |
| ConvImplementation | `LineBuffer`                         |
| Batch              | 1                                    |
| Softmax            | não                                  |
| Saída              | logits                               |
| Precisão           | `ap_fixed<22,12,AP_RND_CONV,AP_SAT>` |

Reuse Factors:

```text
Conv1   = 5
Conv2   = 50
Dense1  = 64
Dense2  = 60
Output  = 42
```

---

# 6. Formato numérico Q22.12

Tipo:

```cpp
ap_fixed<22,12,AP_RND_CONV,AP_SAT>
```

Temos:

```text
22 bits totais
12 bits para parte inteira + sinal
10 bits fracionários
```

Escala:

```text
2^10 = 1024
```

Conversão aproximada:

```text
float → fixed:
round(x × 1024)
```

Conversão inversa:

```text
fixed → float:
signed_integer / 1024
```

O bit de sinal do valor Q22.12 é:

```text
bit 21
```

Máscara de 22 bits:

```text
0x003FFFFF
```

Bit de sinal:

```text
0x00200000
```

Módulo para complemento de dois:

```text
0x00400000
```

---

# 7. Validação hls4ml antes da implementação física

Validação C++ realizada sobre as **10.000 imagens de teste do MNIST**:

| Métrica              |  Resultado |
| -------------------- | ---------: |
| Acurácia Keras       |     98,98% |
| Acurácia hls4ml C++  |     98,99% |
| Concordância top-1   |     99,99% |
| MAE dos logits       | 0,01564898 |
| Erro absoluto máximo | 0,09856367 |

Esses números formam a referência funcional para a FPGA.

A FPGA **não deve ser considerada correta apenas porque gera saída**.

O teste físico deverá comparar:

```text
Keras
vs
hls4ml C++
vs
FPGA
```

usando as mesmas imagens e o mesmo preprocessing.

---

# 8. IP hls4ml promovido

Diretório:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/ip_repo/lenet_mnist_cap64_hls_v1_0
```

VLNV:

```text
xilinx.com:hls:lenet_mnist_cap64_hls:1.0
```

Esse diretório deve ser tratado como artefato congelado deste experimento.

Não editar manualmente RTL gerado dentro do IP para tentar corrigir timing ou warnings.

Qualquer modificação do núcleo deve ser realizada na configuração hls4ml e seguida por uma nova geração e nova validação.

---

# 9. Interface externa do IP LeNet

## 9.1 Clock

```text
ap_clk
```

Frequência utilizada:

```text
100 MHz
```

---

## 9.2 Reset

```text
ap_rst_n
```

Reset:

```text
ativo em nível baixo
```

---

## 9.3 Controle

O IP utiliza:

```text
ap_ctrl_hs
```

Sinais:

```text
ap_start
ap_done
ap_idle
ap_ready
```

O IP **não possui AXI4-Lite para controle**.

Na implementação atual:

```text
ap_start = 1 permanentemente
```

Isso é realizado pelo wrapper.

Dessa forma, o ARM controla somente o AXI DMA.

---

# 10. Interface de entrada do IP

Interface de entrada original:

```text
input_layer_TDATA[31:0]
input_layer_TVALID
input_layer_TREADY
```

Cada pixel é transmitido como um beat AXI4-Stream equivalente a:

```text
32 bits = 4 bytes
```

Por inferência:

```text
28 × 28 = 784 pixels
```

Portanto:

```text
784 beats
784 × 4 bytes
3136 bytes
```

Tamanho exato do envio DMA:

```text
3136 bytes
```

O valor Q22.12 ocupa apenas os 22 bits inferiores da lane de 32 bits.

Os 10 bits superiores devem permanecer em zero.

---

# 11. Interface de saída original do IP

Saída original:

```text
layer13_out_TDATA[319:0]
layer13_out_TVALID
layer13_out_TREADY
```

São:

```text
320 bits
```

correspondentes a:

```text
10 logits × 32 bits
```

Cada lane de 32 bits contém um valor Q22.12 nos 22 bits inferiores.

A interface original **não possui TLAST**.

Isso impede a conexão direta ao canal S2MM convencional do AXI DMA, pois o DMA precisa identificar o final do pacote.

Por esse motivo existe o wrapper.

---

# 12. Wrapper AXI DMA da LeNet

Arquivo:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/hardware/src/lenet_dma_wrapper.vhd
```

Instância no Block Design:

```text
lenet_dma_wrapper_0
```

O wrapper possui três funções principais:

1. adaptar a interface de entrada para o IP;
2. adaptar a saída de 320 bits para 512 bits;
3. gerar `TLAST` para o AXI DMA.

---

# 13. Wrapper — entrada

A interface AXIS de entrada do wrapper possui:

```text
TDATA = 32 bits
TVALID
TREADY
TLAST
```

O `TLAST` fornecido pelo DMA no lado de entrada não é necessário pelo IP LeNet.

A quantidade de dados de entrada é determinada pelo BTT do AXI DMA:

```text
3136 bytes
```

Conexões lógicas:

```vhdl
accel_input_data  <= s_axis_tdata;
accel_input_valid <= s_axis_tvalid;
s_axis_tready     <= accel_input_ready;
```

---

# 14. Wrapper — controle

O acelerador é mantido pronto para iniciar:

```vhdl
accel_ap_start <= '1';
```

Portanto não existe necessidade de acesso AXI-Lite ao IP LeNet.

---

# 15. Wrapper — saída

A saída original possui:

```text
320 bits
```

O canal S2MM foi configurado para:

```text
512 bits
```

O wrapper realiza:

```vhdl
m_axis_tdata(319 downto 0) <= accel_output_data;

m_axis_tdata(511 downto 320) <= (others => '0');
```

Logo:

```text
bits 319:0   = dez logits
bits 511:320 = padding zero
```

A saída completa possui:

```text
512 bits
64 bytes
16 palavras de 32 bits
```

Estrutura do buffer recebido:

```text
word 0  = logit classe 0
word 1  = logit classe 1
...
word 9  = logit classe 9
word 10 = 0
word 11 = 0
word 12 = 0
word 13 = 0
word 14 = 0
word 15 = 0
```

---

# 16. Geração de TLAST

O IP original não possui TLAST.

Como existe apenas um beat de 512 bits por inferência na saída, o wrapper utiliza:

```vhdl
m_axis_tlast <= accel_output_valid;
```

Também:

```vhdl
m_axis_tvalid <= accel_output_valid;
accel_output_ready <= m_axis_tready;
```

Enquanto ocorrer backpressure:

```text
TVALID = 1
TLAST  = 1
```

até o handshake ocorrer.

---

# 17. Arquitetura final do Block Design

O Block Design final possui exatamente os seguintes blocos principais:

```text
zynq_ultra_ps_e_0
rst_ps8_0_100M
axi_smc_ctrl
axi_smc_mem
axi_dma_0
lenet_dma_wrapper_0
lenet_mnist_cap64_hls_0
```

Não existem no projeto final:

```text
Utility Vector Logic para reset
xlconstant de reset
Clock Wizard
HPM1
AXI Interrupt Controller
Concat para IRQ
```

A primeira implementação utiliza polling pelo software.

---

# 18. Diagrama geral do sistema

```text
                         Zynq UltraScale+ MPSoC
                     ┌────────────────────────────┐
                     │                            │
                     │ ARM Cortex-A53             │
                     │ DDR                        │
                     │                            │
                     └──────────┬─────────▲───────┘
                                │         │
                    M_AXI_HPM0  │         │ S_AXI_HPC0
                                │         │
                                v         │
                       ┌─────────────┐    │
                       │ SmartConnect│    │
                       │    CTRL     │    │
                       └──────┬──────┘    │
                              │           │
                              v           │
                       ┌─────────────┐    │
                       │   AXI DMA   │    │
                       │             │    │
                       │ S_AXI_LITE  │    │
                       │ MM2S        │────┼─────────────┐
                       │ S2MM        │◄───┼──────────┐  │
                       └─────────────┘    │          │  │
                                         │          │  │
                                   ┌─────┴─────┐    │  │
                                   │SmartConnect│   │  │
                                   │    MEM     │   │  │
                                   └────────────┘   │  │
                                                    │  │
                         AXIS 512                   │  │ AXIS 32
                                                    │  │
                                    ┌───────────────┘  │
                                    │                  │
                              ┌─────▼───────────────▼──┐
                              │ lenet_dma_wrapper_0    │
                              └──────────┬─────────────┘
                                         │
                                  sinais do IP
                                         │
                              ┌──────────▼───────────┐
                              │ LeNet hls4ml IP      │
                              │ Q22.12               │
                              │ 100 MHz              │
                              └──────────────────────┘
```

Forma simplificada do datapath:

```text
DDR
 ↓
DMA MM2S
 ↓
AXIS 32 bits
 ↓
wrapper
 ↓
LeNet hls4ml
 ↓
320 bits
 ↓
wrapper
 ↓
512 bits + TLAST
 ↓
DMA S2MM
 ↓
DDR
```

---

# 19. Configuração do Zynq UltraScale+ PS

Instância:

```text
zynq_ultra_ps_e_0
```

Board:

```text
xilinx.com:zcu104:part0:1.1
```

Part:

```text
xczu7ev-ffvc1156-2-e
```

Interfaces utilizadas:

```text
M_AXI_HPM0_FPD = habilitada
M_AXI_HPM1_FPD = desabilitada
S_AXI_HPC0_FPD = habilitada
```

Motivação:

```text
HPM0:
ARM → periféricos no PL
controle AXI-Lite do DMA

HPC0:
PL → PS/DDR
acesso de alta largura de banda à DDR
```

Não existe motivo para habilitar HPM1 nesta implementação porque há apenas um caminho de controle AXI-Lite relevante.

---

# 20. SmartConnect de controle

Instância:

```text
axi_smc_ctrl
```

Configuração:

```text
NUM_SI = 1
NUM_MI = 1
```

Conexões:

```text
zynq_ultra_ps_e_0/M_AXI_HPM0_FPD
    ↓
axi_smc_ctrl/S00_AXI

axi_smc_ctrl/M00_AXI
    ↓
axi_dma_0/S_AXI_LITE
```

Esse caminho é utilizado apenas para programação dos registradores do AXI DMA.

O Vivado configurou esse SmartConnect em:

```text
Low-Area Mode
```

Isso é aceitável porque o tráfego é essencialmente AXI4-Lite de controle.

---

# 21. SmartConnect de memória

Instância:

```text
axi_smc_mem
```

Configuração:

```text
NUM_SI = 2
NUM_MI = 1
```

Conexões:

```text
axi_dma_0/M_AXI_MM2S
    ↓
axi_smc_mem/S00_AXI
```

e:

```text
axi_dma_0/M_AXI_S2MM
    ↓
axi_smc_mem/S01_AXI
```

saída:

```text
axi_smc_mem/M00_AXI
    ↓
zynq_ultra_ps_e_0/S_AXI_HPC0_FPD
```

Esse SmartConnect foi automaticamente colocado pelo Vivado em:

```text
High-performance Mode
```

Esse é o datapath responsável pelos acessos do DMA à DDR.

---

# 22. AXI DMA

Instância:

```text
axi_dma_0
```

Configuração congelada:

```text
Scatter Gather       = OFF
Micro DMA            = OFF
Address width        = 32 bits
Buffer length width  = 26 bits
```

## MM2S

```text
Memory map width = 128 bits
AXIS width       = 32 bits
Max burst        = 64
DRE              = ON
```

Fluxo:

```text
DDR
→ M_AXI_MM2S
→ DMA
→ M_AXIS_MM2S
→ wrapper
→ LeNet
```

## S2MM

```text
Memory map width = 512 bits
AXIS width       = 512 bits
Max burst        = 64
DRE              = ON
```

Fluxo:

```text
LeNet
→ wrapper
→ S_AXIS_S2MM
→ DMA
→ M_AXI_S2MM
→ DDR
```

---

# 23. Conexões AXI4-Stream DMA ↔ wrapper

Entrada:

```text
axi_dma_0/M_AXIS_MM2S
    ↓
lenet_dma_wrapper_0/s_axis
```

Largura:

```text
32 bits
```

Saída:

```text
lenet_dma_wrapper_0/m_axis
    ↓
axi_dma_0/S_AXIS_S2MM
```

Largura:

```text
512 bits
```

---

# 24. Conexões wrapper ↔ LeNet

O IP hls4ml não expôs as interfaces de dados da forma mais conveniente para conexão automática ao wrapper.

As conexões foram realizadas entre os sinais correspondentes.

Entrada:

```text
wrapper/accel_input_data
    →
LeNet/input_layer_TDATA
```

```text
wrapper/accel_input_valid
    →
LeNet/input_layer_TVALID
```

```text
LeNet/input_layer_TREADY
    →
wrapper/accel_input_ready
```

Saída:

```text
LeNet/layer13_out_TDATA
    →
wrapper/accel_output_data
```

```text
LeNet/layer13_out_TVALID
    →
wrapper/accel_output_valid
```

```text
wrapper/accel_output_ready
    →
LeNet/layer13_out_TREADY
```

Controle:

```text
wrapper/accel_ap_start
    →
LeNet/ap_start
```

Observabilidade:

```text
LeNet/ap_done
    →
wrapper/accel_ap_done

LeNet/ap_idle
    →
wrapper/accel_ap_idle

LeNet/ap_ready
    →
wrapper/accel_ap_ready
```

Durante a construção o Vivado apresentou warnings ao conectar individualmente pins pertencentes a interfaces agrupadas do IP HLS.

Esses warnings não se transformaram em erro de Validate, síntese ou implementação.

O design final foi sintetizado e roteado corretamente.

---

# 25. Clock

Clock principal:

```text
zynq_ultra_ps_e_0/pl_clk0
```

Frequência:

```text
100 MHz
```

Período:

```text
10 ns
```

Todos os blocos principais pertencem ao mesmo domínio.

Conexões de clock:

```text
pl_clk0
 ├── rst_ps8_0_100M/slowest_sync_clk
 ├── zynq_ultra_ps_e_0/maxihpm0_fpd_aclk
 ├── zynq_ultra_ps_e_0/saxihpc0_fpd_aclk
 ├── axi_smc_ctrl/aclk
 ├── axi_smc_mem/aclk
 ├── axi_dma_0/s_axi_lite_aclk
 ├── axi_dma_0/m_axi_mm2s_aclk
 ├── axi_dma_0/m_axi_s2mm_aclk
 ├── lenet_dma_wrapper_0/ap_clk
 └── lenet_mnist_cap64_hls_0/ap_clk
```

Não é utilizado Clock Wizard.

---

# 26. Reset

Bloco:

```text
rst_ps8_0_100M
```

O reset do PS:

```text
zynq_ultra_ps_e_0/pl_resetn0
```

é ativo em nível baixo.

A conexão final correta é direta:

```text
zynq_ultra_ps_e_0/pl_resetn0
    ↓
rst_ps8_0_100M/ext_reset_in
```

Depois do `validate_bd_design`, o Vivado propagou corretamente:

```text
PS pl_resetn0 POLARITY = ACTIVE_LOW
ext_reset_in POLARITY  = ACTIVE_LOW
C_EXT_RESET_HIGH       = 0
```

Portanto:

**não utilizar inversor externo.**

Também não são necessários blocos `xlconstant`.

Os inputs auxiliares do `proc_sys_reset` possuem defaults apropriados:

```text
dcm_locked:
DEFAULT_DRIVER = 1

aux_reset_in:
ACTIVE_LOW
DEFAULT_DRIVER = 1

mb_debug_sys_rst:
ACTIVE_HIGH
DEFAULT_DRIVER = 0
```

---

# 27. Distribuição do reset

Saída utilizada:

```text
rst_ps8_0_100M/peripheral_aresetn
```

Conexões:

```text
peripheral_aresetn
 ├── axi_smc_ctrl/aresetn
 ├── axi_smc_mem/aresetn
 ├── axi_dma_0/axi_resetn
 ├── lenet_dma_wrapper_0/ap_rst_n
 └── lenet_mnist_cap64_hls_0/ap_rst_n
```

Arquitetura:

```text
PS pl_resetn0
      |
      v
proc_sys_reset
      |
      v
peripheral_aresetn
      |
      +---- SmartConnect CTRL
      +---- SmartConnect MEM
      +---- AXI DMA
      +---- wrapper
      +---- LeNet
```

---

# 28. Mapa de endereços

## 28.1 Controle do DMA

Address space:

```text
zynq_ultra_ps_e_0/Data
```

Segmento:

```text
axi_dma_0/S_AXI_LITE/Reg
```

Endereço:

```text
0xA0000000
```

Range:

```text
0x00010000
64 KiB
```

---

# 29. DMA MM2S → DDR

Address space:

```text
axi_dma_0/Data_MM2S
```

Destino:

```text
zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW
```

Offset:

```text
0x00000000
```

Range:

```text
0x80000000
2 GiB
```

---

# 30. DMA S2MM → DDR

Address space:

```text
axi_dma_0/Data_S2MM
```

Destino:

```text
zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW
```

Offset:

```text
0x00000000
```

Range:

```text
0x80000000
2 GiB
```

---

# 31. Segmentos deliberadamente excluídos

Para MM2S e S2MM foram excluídos:

```text
HPC0_QSPI
HPC0_LPS_OCM
```

Somente:

```text
HPC0_DDR_LOW
```

é necessário ao DMA.

Os warnings do Vivado informando que QSPI e OCM estão excluídos são esperados.

---

# 32. Warnings de interface HPC0

Durante Validate foram observados:

```text
AWUSER_WIDTH mismatch
ARUSER_WIDTH mismatch
```

entre:

```text
zynq_ultra_ps_e_0/S_AXI_HPC0_FPD
```

e:

```text
axi_smc_mem/M00_AXI
```

Esses warnings não impediram:

```text
Validate
Synthesis
Placement
Routing
Timing closure
Bitstream
```

Portanto não alterar a arquitetura apenas para eliminar esses warnings.

---

# 33. Projeto Vivado atual

O projeto foi criado a partir de:

```text
/home/miguel
```

com nome:

```text
LeNet-tcc3-08_09
```

Arquivo principal:

```text
/home/miguel/LeNet-tcc3-08_09.xpr
```

Diretórios relacionados:

```text
/home/miguel/LeNet-tcc3-08_09.srcs
/home/miguel/LeNet-tcc3-08_09.gen
/home/miguel/LeNet-tcc3-08_09.runs
```

Block Design:

```text
/home/miguel/LeNet-tcc3-08_09.srcs/sources_1/bd/system/system.bd
```

IMPORTANTE:

esses arquivos estão fora do workspace do TCC.

Por isso, os artefatos necessários devem ser copiados para:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml
```

antes de considerar este experimento arquivado.

---

# 34. HDL wrapper do Block Design

Top do projeto:

```text
system_wrapper
```

O top foi gerado pelo Vivado a partir do:

```text
system.bd
```

O `system_wrapper` é diferente do:

```text
lenet_dma_wrapper.vhd
```

Funções:

```text
system_wrapper:
top global do projeto Vivado

lenet_dma_wrapper:
adaptador entre DMA e IP LeNet
```

---

# 35. Recursos do IP LeNet isolado

Resultados anteriores para o IP isolado:

| Recurso   |   Usado | Disponível |    Uso |
| --------- | ------: | ---------: | -----: |
| LUT       | 141.417 |    230.400 | 61,38% |
| FF        |  81.051 |    460.800 | 17,59% |
| DSP48E2   |     746 |      1.728 | 43,17% |
| BRAM Tile |     114 |        312 | 36,54% |
| URAM      |       0 |         96 |     0% |
| CARRY8    |  15.933 |     28.800 | 55,32% |

Esses números não devem ser confundidos com o uso do sistema completo.

---

# 36. Recursos do sistema completo pós-route

Resultado final após integração com:

```text
PS
DMA
SmartConnect
reset
wrapper
LeNet
```

| Recurso                 |   Usado | Disponível |    Uso |
| ----------------------- | ------: | ---------: | -----: |
| CLB LUTs                | 154.485 |    230.400 | 67,05% |
| LUT como lógica         | 151.780 |    230.400 | 65,88% |
| LUT como memória        |   2.705 |    101.760 |  2,66% |
| CLB Registers           |  96.655 |    460.800 | 20,98% |
| CARRY8                  |  16.075 |     28.800 | 55,82% |
| DSP48E2                 |     746 |      1.728 | 43,17% |
| RAMB36E2                |     122 |          — |      — |
| RAMB18E2                |       6 |          — |      — |
| BRAM Tiles equivalentes |     125 |        312 | 40,06% |
| URAM                    |       0 |         96 |     0% |

---

# 37. Ocupação física de CLBs

Apesar do uso de LUTs ser aproximadamente:

```text
67,05%
```

a ocupação física foi:

```text
CLBs usados = 28.154
CLBs disponíveis = 28.800
Uso = 97,76%
```

Esse é um resultado muito importante.

A implementação está próxima do limite de ocupação física da ZCU104.

Isso explica os avisos de congestionamento observados durante placement/routing.

Não interpretar somente a porcentagem de LUTs para decidir se uma configuração “cabe confortavelmente”.

---

# 38. Congestionamento

Durante route o Vivado indicou congestionamento estimado de nível:

```text
6
```

O router inicialmente apresentou grande quantidade de overlaps, mas conseguiu eliminá-los durante rip-up and reroute.

Resultado final:

```text
Failed nets            = 0
Unrouted nets          = 0
Partially routed nets  = 0
Node overlaps          = 0
```

No signoff resumido:

```text
routable nets = 280.922
fully routed  = 280.922
routing errors = 0
```

Portanto o design foi completamente roteado.

---

# 39. Timing final

Clock alvo:

```text
100 MHz
10 ns
```

Signoff manual pós-route:

```text
WNS_SETUP = +0.033 ns
WHS_HOLD  = +0.010 ns
```

Ou seja:

```text
setup timing = PASS
hold timing  = PASS
```

O router também reportou:

```text
WNS = +0.030 ns
TNS = 0.000 ns
WHS = +0.010 ns
THS = 0.000 ns
```

Portanto a implementação fechou timing em 100 MHz.

---

# 40. check_timing

Após route:

```text
no_clock                        = 0
constant_clock                  = 0
pulse_width_clock               = 0
unconstrained_internal_endpoints = 0
no_input_delay                  = 0
no_output_delay                 = 0
multiple_clock                  = 0
generated_clocks                = 0
loops                           = 0
partial_input_delay             = 0
partial_output_delay            = 0
latch_loops                     = 0
```

A advertência `no_clock (8)` observada antes do route desapareceu no design físico final.

---

# 41. DRC final

Resultado:

```text
DRC_ERRORS   = 0
DRC_CRITICAL = 0
```

Foram observados:

```text
2999 warnings
```

Resumo:

| Regra     | Quantidade | Significado                |
| --------- | ---------: | -------------------------- |
| DPIP-2    |       1492 | DSP input pipelining       |
| DPOP-3    |        746 | DSP PREG output pipelining |
| DPOP-4    |        746 | DSP MREG pipelining        |
| PDRC-136  |          1 | packing de LUT             |
| PDRC-138  |          2 | packing de LUT             |
| REQP-1934 |          2 | advisory BRAM18            |
| REQP-1935 |         10 | advisory BRAM36            |

A enorme maioria dos warnings é recomendação de pipeline em DSPs gerados pelo hls4ml/Vitis HLS.

Não alterar manualmente os DSPs do RTL gerado somente para eliminar esses warnings.

O design já fecha timing a 100 MHz.

---

# 42. Bitstream final

Bitstream gerado com sucesso:

```text
/home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper.bit
```

Status:

```text
write_bitstream Complete!
PROGRESS = 100%
```

O BitGen terminou com:

```text
0 Warnings
0 Critical Warnings
0 Errors
```

---

# 43. HWH final

Arquivo encontrado:

```text
/home/miguel/LeNet-tcc3-08_09.gen/sources_1/bd/system/hw_handoff/system.hwh
```

Para PYNQ, é recomendado copiar `.bit` e `.hwh` com o mesmo basename.

Por exemplo:

```text
lenet_zcu104.bit
lenet_zcu104.hwh
```

---

# 44. Pasta de deployment recomendada

Criar:

```text
hls4ml/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/
```

Conteúdo desejado:

```text
lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/
├── lenet_zcu104.bit
├── lenet_zcu104.hwh
├── lenet_dma_wrapper.vhd
├── system_final_bd.tcl
├── project_final.tcl
├── metadata.txt
└── reports/
```

---

# 45. Comandos para preservar BIT e HWH

Executar no shell Linux:

```bash
cd "/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"

DEPLOY="hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz"

mkdir -p "$DEPLOY/reports"

cp \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper.bit \
  "$DEPLOY/lenet_zcu104.bit"

cp \
  /home/miguel/LeNet-tcc3-08_09.gen/sources_1/bd/system/hw_handoff/system.hwh \
  "$DEPLOY/lenet_zcu104.hwh"

cp \
  hardware/src/lenet_dma_wrapper.vhd \
  "$DEPLOY/lenet_dma_wrapper.vhd"
```

Depois:

```bash
ls -lh "$DEPLOY"
```

---

# 46. Copiar relatórios finais

Executar:

```bash
cd "/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"

DEPLOY="hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz"

cp -v \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper_timing_summary_routed.rpt \
  "$DEPLOY/reports/" 2>/dev/null || true

cp -v \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper_utilization_routed.rpt \
  "$DEPLOY/reports/" 2>/dev/null || true

cp -v \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper_drc_routed.rpt \
  "$DEPLOY/reports/" 2>/dev/null || true

cp -v \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper_route_status.rpt \
  "$DEPLOY/reports/" 2>/dev/null || true

cp -v \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper_methodology_drc_routed.rpt \
  "$DEPLOY/reports/" 2>/dev/null || true

cp -v \
  /home/miguel/LeNet-tcc3-08_09.runs/impl_1/system_wrapper_power_routed.rpt \
  "$DEPLOY/reports/" 2>/dev/null || true
```

Verificar:

```bash
find "$DEPLOY" -maxdepth 2 -type f -printf '%p\n'
```

---

# 47. Salvar o Block Design como Tcl

No **Tcl Console do Vivado**, com o projeto aberto:

```tcl
set out_dir {/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz}

write_bd_tcl \
    -force \
    [file join $out_dir system_final_bd.tcl]
```

O Tcl do BD depende de:

```text
IP promovido
wrapper VHDL
```

Portanto esses artefatos também precisam ser preservados.

---

# 48. Salvar Tcl do projeto Vivado

Ainda no Vivado:

```tcl
set out_dir {/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz}

write_project_tcl \
    -force \
    [file join $out_dir project_final.tcl]
```

Isso é recomendado porque o `.xpr` sozinho não representa uma cópia portátil completa do projeto.

---

# 49. Opcional — preservar o projeto Vivado inteiro

Para backup bruto completo:

```bash
cd /home/miguel

tar -czf \
"/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/LeNet-tcc3-08_09_Vivado_full.tar.gz" \
LeNet-tcc3-08_09.xpr \
LeNet-tcc3-08_09.srcs \
LeNet-tcc3-08_09.gen \
LeNet-tcc3-08_09.runs
```

Esse arquivo pode ser grande.

Ele é opcional, mas é o backup mais seguro caso haja espaço disponível.

---

# 50. metadata.txt recomendado

Criar:

```bash
cd "/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"

cat > \
hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/metadata.txt <<'EOF'
MODEL=lenet_mnist_final.h5
BOARD=ZCU104
PART=xczu7ev-ffvc1156-2-e
BOARD_PART=xilinx.com:zcu104:part0:1.1

HLS4ML=1.3.0
VITIS_HLS=2024.2
VIVADO=2024.2

CLOCK_NS=10
CLOCK_MHZ=100

IOTYPE=io_stream
STRATEGY=Resource
CONV_IMPLEMENTATION=LineBuffer

PRECISION=ap_fixed<22,12,AP_RND_CONV,AP_SAT>

RF_CONV1=5
RF_CONV2=50
RF_DENSE1=64
RF_DENSE2=60
RF_OUTPUT=42

BATCH=1
SOFTMAX=NO

INPUT_PIXELS=784
INPUT_BYTES=3136

OUTPUT_LOGITS=10
OUTPUT_DMA_WORDS=16
OUTPUT_BYTES=64

DMA_CONTROL_BASE=0xA0000000
DMA_CONTROL_RANGE=0x00010000

FPGA_CPP_ACCURACY_REFERENCE=98.99%
KERAS_ACCURACY_REFERENCE=98.98%

POST_ROUTE_LUT=154485
POST_ROUTE_FF=96655
POST_ROUTE_DSP=746
POST_ROUTE_BRAM_TILE_EQUIV=125
POST_ROUTE_CLB_USED=28154
POST_ROUTE_CLB_AVAILABLE=28800

WNS_SETUP_NS=0.033
WHS_HOLD_NS=0.010

DRC_ERRORS=0
DRC_CRITICAL_WARNINGS=0
ROUTING_ERRORS=0

BITSTREAM=system_wrapper.bit
HWH_SOURCE=system.hwh
EOF
```

---

# 51. Arquivos mínimos necessários para executar no PYNQ

Na ZCU104 são necessários, no mínimo:

```text
lenet_zcu104.bit
lenet_zcu104.hwh
```

É recomendado também levar:

```text
notebook.ipynb
uma ou mais imagens de teste
labels de referência
logits de referência
```

O IP HLS ou projeto Vivado completo **não é necessário na placa para executar o overlay**.

---

# 52. Estrutura recomendada na ZCU104

Exemplo:

```text
/home/xilinx/jupyter_notebooks/lenet/
├── lenet_zcu104.bit
├── lenet_zcu104.hwh
├── test_lenet.ipynb
└── data/
```

Dependendo da imagem PYNQ, o usuário e caminho home podem variar.

---

# 53. PYNQ — carregar overlay

No notebook:

```python
from pynq import Overlay

BITSTREAM = "/home/xilinx/jupyter_notebooks/lenet/lenet_zcu104.bit"

overlay = Overlay(BITSTREAM)
```

Depois:

```python
print(overlay.ip_dict.keys())
```

Esperado encontrar algo semelhante a:

```text
axi_dma_0
```

Obter DMA:

```python
dma = overlay.axi_dma_0
```

Não deve existir driver AXI-Lite para a LeNet porque o IP não possui AXI-Lite de controle.

O `ap_start` já está amarrado em 1 pelo wrapper.

---

# 54. Buffers PYNQ

Importar:

```python
import numpy as np
from pynq import allocate
```

Entrada:

```python
input_buffer = allocate(
    shape=(784,),
    dtype=np.uint32
)
```

Tamanho:

```text
784 × 4 = 3136 bytes
```

Saída:

```python
output_buffer = allocate(
    shape=(16,),
    dtype=np.uint32
)
```

Tamanho:

```text
16 × 4 = 64 bytes
```

---

# 55. Conversão float → Q22.12

```python
import numpy as np

Q_BITS = 22
Q_FRAC = 10

Q_SCALE = 1 << Q_FRAC
Q_MASK = (1 << Q_BITS) - 1

Q_MIN = -(1 << (Q_BITS - 1))
Q_MAX = (1 << (Q_BITS - 1)) - 1


def float_to_q22_12(values):
    x = np.asarray(values, dtype=np.float64)

    q = np.rint(x * Q_SCALE).astype(np.int64)

    q = np.clip(
        q,
        Q_MIN,
        Q_MAX
    )

    return (q & Q_MASK).astype(np.uint32)
```

Para MNIST normalizado:

```text
0.0 ≤ pixel ≤ 1.0
```

portanto não haverá valores negativos na entrada normal.

---

# 56. Conversão Q22.12 → float

```python
def q22_12_to_float(words):
    raw = np.asarray(
        words,
        dtype=np.uint32
    ) & 0x003FFFFF

    signed = raw.astype(np.int64)

    negative = (
        raw & 0x00200000
    ) != 0

    signed[negative] -= 0x00400000

    return signed.astype(np.float32) / 1024.0
```

---

# 57. Preprocessing MNIST

O preprocessing congelado é:

```python
pixels = image.astype(np.float32) / 255.0
```

Depois:

```python
pixels = pixels.reshape(-1)
```

Ordem:

```text
row-major
```

Quantidade:

```text
784 pixels
```

Não utilizar preprocessing diferente no teste FPGA.

---

# 58. Primeira função de inferência

```python
def infer_one(image_uint8):
    pixels = np.asarray(
        image_uint8,
        dtype=np.float32
    ).reshape(-1) / 255.0

    if pixels.size != 784:
        raise ValueError(
            f"Esperava 784 pixels, recebeu {pixels.size}"
        )

    input_buffer[:] = float_to_q22_12(pixels)

    output_buffer[:] = 0

    input_buffer.flush()
    output_buffer.flush()

    # IMPORTANTE:
    # armar recepção primeiro.
    dma.recvchannel.transfer(output_buffer)

    # depois iniciar envio.
    dma.sendchannel.transfer(input_buffer)

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    output_buffer.invalidate()

    if np.any(output_buffer[10:] != 0):
        raise RuntimeError(
            "Padding words 10:16 não é zero. "
            "Revisar packing/wrapper."
        )

    logits = q22_12_to_float(
        output_buffer[:10]
    )

    pred = int(np.argmax(logits))

    return logits, pred
```

---

# 59. Ordem obrigatória das operações DMA

Usar:

```text
1. preencher input_buffer
2. limpar output_buffer
3. flush
4. iniciar recvchannel/S2MM
5. iniciar sendchannel/MM2S
6. esperar send
7. esperar recv
8. invalidate output
9. verificar padding
10. decodificar logits
11. argmax
```

Especialmente:

```text
RECV antes de SEND
```

Isso evita que a saída do acelerador encontre o S2MM ainda não armado.

---

# 60. Smoke test

Primeiro teste físico:

```text
1 imagem
```

Registrar:

```text
label verdadeira
classe Keras
classe hls4ml C++
classe FPGA

10 logits hls4ml C++
10 logits FPGA

padding words 10:16
status DMA
```

Não iniciar diretamente com as 10.000 imagens.

---

# 61. Critério para considerar smoke test aprovado

Esperado:

```text
DMA send completa
DMA recv completa
padding = zero
logits finitos
classe coerente
```

Depois comparar numericamente:

```text
FPGA vs hls4ml C++
```

Diferenças pequenas podem ocorrer devido à implementação e aos detalhes aritméticos.

Uma diferença grosseira normalmente indica:

```text
preprocessing
packing
ordem de lanes
sinal
escala
TLAST
ou DMA
```

---

# 62. Se o DMA travar

Verificar nesta ordem:

```text
1. recv/S2MM foi iniciado antes de send/MM2S?
2. input tem exatamente 3136 bytes?
3. output tem exatamente 64 bytes?
4. wrapper está gerando TLAST?
5. ap_start está em 1?
6. reset não está preso ativo?
7. TVALID/TREADY estão funcionando?
8. DMA reportou algum bit de erro?
```

Ponto crítico:

```text
m_axis_tlast <= accel_output_valid
```

e deve existir exatamente um beat de saída.

---

# 63. Se os logits estiverem errados

Verificar:

```text
imagem / 255.0
row-major
784 pixels
Q22.12
escala = 1024
22 bits válidos
bit 21 = sinal
words 0:10 = logits
words 10:16 = zero
```

Não aplicar Softmax para tentar “corrigir” uma classe errada.

Primeiro comparar os logits crus.

---

# 64. Verificação manual de uma palavra

Exemplo:

```python
word = int(output_buffer[0])

raw22 = word & 0x003FFFFF

if raw22 & 0x00200000:
    signed = raw22 - 0x00400000
else:
    signed = raw22

value = signed / 1024.0

print(
    hex(word),
    raw22,
    signed,
    value
)
```

---

# 65. Validação com 100 imagens

Depois do smoke test:

```text
100 imagens
```

Salvar por imagem:

```text
index
label
pred Keras
pred hls4ml C++
pred FPGA
acerto FPGA
concordância C++/FPGA
```

Também é recomendado salvar:

```text
MAE dos logits
max absolute error
```

para cada imagem.

Se houver divergências, investigar antes do teste de 10.000 imagens.

---

# 66. Validação oficial com 10.000 imagens

Condições:

```text
dataset = MNIST test
imagens = 10.000
batch = 1
preprocessing = x / 255
mesma ordem usada na validação C++
```

Métricas:

```text
acurácia FPGA
concordância FPGA × Keras
concordância FPGA × hls4ml C++
matriz de confusão
número de divergências
índices das divergências
```

Referência do C++:

```text
98,99%
```

Não assumir que a FPGA terá exatamente 98,99%.

Medir.

---

# 67. Benchmark — separar as métricas

Para o TCC não misturar:

```text
latência da inferência
latência end-to-end
throughput
potência
energia
```

---

# 68. Latência DMA/inferência

Buffers já preparados.

Medir apenas:

```text
recv transfer
send transfer
wait send
wait recv
```

Exemplo:

```python
import time

t0 = time.perf_counter_ns()

dma.recvchannel.transfer(output_buffer)
dma.sendchannel.transfer(input_buffer)

dma.sendchannel.wait()
dma.recvchannel.wait()

t1 = time.perf_counter_ns()

latency_ms = (t1 - t0) / 1e6
```

Essa métrica ainda inclui DMA e comunicação.

Não chamá-la de latência “somente núcleo HLS”.

---

# 69. Latência end-to-end

Incluir:

```text
preprocessing
quantização
cópia para buffer
flush
DMA
FPGA
invalidate
decode
argmax
```

Essa métrica representa a experiência completa da aplicação.

---

# 70. Latência estimada do núcleo pelo HLS

HLS havia estimado aproximadamente:

```text
9412 – 9460 ciclos
```

Com:

```text
100 MHz
10 ns/ciclo
```

aproximadamente:

```text
94,12 – 94,60 µs
```

Esse número é previsão do HLS.

Não comparar diretamente com a latência end-to-end do PYNQ sem deixar claro que são métricas diferentes.

---

# 71. Throughput

Realizar warm-up antes da medição.

Depois executar uma janela longa.

Exemplo:

```text
warm-up = pelo menos dezenas de inferências
medição = milhares de inferências ou janela temporal suficientemente longa
```

Calcular:

```text
FPS = número de inferências / tempo
```

Registrar separadamente:

```text
throughput DMA+FPGA
throughput end-to-end
```

---

# 72. Potência

Não utilizar o `report_power` vetorial-less do Vivado como substituto da medição física.

Durante implementação apareceu warning indicando que a atividade estimada dos resets poderia tornar a análise de potência imprecisa.

Para o TCC:

```text
potência Vivado = estimativa

potência física da ZCU104 = resultado experimental principal
```

Medir:

```text
idle
carga
```

Depois:

```text
P_dynamic = P_load - P_idle
```

---

# 73. Energia dinâmica por inferência

Usar:

```text
E_dynamic =
P_dynamic / throughput
```

ou equivalentemente integrar potência no tempo se houver telemetria temporal suficientemente detalhada.

Registrar unidade:

```text
J/inferência
mJ/inferência
```

---

# 74. Comparação com Vitis AI

Para comparação justa:

manter iguais sempre que possível:

```text
dataset
modelo lógico
preprocessing
número de imagens
batch
critério de acurácia
warm-up
janela de benchmark
definição de latência
definição de throughput
```

Documentar explicitamente diferenças inevitáveis:

```text
hls4ml:
hardware customizado

Vitis AI:
DPU programável

hls4ml:
precisão Q22.12 deste experimento

Vitis AI:
INT8 após quantização

hls4ml:
estrutura específica sintetizada

Vitis AI:
rede compilada para arquitetura DPU
```

---

# 75. Softmax e comparação

Este modelo utiliza:

```text
logits
```

No hardware:

```text
não existe Softmax
```

No software:

```python
pred = np.argmax(logits)
```

Para comparação com Vitis AI, preferir também comparar logits/argmax sempre que o fluxo permitir e evitar incluir Softmax apenas em um dos lados.

---

# 76. O que NÃO alterar antes do teste físico

Não alterar:

```text
precisão
Reuse Factors
clock
wrapper
larguras DMA
preprocessing
modelo
pesos
Softmax
```

O bitstream atual representa uma configuração experimental congelada.

Primeiro validar esta versão na placa.

Depois criar novas variantes experimentais.

---

# 77. Identidade desta configuração

Nome recomendado:

```text
LeNet_MNIST_hls4ml_Q22.12_RF5-50-64-60-42_Resource_LineBuffer_100MHz_ZCU104
```

Abreviação:

```text
LENET_Q22_12_RF_CAP64_100M
```

---

# 78. Checklist antes de fechar o computador / apagar arquivos

Verificar:

```text
[ ] README_FINAL salvo
[ ] bitstream copiado para hls4ml/hardware/deploy
[ ] HWH copiado para hls4ml/hardware/deploy
[ ] bit e HWH possuem mesmo basename
[ ] wrapper VHDL preservado
[ ] IP repo preservado
[ ] modelo H5 preservado
[ ] configurações hls4ml preservadas
[ ] scripts de geração preservados
[ ] relatórios de síntese preservados
[ ] relatórios pós-route preservados
[ ] system_final_bd.tcl exportado
[ ] project_final.tcl exportado
[ ] metadata.txt criado
[ ] opcionalmente projeto Vivado completo arquivado
```

---

# 79. Comando rápido de verificação do deployment

```bash
cd "/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"

DEPLOY="hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz"

echo "========== DEPLOY LENET =========="

find "$DEPLOY" \
    -maxdepth 2 \
    -type f \
    -printf '%p  %k KB\n' \
    | sort
```

---

# 80. Hashes recomendados

Depois de copiar os artefatos:

```bash
cd "/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"

DEPLOY="hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz"

sha256sum \
    "$DEPLOY/lenet_zcu104.bit" \
    "$DEPLOY/lenet_zcu104.hwh" \
    "$DEPLOY/lenet_dma_wrapper.vhd"
```

Guardar a saída em:

```bash
sha256sum \
    "$DEPLOY/lenet_zcu104.bit" \
    "$DEPLOY/lenet_zcu104.hwh" \
    "$DEPLOY/lenet_dma_wrapper.vhd" \
    > "$DEPLOY/SHA256SUMS.txt"
```

Isso permite garantir no futuro que o arquivo testado na placa é exatamente o mesmo artefato desta implementação.

---

# 81. Status final desta etapa

Em 08/09/2026, a implementação alcançou:

```text
Keras model             PASS
hls4ml conversion       PASS
C++ validation          PASS
Vitis HLS synthesis     PASS
IP export               PASS
Vivado IP integration   PASS
Block Design            PASS
Validate Design         PASS
Integrated synthesis    PASS
Placement               PASS
Routing                 PASS
Setup timing            PASS
Hold timing             PASS
DRC Errors              0
DRC Critical Warnings   0
Bitstream               PASS
HWH                     AVAILABLE
Physical inference      PENDING
```

---

# 82. Resumo do hardware final

```text
BOARD:
ZCU104

FPGA:
XCZU7EV-FFVC1156-2-E

CLOCK:
100 MHz

MODEL:
LeNet / MNIST

PARAMETERS:
44.426

PRECISION:
Q22.12
ap_fixed<22,12,AP_RND_CONV,AP_SAT>

HLS4ML:
1.3.0

STRATEGY:
Resource

IO:
io_stream

CONV:
LineBuffer

REUSE FACTORS:
5 / 50 / 64 / 60 / 42

INPUT:
784 × uint32
3136 bytes

OUTPUT:
16 × uint32
64 bytes

VALID LOGITS:
words 0:10

PADDING:
words 10:16 = zero

DMA CONTROL:
0xA0000000

DDR PATH:
S_AXI_HPC0_FPD

LUT:
154.485 / 230.400
67,05%

FF:
96.655 / 460.800
20,98%

DSP:
746 / 1.728
43,17%

BRAM:
125 / 312 tiles equivalentes
40,06%

PHYSICAL CLB:
28.154 / 28.800
97,76%

SETUP SLACK:
+0.033 ns

HOLD SLACK:
+0.010 ns

ROUTING ERRORS:
0

DRC ERRORS:
0

DRC CRITICAL WARNINGS:
0
```

---

# 83. Próximo ponto de retomada

Ao retomar este projeto, **não abrir o Vivado para alterar o hardware**.

A próxima etapa é:

```text
1. copiar BIT + HWH para a ZCU104;
2. iniciar a imagem PYNQ;
3. abrir Jupyter;
4. carregar Overlay;
5. localizar axi_dma_0;
6. alocar buffers 784 e 16 uint32;
7. carregar uma imagem MNIST conhecida;
8. normalizar /255;
9. converter para Q22.12;
10. iniciar S2MM;
11. iniciar MM2S;
12. aguardar DMA;
13. verificar padding;
14. decodificar os 10 logits;
15. argmax;
16. comparar com hls4ml C++;
```

Somente depois desse smoke test deve começar a validação de 100 e 10.000 imagens.

---

# 84. Regra experimental

Qualquer nova configuração deve possuir um identificador próprio e seus próprios artefatos:

```text
config
logs HLS
IP
relatórios
BIT
HWH
resultados físicos
```

Não sobrescrever esta versão quando iniciar experimentos com:

```text
Reuse Factor
precisão
HGQ
QKeras
FIFO optimization
Strategy
IOType
ou outras variantes
```

Esta implementação deve permanecer como baseline reproduzível.

---

# 85. Conclusão

A etapa de implementação física no Vivado foi concluída com sucesso.

O sistema final integra:

```text
Zynq UltraScale+ PS
+
AXI DMA
+
SmartConnect
+
DDR via HPC0
+
wrapper AXIS
+
LeNet hls4ml
```

e conseguiu ser implementado na ZCU104 em:

```text
100 MHz
```

apesar da elevada ocupação física de CLBs.

O bitstream foi gerado sem erros e o HWH correspondente está disponível.

A partir deste ponto, a próxima etapa do fluxo não é mais síntese ou implementação, mas sim:

```text
VALIDAÇÃO FÍSICA NA ZCU104 COM PYNQ
```

seguida de:

```text
acurácia
latência
throughput
potência
energia por inferência
```

para posterior comparação controlada com o fluxo Vitis AI/DPU.

**Não modificar o hardware atual antes de concluir o smoke test e preservar os artefatos desta versão.**
