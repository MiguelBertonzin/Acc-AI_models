# Handoff completo — integração da LeNet hls4ml no Vivado/ZCU104

> Documento autocontido para continuar, com auxílio do ChatGPT Web, a integração
> do IP LeNet já exportado pelo Vitis HLS 2024.2. O objetivo é construir o Block
> Design, implementar o hardware, gerar bitstream e preparar a primeira inferência
> na ZCU104 sem alterar o modelo nem o IP promovido.

## 1. Objetivo e fronteira deste documento

O sistema deve:

1. carregar uma imagem MNIST de 28 × 28 em memória DDR;
2. converter cada pixel para Q22.12;
3. enviar os 784 pixels ao PL por AXI DMA MM2S;
4. executar a LeNet no IP hls4ml;
5. receber os dez logits por AXI DMA S2MM;
6. aplicar `argmax` no ARM, pois não existe Softmax no modelo;
7. validar acurácia e, depois, medir latência, throughput e potência.

Este documento cobre a arquitetura recomendada e a sequência de construção.
Ainda não existe Block Design LeNet implementado nem bitstream LeNet. Não
confundir os resultados pós-síntese do IP isolado com resultados do sistema
integrado ou da placa.

## 2. Estado atual real

### Concluído

- [x] modelo Keras/H5 treinado;
- [x] modelo auditado: 44.426 parâmetros, saída linear em logits;
- [x] conversão para hls4ml 1.3.0;
- [x] precisão Q22.12;
- [x] ReuseFactor customizado por camada, limitado a 64;
- [x] validação C++ nas 10.000 imagens de teste do MNIST;
- [x] C synthesis no Vitis HLS 2024.2;
- [x] export do IP para catálogo;
- [x] síntese do RTL no Vivado 2024.2;
- [x] relatório HLS versus Vivado.

### Ainda não concluído

- [ ] wrapper AXI DMA específico para a LeNet;
- [ ] projeto Vivado da LeNet;
- [ ] Block Design PS–DMA–wrapper–LeNet;
- [ ] síntese do sistema integrado;
- [ ] place-and-route;
- [ ] fechamento de timing;
- [ ] bitstream, HWH e XSA;
- [ ] inferência física na ZCU104;
- [ ] validação de 10.000 imagens na placa;
- [ ] benchmark e telemetria física.

## 3. Identidade dos artefatos

Workspace:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet
```

Modelo:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/lenet_mnist_final.h5
```

IP promovido:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/ip_repo/lenet_mnist_cap64_hls_v1_0
```

VLNV:

```text
xilinx.com:hls:lenet_mnist_cap64_hls:1.0
```

Dispositivo:

```text
ZCU104
xczu7ev-ffvc1156-2-e
xilinx.com:zcu104:part0:1.1
```

Hashes SHA-256:

```text
H5:
7d052aa27f18565af0a948e656cbd23eac842374ac73c0aaf18e98d5d913226c

component.xml:
04befcac8962193abbcf35b3f60621d8c19582565f53e0f3fb960eab9c7a168d

RTL top lenet_mnist_cap64_hls.v:
120d030bb96fdb7935fb49a5bc9d83d3e8321621cf7b524d6f6688cd7611c936
```

O diretório do IP deve ser tratado como somente leitura. Wrapper, projeto
Vivado, relatórios e bitstream devem ficar fora dele.

## 4. Arquitetura do modelo

```text
Entrada 28×28×1
    |
Conv2D: 6 filtros 5×5, valid
    | 24×24×6
ReLU
    |
MaxPooling2D 2×2
    | 12×12×6
Conv2D: 16 filtros 5×5, valid
    | 8×8×16
ReLU
    |
MaxPooling2D 2×2
    | 4×4×16
Flatten
    | 256
Dense 120 + ReLU
    |
Dense 84 + ReLU
    |
Dense 10, ativação linear
    |
10 logits; argmax executado no software
```

| Camada parametrizada | Parâmetros |
|---|---:|
| Conv1 | 156 |
| Conv2 | 2.416 |
| Dense1 | 30.840 |
| Dense2 | 10.164 |
| Saída | 850 |
| **Total** | **44.426** |

Não há Softmax. Não adicionar Softmax ao wrapper ou ao Block Design. Para
classificação, `argmax(logits)` produz a mesma classe que
`argmax(softmax(logits))`.

## 5. Configuração hls4ml congelada

| Propriedade | Valor |
|---|---|
| hls4ml | 1.3.0 |
| Backend | Vitis |
| Vitis HLS/Vivado | 2024.2 |
| Clock | 10 ns / 100 MHz |
| IOType | `io_stream` |
| Strategy | `Resource` |
| ConvImplementation | `LineBuffer` |
| Precisão | `ap_fixed<22,12,AP_RND_CONV,AP_SAT>` |
| Batch | 1 |
| Softmax | não |
| RF conv1/conv2/dense1/dense2/output | `5/50/64/60/42` |

Validação C++:

| Métrica | Resultado |
|---|---:|
| Imagens | 10.000 |
| Acurácia Keras | 98,98% |
| Acurácia hls4ml C++ | 98,99% |
| Concordância top-1 Keras/hls4ml | 99,99% |
| MAE dos logits | 0,01564898 |
| Erro absoluto máximo dos logits | 0,09856367 |

## 6. Recursos atuais do IP isolado

Resultados do Vivado depois de `synth_design` e `opt_design`:

| Recurso | Usado | Disponível | Utilização |
|---|---:|---:|---:|
| LUT | 141.417 | 230.400 | 61,38% |
| FF | 81.051 | 460.800 | 17,59% |
| DSP48E2 | 746 | 1.728 | 43,17% |
| Block RAM Tile | 114 | 312 | 36,54% |
| URAM | 0 | 96 | 0% |
| CARRY8 | 15.933 | 28.800 | 55,32% |

O Vitis HLS havia estimado 121,69% das LUTs. O Vivado mostrou que o IP
isolado cabe, mas ainda faltam DMA, SmartConnect, PS, wrapper e roteamento.
O uso relativamente alto de LUT/CARRY8 pode causar congestionamento. Somente o
relatório pós-route do sistema completo permite afirmar que a implementação
fecha.

Relatório:

```text
hls4ml/reports/cap64_q22_12_rate_balanced/HLS_VS_VIVADO_UTILIZATION.md
```

## 7. Contrato externo exato do IP

### 7.1 Clock, reset e controle

```text
ap_clk       input
ap_rst_n     input, ativo em nível baixo
ap_start     input
ap_done      output
ap_ready     output
ap_idle      output
```

O protocolo é `ap_ctrl_hs`. O acelerador não possui interface AXI4-Lite de
controle.

Na primeira implementação:

```text
ap_start = 1 permanentemente
```

O software controla apenas o DMA. Manter `ap_start` alto permite que o
acelerador aceite uma nova imagem quando estiver pronto.

### 7.2 AXI4-Stream de entrada

```text
input_layer_TDATA  [31:0]
input_layer_TVALID
input_layer_TREADY
```

Propriedades confirmadas no `component.xml`:

```text
modo: slave
TDATA_NUM_BYTES = 4
sem TLAST, TKEEP, TSTRB, TUSER, TID ou TDEST
```

Uma imagem usa exatamente 784 handshakes:

```text
transferência aceita = TVALID && TREADY
ordem = row-major, 28 linhas × 28 colunas
um pixel por beat
```

O RTL usa somente `input_layer_TDATA[21:0]`. Os bits `[31:22]` devem ser
zero no buffer enviado pelo ARM.

### 7.3 AXI4-Stream de saída

```text
layer13_out_TDATA  [319:0]
layer13_out_TVALID
layer13_out_TREADY
```

Propriedades:

```text
modo: master
TDATA_NUM_BYTES = 40
um beat por inferência
sem TLAST, TKEEP, TSTRB, TUSER, TID ou TDEST
```

O AXI DMA S2MM precisa de `TLAST` para terminar o pacote. Portanto a saída do
IP não deve ser ligada diretamente ao S2MM. O wrapper descrito adiante:

- expande 320 para 512 bits;
- preenche os 192 bits superiores com zero;
- gera `TLAST` no único beat de saída;
- preserva `TVALID/TREADY`;
- mantém `ap_start=1`.

## 8. Formato numérico e packing

### 8.1 Q22.12

```text
W = 22 bits totais
I = 12 bits incluindo sinal
F = 10 bits fracionários
escala = 2^10 = 1024
faixa raw = [-2^21, 2^21-1]
```

Conversão compatível com `AP_RND_CONV` usando NumPy:

```python
import numpy as np

SCALE = 1 << 10
MASK22 = (1 << 22) - 1
MIN_RAW = -(1 << 21)
MAX_RAW = (1 << 21) - 1

def float_to_q22_12(values):
    x = np.asarray(values, dtype=np.float64)
    raw = np.rint(x * SCALE)             # ties-to-even
    raw = np.clip(raw, MIN_RAW, MAX_RAW)
    return (raw.astype(np.int64) & MASK22).astype(np.uint32)

def q22_12_to_float(words):
    raw = np.asarray(words, dtype=np.uint32) & MASK22
    signed = raw.astype(np.int64)
    signed[(raw & (1 << 21)) != 0] -= 1 << 22
    return signed.astype(np.float32) / SCALE
```

### 8.2 Preprocessing MNIST

O preprocessing utilizado no treino, CPU, GPU e validação hls4ml foi:

```python
image_float = image_uint8.astype(np.float32) / 255.0
```

Não aplicar média/desvio padrão, inversão de cor ou redimensionamento. A ordem é
`image.reshape(-1)` em row-major.

### 8.3 Buffer de entrada

```python
pixels = image_uint8.astype(np.float32).reshape(-1) / 255.0
input_words = float_to_q22_12(pixels)
```

Contrato físico recomendado:

```text
shape = (784,)
dtype = uint32
tamanho DMA = 784 × 4 = 3.136 bytes
bits [21:0] = pixel Q22.12
bits [31:22] = zero
```

### 8.4 Buffer de saída

O IP produz dez lanes de 32 bits:

```text
logit 0 -> TDATA[31:0],   payload em [21:0]
logit 1 -> TDATA[63:32],  payload em [53:32]
...
logit 9 -> TDATA[319:288], payload em [309:288]
```

O sinal de números negativos está no bit 21 de cada lane, não no bit 31.
Depois da adaptação para 512 bits:

```text
shape = (16,)
dtype = uint32
tamanho DMA = 16 × 4 = 64 bytes
palavras [0:10] = dez logits
palavras [10:16] = padding zero
```

Decodificação:

```python
logits = q22_12_to_float(output_buffer[:10])
prediction = int(np.argmax(logits))
assert np.all(output_buffer[10:16] == 0)
```

## 9. Arquitetura recomendada do Block Design

```text
                          Zynq UltraScale+ MPSoC
                   +--------------------------------+
                   | ARM / Linux / PYNQ             |
                   |                                |
controle ----------| M_AXI_HPM0_FPD                 |
DDR ---------------| S_AXI_HPC0_FPD                 |
                   +-------+------------------^-----+
                           |                  |
                           |                  | MM2S/S2MM memory mapped
                    +------v------+    +------|------+
                    | SmartConnect|    | SmartConnect|
                    | controle    |    | memória     |
                    +------+------+    +------^------+
                           |                  |
                           v AXI-Lite         |
                     +-----------------------------+
                     | AXI DMA, simple mode        |
                     | MM2S 32-bit | S2MM 512-bit |
                     +-------+------------^--------+
                             |            |
                         AXIS 32       AXIS 512 + TLAST
                             v            |
                     +-----------------------------+
                     | lenet_dma_wrapper           |
                     | 32→32; 320→512; TLAST       |
                     | ap_start permanente         |
                     +-------+------------^--------+
                             |            |
                         AXIS 32       AXIS 320
                             v            |
                     +-----------------------------+
                     | lenet_mnist_cap64_hls 1.0   |
                     +-----------------------------+
```

Blocos:

```text
zynq_ultra_ps_e_0
rst_ps8_0_100M
axi_smc_ctrl
axi_smc_mem
axi_dma_0
lenet_dma_wrapper_0
lenet_mnist_cap64_hls_0
```

Para o primeiro bitstream, usar polling e deixar interrupções do DMA
desconectadas. Interrupções podem ser adicionadas depois que a inferência
funcionar.

## 10. Configuração recomendada

### 10.1 Processing System

Aplicar o preset da ZCU104 e habilitar:

```text
M_AXI_HPM0_FPD = ON    # ARM controla registradores do DMA
S_AXI_HPC0_FPD = ON    # DMA acessa DDR
pl_clk0 = 100 MHz
pl_resetn0 = ON
```

Uma única porta HPM é suficiente para o AXI-Lite do DMA.

### 10.2 Processor System Reset

```text
pl_clk0    -> slowest_sync_clk
pl_resetn0 -> ext_reset_in
```

Usar `peripheral_aresetn` para:

- `axi_dma_0/axi_resetn`;
- SmartConnects;
- `lenet_dma_wrapper_0/ap_rst_n`;
- `lenet_mnist_cap64_hls_0/ap_rst_n`.

Não inverter o reset: `ap_rst_n` já é ativo baixo.

### 10.3 SmartConnect de controle

```text
NUM_SI = 1
NUM_MI = 1

PS/M_AXI_HPM0_FPD -> S00_AXI
M00_AXI -> DMA/S_AXI_LITE
```

### 10.4 SmartConnect de memória

```text
NUM_SI = 2
NUM_MI = 1

DMA/M_AXI_MM2S -> S00_AXI
DMA/M_AXI_S2MM -> S01_AXI
M00_AXI -> PS/S_AXI_HPC0_FPD
```

O SmartConnect faz conversão de largura quando necessário.

### 10.5 AXI DMA

Primeira configuração recomendada:

```text
Scatter Gather                 OFF
Micro DMA                      OFF
Address Width                  32
Buffer Length Register Width   26

MM2S memory width              128
MM2S stream width              32
MM2S burst                     64
MM2S DRE                       ON

S2MM memory width              512
S2MM stream width              512
S2MM burst                     64
S2MM DRE                       ON
```

Se o Vivado 2024.2 não oferecer exatamente um desses nomes na GUI, inspecionar
`report_property [get_bd_cells axi_dma_0]` antes de alterar Tcl. Não inventar
nomes de propriedades.

## 11. Wrapper VHDL de referência

Criar:

```text
hls4ml/hardware/src/lenet_dma_wrapper.vhd
```

Conteúdo proposto:

```vhdl
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity lenet_dma_wrapper is
    Port (
        ap_clk   : in  std_logic;
        ap_rst_n : in  std_logic;

        -- AXI4-Stream vindo do DMA MM2S
        s_axis_tdata  : in  std_logic_vector(31 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;
        s_axis_tlast  : in  std_logic;

        -- AXI4-Stream enviado ao DMA S2MM
        m_axis_tdata  : out std_logic_vector(511 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic;
        m_axis_tlast  : out std_logic;

        -- Controle ap_ctrl_hs do acelerador
        accel_ap_start : out std_logic;
        accel_ap_done  : in  std_logic;
        accel_ap_idle  : in  std_logic;
        accel_ap_ready : in  std_logic;

        -- Stream de entrada do IP LeNet
        accel_input_data  : out std_logic_vector(31 downto 0);
        accel_input_valid : out std_logic;
        accel_input_ready : in  std_logic;

        -- Stream de saída do IP LeNet
        accel_output_data  : in  std_logic_vector(319 downto 0);
        accel_output_valid : in  std_logic;
        accel_output_ready : out std_logic
    );
end lenet_dma_wrapper;

architecture rtl of lenet_dma_wrapper is
    attribute X_INTERFACE_INFO : string;
    attribute X_INTERFACE_PARAMETER : string;

    attribute X_INTERFACE_INFO of ap_clk : signal is
        "xilinx.com:signal:clock:1.0 ap_clk CLK";
    attribute X_INTERFACE_PARAMETER of ap_clk : signal is
        "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET ap_rst_n";

    attribute X_INTERFACE_INFO of ap_rst_n : signal is
        "xilinx.com:signal:reset:1.0 ap_rst_n RST";
    attribute X_INTERFACE_PARAMETER of ap_rst_n : signal is
        "POLARITY ACTIVE_LOW";

    attribute X_INTERFACE_INFO of s_axis_tdata : signal is
        "xilinx.com:interface:axis:1.0 s_axis TDATA";
    attribute X_INTERFACE_INFO of s_axis_tvalid : signal is
        "xilinx.com:interface:axis:1.0 s_axis TVALID";
    attribute X_INTERFACE_INFO of s_axis_tready : signal is
        "xilinx.com:interface:axis:1.0 s_axis TREADY";
    attribute X_INTERFACE_INFO of s_axis_tlast : signal is
        "xilinx.com:interface:axis:1.0 s_axis TLAST";
    attribute X_INTERFACE_PARAMETER of s_axis_tdata : signal is
        "TDATA_NUM_BYTES 4, HAS_TKEEP 0, HAS_TSTRB 0, HAS_TLAST 1";

    attribute X_INTERFACE_INFO of m_axis_tdata : signal is
        "xilinx.com:interface:axis:1.0 m_axis TDATA";
    attribute X_INTERFACE_INFO of m_axis_tvalid : signal is
        "xilinx.com:interface:axis:1.0 m_axis TVALID";
    attribute X_INTERFACE_INFO of m_axis_tready : signal is
        "xilinx.com:interface:axis:1.0 m_axis TREADY";
    attribute X_INTERFACE_INFO of m_axis_tlast : signal is
        "xilinx.com:interface:axis:1.0 m_axis TLAST";
    attribute X_INTERFACE_PARAMETER of m_axis_tdata : signal is
        "TDATA_NUM_BYTES 64, HAS_TKEEP 0, HAS_TSTRB 0, HAS_TLAST 1";
begin
    -- O IP inicia automaticamente quando há dados e está pronto.
    accel_ap_start <= '1';

    -- A entrada já possui a largura física do IP.
    accel_input_data  <= s_axis_tdata;
    accel_input_valid <= s_axis_tvalid;
    s_axis_tready     <= accel_input_ready;

    -- s_axis_tlast marca o fim do pacote DMA, mas o IP não possui TLAST.
    -- A transferência contém exatamente 784 beats, controlados pelo BTT.

    -- Um único beat de saída: dez lanes de 32 bits e padding até 512.
    m_axis_tdata(319 downto 0)   <= accel_output_data;
    m_axis_tdata(511 downto 320) <= (others => '0');
    m_axis_tvalid                <= accel_output_valid;
    accel_output_ready           <= m_axis_tready;

    -- Deve permanecer alto junto com TVALID mesmo sob backpressure.
    m_axis_tlast <= accel_output_valid;
end rtl;
```

`accel_ap_done`, `accel_ap_idle` e `accel_ap_ready` são conectados para
observabilidade, embora a primeira versão não os use na lógica. Não inserir
registradores apenas em `TDATA` ou apenas em `TVALID`. Se timing exigir
pipeline, usar AXI Register Slice ou skid buffer que preserve o handshake.

## 12. Construção manual na GUI do Vivado

### 12.1 Antes de abrir

Criar o projeto em caminho sem espaços:

```text
/home/miguel/lenet_zcu104_vivado
```

O repositório do IP pode permanecer no workspace com espaços.

### 12.2 Criar projeto

1. abrir Vivado 2024.2;
2. selecionar **Create Project**;
3. nome sugerido: `lenet_zcu104`;
4. escolher **RTL Project**;
5. selecionar a board ZCU104 ou o part `xczu7ev-ffvc1156-2-e`;
6. em **Project Settings → IP → Repository**, adicionar:

```text
/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml/ip_repo/lenet_mnist_cap64_hls_v1_0
```

7. confirmar que o catálogo mostra
   `xilinx.com:hls:lenet_mnist_cap64_hls:1.0`;
8. adicionar `lenet_dma_wrapper.vhd` como Design Source.

### 12.3 Criar Block Design

1. **Create Block Design**, nome `system`;
2. adicionar **Zynq UltraScale+ MPSoC**;
3. executar **Run Block Automation** com o board preset;
4. habilitar HPM0 FPD, HPC0 FPD, PL clock 100 MHz e PL reset;
5. adicionar **Processor System Reset**;
6. adicionar dois **SmartConnect**;
7. adicionar **AXI DMA** e configurar como na seção 10.5;
8. adicionar o IP `lenet_mnist_cap64_hls`;
9. adicionar o wrapper por **Add Module**;
10. fazer as conexões conforme as seções seguintes.

### 12.4 Conectar caminhos AXI

Controle:

```text
PS M_AXI_HPM0_FPD
 -> axi_smc_ctrl/S00_AXI
 -> axi_smc_ctrl/M00_AXI
 -> axi_dma_0/S_AXI_LITE
```

Memória:

```text
axi_dma_0/M_AXI_MM2S -> axi_smc_mem/S00_AXI
axi_dma_0/M_AXI_S2MM -> axi_smc_mem/S01_AXI
axi_smc_mem/M00_AXI  -> PS/S_AXI_HPC0_FPD
```

Streams:

```text
axi_dma_0/M_AXIS_MM2S -> lenet_dma_wrapper_0/s_axis
lenet_dma_wrapper_0/m_axis -> axi_dma_0/S_AXIS_S2MM
```

Wrapper para LeNet:

```text
wrapper/accel_input_data  -> LeNet/input_layer_TDATA
wrapper/accel_input_valid -> LeNet/input_layer_TVALID
LeNet/input_layer_TREADY  -> wrapper/accel_input_ready

LeNet/layer13_out_TDATA   -> wrapper/accel_output_data
LeNet/layer13_out_TVALID  -> wrapper/accel_output_valid
wrapper/accel_output_ready -> LeNet/layer13_out_TREADY

wrapper/accel_ap_start -> LeNet/ap_start
LeNet/ap_done  -> wrapper/accel_ap_done
LeNet/ap_idle  -> wrapper/accel_ap_idle
LeNet/ap_ready -> wrapper/accel_ap_ready
```

### 12.5 Clock

Conectar `pl_clk0` a:

- `rst_ps8_0_100M/slowest_sync_clk`;
- clocks HPM0/HPC0 do PS;
- clocks dos dois SmartConnects;
- todos os clocks AXI do DMA;
- `lenet_dma_wrapper_0/ap_clk`;
- `lenet_mnist_cap64_hls_0/ap_clk`.

Não deixar nenhum clock AXI desconectado.

### 12.6 Reset

```text
PS/pl_resetn0 -> rst_ps8_0_100M/ext_reset_in

rst/peripheral_aresetn
 -> DMA/axi_resetn
 -> SmartConnects/aresetn
 -> wrapper/ap_rst_n
 -> LeNet/ap_rst_n
```

### 12.7 Endereço

No **Address Editor**, usar **Auto Assign Address**. O único periférico
controlado pelo ARM é o AXI-Lite do DMA. Endereço sugerido, se estiver livre:

```text
axi_dma_0/S_AXI_LITE
base  = 0xA0000000
range = 64 KiB
```

Não hardcodear esse endereço no benchmark antes de conferir o `.hwh` ou
`overlay.ip_dict`.

### 12.8 Validar

Executar:

1. **Validate Design**;
2. criar HDL wrapper do BD, **Let Vivado manage wrapper**;
3. **Generate Output Products**;
4. revisar warnings antes da síntese.

Erros de largura, clock, reset ou interface devem ser corrigidos; não usar
`force` para esconder incompatibilidades.

## 13. Tcl de auditoria e esqueleto

No Vivado Tcl Console:

```tcl
set workspace_root [file normalize {/home/miguel/Downloads/Plano testes TCC/LeNet}]
set project_root   [file normalize {/home/miguel/lenet_zcu104_vivado}]
set ip_repo [file normalize [file join $workspace_root hls4ml ip_repo lenet_mnist_cap64_hls_v1_0]]
set wrapper_src [file normalize [file join $workspace_root hls4ml hardware src lenet_dma_wrapper.vhd]]
set expected_vlnv {xilinx.com:hls:lenet_mnist_cap64_hls:1.0}

foreach required [list +    [file join $ip_repo component.xml] +    [file join $ip_repo hdl verilog lenet_mnist_cap64_hls.v] +    $wrapper_src] {
    if {![file exists $required]} {
        error "Arquivo obrigatório ausente: $required"
    }
}

create_project lenet_zcu104 $project_root -part xczu7ev-ffvc1156-2-e
set_property BOARD_PART xilinx.com:zcu104:part0:1.1 [current_project]
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog -rebuild

set defs [get_ipdefs -all -quiet $expected_vlnv]
if {[llength $defs] != 1} {
    error "Esperava exatamente um IP $expected_vlnv; encontrado: $defs"
}

add_files -norecurse $wrapper_src
update_compile_order -fileset sources_1
puts "IP_OK=$defs"
puts "PART=[get_property PART [current_project]]"
puts "BOARD=[get_property BOARD_PART [current_project]]"
```

Depois de criar o BD, comandos de auditoria úteis:

```tcl
validate_bd_design
report_property [get_bd_cells axi_dma_0]
get_bd_cells
get_bd_intf_nets
get_bd_nets

foreach pin {
    lenet_mnist_cap64_hls_0/input_layer_TDATA
    lenet_mnist_cap64_hls_0/layer13_out_TDATA
    lenet_dma_wrapper_0/accel_input_data
    lenet_dma_wrapper_0/accel_output_data
} {
    puts "$pin LEFT=[get_property LEFT [get_bd_pins $pin]] RIGHT=[get_property RIGHT [get_bd_pins $pin]]"
}
```

Esperado:

```text
entrada LeNet e wrapper = 31:0
saída LeNet e wrapper   = 319:0
```

Os nomes e versões exatos de propriedades do PS/DMA podem variar com o catálogo.
Antes de automatizar todo o BD, configurar os blocos na GUI ou enviar ao
ChatGPT Web a saída de `report_property`.

## 14. Síntese, implementação e relatórios

### 14.1 Síntese integrada

Executar **Run Synthesis**. Depois:

```tcl
open_run synth_1
file mkdir reports
report_utilization -file reports/utilization_post_synth.rpt
report_utilization -hierarchical -file reports/utilization_hierarchical_post_synth.rpt
report_timing_summary -file reports/timing_post_synth.rpt
report_clock_utilization -file reports/clocks_post_synth.rpt
```

Conferir separadamente:

- recursos do IP LeNet;
- recursos do DMA/SmartConnect/wrapper;
- total do sistema;
- clocks e endpoints sem clock;
- margem restante de LUT, DSP, BRAM e CARRY.

### 14.2 Place-and-route

Executar **Run Implementation**. Depois:

```tcl
open_run impl_1
report_utilization -file reports/utilization_post_route.rpt
report_utilization -hierarchical -file reports/utilization_hierarchical_post_route.rpt
report_timing_summary -delay_type min_max -report_unconstrained +    -file reports/timing_post_route.rpt
report_route_status -file reports/route_status_post_route.rpt
report_drc -file reports/drc_post_route.rpt
check_timing -verbose -file reports/check_timing_post_route.rpt
report_clock_utilization -file reports/clocks_post_route.rpt
report_design_analysis -congestion -file reports/congestion_post_route.rpt
```

Critérios mínimos:

```text
WNS >= 0
TNS = 0
WHS >= 0
THS = 0
unrouted nets = 0
partially routed nets = 0
DRC errors = 0
no_clock = 0 para endpoints internos relevantes
```

Não reduzir o clock ou mudar RF/precisão silenciosamente. Qualquer alteração
gera uma nova configuração experimental e exige nova validação.

### 14.3 Potência

`report_power` pós-route é uma estimativa. Para torná-la menos frágil:

- garantir clock de 100 MHz definido;
- fornecer atividade de comutação realista, se possível;
- registrar condições e assumptions do relatório.

Para o benchmark do TCC, usar também telemetria física da ZCU104. Não substituir
potência medida na placa por uma estimativa vectorless do Vivado.

## 15. Bitstream, HWH e XSA

Depois de fechar implementação:

1. **Generate Bitstream**;
2. **File → Export → Export Hardware**, incluindo bitstream;
3. preservar o `.bit`, `.hwh` e `.xsa`;
4. copiar `.bit` e `.hwh` com o mesmo nome-base.

Estrutura sugerida:

```text
hls4ml/vivado_zcu104_lenet/
├── README.md
├── src/
│   └── lenet_dma_wrapper.vhd
├── tcl/
├── logs/
├── reports/
├── output/
│   ├── lenet_zcu104.bit
│   ├── lenet_zcu104.hwh
│   └── lenet_zcu104.xsa
├── pynq/
├── results_zcu104/
└── manifests/
```

Calcular SHA-256 dos três artefatos e registrar Vivado, precisão, RFs, clock e
hash do H5.

## 16. Primeiro teste no PYNQ

```python
import time
import numpy as np
from pynq import Overlay, allocate

overlay = Overlay("lenet_zcu104.bit")
print(overlay.ip_dict)

# Ajustar os nomes conforme overlay.ip_dict.
dma = overlay.axi_dma_0

input_buffer = allocate(shape=(784,), dtype=np.uint32)
output_buffer = allocate(shape=(16,), dtype=np.uint32)

def float_to_q22_12(values):
    x = np.asarray(values, dtype=np.float64)
    raw = np.rint(x * 1024.0)
    raw = np.clip(raw, -(1 << 21), (1 << 21) - 1)
    return (raw.astype(np.int64) & 0x003FFFFF).astype(np.uint32)

def q22_12_to_float(words):
    raw = np.asarray(words, dtype=np.uint32) & 0x003FFFFF
    signed = raw.astype(np.int64)
    signed[(raw & 0x00200000) != 0] -= 0x00400000
    return signed.astype(np.float32) / 1024.0

def infer_one(image_uint8, timeout_s=2.0):
    pixels = np.asarray(image_uint8, dtype=np.float32).reshape(-1) / 255.0
    if pixels.size != 784:
        raise ValueError(f"Esperava 784 pixels; recebeu {pixels.size}")

    input_buffer[:] = float_to_q22_12(pixels)
    output_buffer[:] = 0
    input_buffer.flush()
    output_buffer.flush()

    # Armar a recepção antes do envio evita bloquear a saída do acelerador.
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.transfer(input_buffer)

    dma.sendchannel.wait()
    dma.recvchannel.wait()
    output_buffer.invalidate()

    if np.any(output_buffer[10:] != 0):
        raise RuntimeError("Padding de saída não é zero; revisar packing/wrapper")

    logits = q22_12_to_float(output_buffer[:10])
    return logits, int(np.argmax(logits))
```

O argumento `timeout_s` deve ser implementado no teste robusto usando status
do DMA ou thread/loop com prazo; não deixar espera infinita durante depuração.

Sequência obrigatória:

```text
1. preparar buffers
2. armar S2MM/recv
3. iniciar MM2S/send
4. aguardar send
5. aguardar recv
6. invalidar cache de saída
7. decodificar os dez logits
```

## 17. Depuração

### DMA não termina

Verificar:

- `TLAST` do wrapper no único beat de saída;
- S2MM armado antes de MM2S;
- reset ativo por engano;
- `ap_start=1`;
- 3.136 bytes enviados;
- 64 bytes recebidos;
- `TVALID/TREADY` nos dois lados;
- bits de erro do DMA.

### Predição errada

Verificar:

- divisão por 255;
- ordem row-major;
- exatamente 784 pixels;
- bits superiores de cada entrada zerados;
- sinal no bit 21;
- escala 1024;
- lanes de saída de 32 bits;
- uso somente das primeiras dez palavras;
- ausência de Softmax não é erro: usar argmax.

### Erro ao conectar interfaces

Verificar:

- wrapper aparece como Module Reference;
- AXIS de entrada do wrapper é 32 bits;
- AXIS de saída do wrapper é 512 bits e possui TLAST;
- saída original do IP é 320 bits e não possui TLAST;
- DMA foi configurado com as mesmas larguras.

### Falha de timing ou route

Não concluir imediatamente que o modelo está inválido. Coletar:

- caminhos críticos;
- utilização hierárquica;
- congestionamento;
- fanout alto;
- SLR/clock region;
- WNS/TNS;
- CARRY8 e LUT por região.

Só então avaliar register slices, pipeline no wrapper, outro placement, RF maior
ou precisão mista. Alterações no núcleo exigem nova validação estatística.

## 18. Validação funcional na placa

### Etapa A — smoke test

- uma imagem conhecida;
- comparar dez logits FPGA versus hls4ml C++;
- comparar classe;
- inspecionar padding;
- confirmar DMA sem erros.

### Etapa B — subconjunto

- 100 imagens;
- registrar classe verdadeira, Keras, hls4ml C++ e FPGA;
- investigar qualquer divergência antes do teste completo.

### Etapa C — teste oficial

- mesmas 10.000 imagens MNIST;
- mesma ordem e preprocessing;
- batch 1;
- acurácia FPGA;
- matriz de confusão;
- concordância pareada com Keras e hls4ml C++;
- intervalo de Wilson de 95%;
- contagem e índices das divergências.

Referência esperada do modelo C++: 98,99%. Não declarar antecipadamente que a
FPGA obterá exatamente esse valor; medir.

## 19. Benchmark posterior

Separar claramente:

1. latência de inferência/DMA: buffers preparados, mede
   `recv transfer → send transfer → waits`;
2. latência ponta a ponta: inclui preprocessing, cópia, DMA, decode e argmax;
3. throughput sustentado: janela longa, após warm-up;
4. potência idle e carga;
5. energia dinâmica por inferência.

Usar:

- warm-up explícito;
- no mínimo 100 repetições por tamanho de conjunto;
- média, mediana, desvio padrão, mínimo, máximo, p5, p95 e p99;
- intervalo de confiança de 95%;
- mesmas imagens e batch 1 das campanhas CPU/GPU;
- clock e bitstream imutáveis durante a campanha.

```text
P_dynamic = P_load - P_idle
E_dynamic_per_inference = P_dynamic / throughput
```

Registrar também temperatura, frequência, rail medido, duração da janela e
quantidade real de inferências.

## 20. O que enviar ao ChatGPT Web em cada etapa

Começar a conversa anexando este README e dizer:

```text
Quero construir este Block Design passo a passo no Vivado 2024.2.
Não presuma que síntese, route ou bitstream já existem.
Use exatamente o IP e os contratos descritos.
Em cada etapa, forneça poucos comandos Tcl, diga o resultado esperado e espere
eu devolver a saída antes de continuar.
Não altere precisão, ReuseFactor, clock, modelo, packing ou arquitetura sem
explicar a necessidade e registrar uma nova configuração experimental.
```

Depois enviar:

### Auditoria do projeto

```tcl
version -short
get_property PART [current_project]
get_property BOARD_PART [current_project]
get_property IP_REPO_PATHS [current_project]
get_ipdefs -all -quiet *:hls:lenet_mnist_cap64_hls:*
```

### Auditoria do Block Design

```tcl
validate_bd_design
get_bd_cells
get_bd_intf_nets
report_property [get_bd_cells axi_dma_0]
```

### Depois da síntese

Anexar:

```text
utilization_post_synth.rpt
utilization_hierarchical_post_synth.rpt
timing_post_synth.rpt
clocks_post_synth.rpt
runme.log
```

### Depois do route

Anexar:

```text
utilization_post_route.rpt
utilization_hierarchical_post_route.rpt
timing_post_route.rpt
route_status_post_route.rpt
drc_post_route.rpt
check_timing_post_route.rpt
congestion_post_route.rpt
runme.log
```

Pedir que qualquer recomendação seja baseada nas linhas dos relatórios, não em
estimativas genéricas.

## 21. Checklist final

### IP e wrapper

- [ ] VLNV é `xilinx.com:hls:lenet_mnist_cap64_hls:1.0`;
- [ ] entrada 32 bits;
- [ ] saída 320 bits;
- [ ] wrapper gera saída 512 bits;
- [ ] wrapper gera TLAST;
- [ ] `ap_start=1`;
- [ ] reset ativo baixo.

### PS e DMA

- [ ] HPM0 habilitado;
- [ ] HPC0 habilitado;
- [ ] PL clock 100 MHz;
- [ ] SG desligado;
- [ ] Micro DMA desligado;
- [ ] MM2S stream 32 bits;
- [ ] S2MM stream 512 bits;
- [ ] recv armado antes de send;
- [ ] entrada 3.136 bytes;
- [ ] saída 64 bytes.

### Implementação

- [ ] Validate Design sem erros;
- [ ] síntese sem erros;
- [ ] recursos integrados registrados;
- [ ] WNS/TNS e WHS/THS aprovados;
- [ ] route completo;
- [ ] DRC sem erros;
- [ ] bitstream gerado;
- [ ] HWH e XSA preservados;
- [ ] hashes calculados.

### Validação

- [ ] preprocessing `uint8/255`;
- [ ] Q22.12, escala 1024;
- [ ] sinal no bit 21;
- [ ] dez logits, sem Softmax;
- [ ] smoke test;
- [ ] 100 imagens;
- [ ] 10.000 imagens;
- [ ] benchmark estatístico;
- [ ] telemetria física.

## 22. Fontes locais de verdade

```text
hls4ml/README.md
hls4ml/configs/design_cap64_q22_12_zcu104.json
hls4ml/configs/reuse_cap64_rate_balanced.json
hls4ml/builds/cap64_q22_12_rate_balanced/cpp_validation.json
hls4ml/builds/cap64_q22_12_rate_balanced/firmware/defines.h
hls4ml/ip_repo/lenet_mnist_cap64_hls_v1_0/component.xml
hls4ml/ip_repo/lenet_mnist_cap64_hls_v1_0/hdl/verilog/lenet_mnist_cap64_hls.v
hls4ml/reports/cap64_q22_12_rate_balanced/HLS_VS_VIVADO_UTILIZATION.md
hls4ml/reports/cap64_q22_12_rate_balanced/vivado_synth.rpt
```

Se este documento divergir de `component.xml` ou do RTL exportado, o
`component.xml` e o RTL são a fonte definitiva para portas e larguras.
