# Handoff completo — integração da ResNet8 hls4ml no Vivado/ZCU104

> Documento de contexto para continuar este trabalho com o ChatGPT Web.
>
> Objetivo imediato: construir, validar e implementar no Vivado 2024.2 um Block
> Design para executar inferência CIFAR-10 na ZCU104 usando o novo IP hls4ml
> `xilinx.com:hls:resnet8_resource_fifo_opt:1.1`.

## 1. Como o ChatGPT Web deve usar este documento

Este arquivo é o contexto canônico da próxima etapa. O assistente deve:

1. ler o documento inteiro antes de sugerir conexões;
2. trabalhar em etapas pequenas e verificáveis;
3. fornecer comandos para o **Vivado Tcl Console** sempre que possível;
4. informar claramente quando um comando deve ser executado no shell Linux e
   quando deve ser executado dentro do Vivado;
5. pedir a saída de `validate_bd_design`, dos runs ou dos relatórios antes de
   afirmar que uma etapa passou;
6. nunca substituir o IP atual 1.1 pelo IP antigo 2.2;
7. não tratar os resultados pós-route do projeto antigo como resultados do novo
   projeto;
8. preservar a arquitetura comprovada primeiro e só experimentar otimizações
   depois que a primeira inferência funcionar;
9. não regenerar o IP pelo Vitis HLS sem uma razão explícita: o IP 1.1 corrigido
   já foi gerado, auditado, empacotado e sintetizado OOC;
10. não usar o ZIP/IP 1.0 produzido pelo export padrão do HLS.

### Prompt inicial sugerido para a nova conversa

Copie este arquivo para a conversa e envie algo equivalente a:

```text
Leia integralmente o README anexado. Ele é o contexto canônico do projeto.

Quero integrar no Vivado 2024.2 o IP
xilinx.com:hls:resnet8_resource_fifo_opt:1.1 na ZCU104, usando como referência a
arquitetura comprovada do projeto Resnet8-Acc-FPGA, mas sem confundir o IP antigo
2.2 com o novo 1.1.

Trabalhe uma fase por vez. Para cada fase:
1. explique o objetivo;
2. forneça comandos Tcl prontos para colar;
3. diga qual saída devo enviar de volta;
4. não avance se houver erro;
5. não declare sucesso sem evidência da ferramenta.

Comece pela Fase 0: auditoria dos caminhos, checksum do ZIP, descoberta do VLNV
no catálogo e criação de um projeto Vivado limpo para a ZCU104.
```

## 2. Objetivo final do projeto

O sistema deve:

1. carregar uma imagem CIFAR-10 `32 x 32 x 3` na DDR da ZCU104;
2. normalizar e converter a imagem para o formato fixo do acelerador;
3. enviar a imagem da DDR ao PL por AXI DMA MM2S;
4. adaptar o stream de 128 bits do DMA ao stream de 96 bits do IP;
5. executar a ResNet8 em hardware;
6. adaptar os 320 bits de logits para um pacote AXI-Stream de 512 bits;
7. produzir `TLAST` corretamente;
8. gravar os logits na DDR por AXI DMA S2MM;
9. ler e decodificar os dez logits no PYNQ;
10. executar `argmax` no ARM;
11. validar acurácia, latência, vazão, timing, recursos, potência e energia por
    inferência.

O resultado esperado da etapa de hardware é um conjunto coerente:

```text
system_wrapper.bit
system_wrapper.hwh
system_bd.tcl
relatórios pós-route
script PYNQ de smoke test/benchmark
```

O `.bit` e o `.hwh` devem ter exatamente o mesmo basename para uso por
`pynq.Overlay`.

## 3. Fontes de verdade e precedência

Em caso de divergência, usar esta ordem:

1. interfaces físicas presentes no `component.xml` e no top Verilog do IP 1.1;
2. manifests e checksums do IP 1.1 atual;
3. relatórios OOC e HLS do IP 1.1 atual;
4. este documento;
5. guia local de replicação do projeto antigo;
6. projeto antigo no GitHub.

Fontes principais:

- Guia local antigo: `RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt`.
- Projeto comprovado anterior:
  <https://github.com/MiguelBertonzin/Resnet8-Acc-FPGA>.
- Commit auditado do projeto anterior:
  `557d384114ce7257464079307e628579482025b8`.
- README atual do fluxo hls4ml: `hls4ml/README.md`.
- Relatório consolidado atual:
  `hls4ml/reports/VITIS_HLS_VS_VIVADO_IP_V1_1.md`.
- Manifesto do IP atual:
  `hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1/ip_manifest.json`.
- Manifesto de validação:
  `hls4ml/ip_validation/resnet8_resource_fifo_opt_v1_1/validation_manifest.json`.

## 4. Ambiente e caminhos atuais

Workspace atual:

```text
/home/miguel/Downloads/Plano testes TCC/resnet8
```

Ferramentas e alvo:

| Item | Valor |
|---|---|
| Placa | AMD/Xilinx ZCU104 |
| Board part | `xilinx.com:zcu104:part0:1.1` |
| FPGA | `xczu7ev-ffvc1156-2-e` |
| Vivado | 2024.2 |
| Vitis HLS | 2024.2 |
| hls4ml | 1.3.0 |
| Clock planejado | 100 MHz |
| Período | 10 ns |
| Ambiente de placa de referência | PYNQ 3.1.1, Python 3.10.4 |

IP repository válido, pronto para ser adicionado diretamente ao Vivado:

```text
/home/miguel/Downloads/Plano testes TCC/resnet8/hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1
```

ZIP transportável válido:

```text
/home/miguel/Downloads/Plano testes TCC/resnet8/hls4ml/ip_exports/xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip
```

SHA-256 esperado do ZIP:

```text
4e148c361116d19430aabd95e00b0dc3ca6789821e83602f1cd5d70313095191
```

VLNV obrigatório:

```text
xilinx.com:hls:resnet8_resource_fifo_opt:1.1
```

Não usar:

- qualquer pacote 1.0 do fluxo atual;
- `xilinx.com:hls:resnet8_resource_fifo_opt:2.2` do projeto antigo;
- um repositório arquivado ou outro IP que apareça no catálogo apenas porque tem
  número de versão aparentemente maior.

A pasta `hls4ml/artifacts/ip/` está vazia. O IP promovido está em `ip_repo/` e
`ip_exports/`.

## 5. O que mudou em relação ao projeto funcional anterior

| Item | Projeto antigo comprovado | Projeto atual a implementar |
|---|---|---|
| VLNV | `...:2.2` | `...:1.1` |
| Modelo de entrada HLS | original com softmax ignorada | arquivo canônico sem softmax |
| Interface externa | 96 bits entrada / 320 saída | igual |
| Controle | `ap_ctrl_hs`, `ap_start=1` | igual |
| DMA | MM2S 128, S2MM 512 | replicar igual inicialmente |
| Wrapper | 128→96 e 320→512 + TLAST | reutilizável sem mudar contrato |
| DSP OOC | 1.174 | 1.174 |
| LUT OOC | aproximadamente 165.242 | 165.564 |
| BRAM OOC | 93,5 tiles | 74,5 tiles |
| URAM OOC | 0 | 11 |
| FIFOs grandes | 1.026, 703, 706 etc. | 2.100, 1.351, 1.464 etc. |
| Pós-route completo | aprovado e medido | ainda não executado |
| Teste físico 10k | 74,92% no sistema antigo | ainda precisa ser repetido |

Conclusão: o contrato externo é compatível com o wrapper anterior, mas o interior
e a distribuição de memória são diferentes. O sucesso antigo torna a arquitetura
uma excelente referência, não uma prova de timing/roteamento do IP atual.

## 6. Estado comprovado do IP 1.1 atual

### 6.1 Modelo

Modelo canônico do acelerador:

```text
resnet8_cifar10_keras3_no_softmax.h5
SHA-256: 047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e
```

Características:

- entrada `(32, 32, 3)`;
- saída com 10 logits;
- 78.714 parâmetros;
- última camada `Dense` linear;
- pesos equivalentes ao modelo original;
- softmax deve ser feita no PS apenas se probabilidades forem necessárias;
- para classificação top-1, usar `argmax(logits)`.

### 6.2 Configuração hls4ml

```text
Backend              = Vitis
IOType                = io_stream
Strategy              = Resource
ConvImplementation    = LineBuffer
Precision             = ap_fixed<22,12,AP_RND_CONV,AP_SAT>
Clock                 = 10 ns
Part                  = xczu7ev-ffvc1156-2-e
ReuseFactor máximo    = 288
```

Reuse efetivo:

| Camada | RF efetivo |
|---|---:|
| `conv2d` | 9 |
| `conv2d_1` | 36 |
| `conv2d_2` | 36 |
| `conv2d_3` | 72 |
| `conv2d_4` | 144 |
| `conv2d_5` | 64 |
| `conv2d_6` | 288 |
| `conv2d_7` | 288 |
| `conv2d_8` | 128 |
| `dense` | 64 |

### 6.3 FIFO optimization

- 42 FIFOs perfilados;
- profundidade temporária de profiling: 4.096;
- maior profundidade otimizada: 2.100;
- nenhum canal saturou o profiling;
- top final não instancia módulos FIFO `d4096`;
- módulos residuais `d4096` foram removidos do pacote promovido.

FIFOs relevantes no top atual:

```text
fifo_w66_d2100
fifo_w352_d1351
fifo_w352_d1464
fifo_w352_d69
fifo_w352_d30
fifo_w704_d45
fifo_w704_d42
fifo_w704_d14
fifo_w1408_d23
fifo_w1408_d21
demais FIFOs = profundidade 1
```

### 6.4 Validação realizada

- equivalência do modelo original/sem softmax: aprovada;
- validação C++ hls4ml em 20 amostras: 100% de concordância top-1;
- C simulation: aprovada;
- RTL co-simulation do estágio de profiling: aprovada;
- síntese HLS final: aprovada;
- IP catalog/`validate_ip`: aprovado;
- `generate_target all`: aprovado;
- síntese OOC do IP empacotado: aprovada;
- ZIP: íntegro e checksum confirmado.

Ressalva de validação: o fluxo final executou `cosim=False` depois de reescrever
as profundidades. Portanto, o RTL FIFO final passou síntese OOC, mas não houve uma
segunda co-simulação completa específica do RTL final. Isso não impede iniciar o
Block Design, mas deve constar no relatório e ser compensado por simulação do
sistema e teste físico.

### 6.5 Estimativas e recursos

HLS:

```text
Clock estimado = 7,298 ns
Latência       = 85.026 a 85.313 ciclos
Tempo a 100MHz = 0,850 a 0,853 ms
```

Vivado OOC do IP 1.1:

| Recurso | Uso | Disponível | Percentual |
|---|---:|---:|---:|
| CLB LUT | 165.564 | 230.400 | 71,86% |
| Registers | 178.894 | 460.800 | 38,82% |
| BRAM tiles | 74,5 | 312 | 23,88% |
| URAM | 11 | 96 | 11,46% |
| DSP48E2 | 1.174 | 1.728 | 67,94% |

Há margem nominal para DMA, SmartConnect e wrapper, mas 71,86% de LUT no IP
isolado exige atenção a congestionamento e timing. Somente place-and-route do
sistema completo dará o resultado definitivo.

## 7. Contrato externo exato do IP 1.1

### 7.1 Clock, reset e controle

```text
ap_clk       input,  100 MHz planejado
ap_rst_n     input,  ativo em nível baixo
ap_start     input
ap_done      output
ap_ready     output
ap_idle      output
```

O protocolo é `ap_ctrl_hs`; não existe AXI4-Lite de controle no acelerador.

Na réplica inicial:

```text
ap_start = 1 permanente
```

Isso reproduz o sistema funcional antigo. O software controla apenas o DMA.

### 7.2 Stream de entrada

```text
input_layer_TDATA   [95:0]
input_layer_TVALID
input_layer_TREADY
```

AXI-Stream slave com:

- `TDATA_NUM_BYTES = 12`;
- sem `TLAST`;
- sem `TKEEP`;
- sem `TSTRB`;
- sem `TUSER`, `TID` ou `TDEST`.

Uma imagem contém 1.024 transferências aceitas (`TVALID && TREADY`), uma por
pixel, em ordem row-major/HWC.

### 7.3 Stream de saída

```text
layer31_out_TDATA   [319:0]
layer31_out_TVALID
layer31_out_TREADY
```

AXI-Stream master com:

- `TDATA_NUM_BYTES = 40`;
- um beat por inferência;
- sem `TLAST`;
- sem `TKEEP`;
- sem `TSTRB`.

O wrapper deve gerar o `TLAST` exigido pelo AXI DMA S2MM.

## 8. Formato numérico e packing

### 8.1 Formato fixo

```text
ap_fixed<22,12,AP_RND_CONV,AP_SAT>
W = 22 bits
I = 12 bits, incluindo sinal
F = 10 bits fracionários
escala = 2^10 = 1024
```

Para converter float em bits:

```text
raw = round_convergent(valor * 1024)
raw = saturar em [-2^21, 2^21-1]
bits22 = raw & 0x003FFFFF
```

Para imagens em `[0,1]`, não há sinal negativo, mas o código deve ser genérico.

### 8.2 Entrada 128 bits do DMA para 96 bits do IP

O DMA MM2S será configurado com stream de 128 bits. Cada beat contém:

```text
127          96 95           64 63           32 31            0
+--------------+---------------+---------------+---------------+
| padding zero | canal B       | canal G       | canal R       |
+--------------+---------------+---------------+---------------+
    32 bits         32 bits         32 bits         32 bits
```

Dentro de cada lane de 32 bits:

```text
bits [21:0]  = representação ap_fixed
bits [31:22] = padding; colocar zero no software
```

O wrapper encaminha somente `s_axis_tdata(95 downto 0)`.

Buffer PYNQ:

```python
input_buffer = allocate(shape=(1024, 4), dtype=np.uint32)
input_buffer[:, 0] = R_fixed
input_buffer[:, 1] = G_fixed
input_buffer[:, 2] = B_fixed
input_buffer[:, 3] = 0
```

Tamanho enviado pelo DMA:

```text
1024 pixels * 16 bytes = 16.384 bytes
```

### 8.3 Saída 320 bits do IP para 512 bits do DMA

O IP produz dez lanes físicas de 32 bits em um único beat:

```text
lane 0: bits [21:0] do logit 0 em TDATA[21:0]
lane 1: bits [21:0] do logit 1 em TDATA[53:32]
...
lane 9: bits [21:0] do logit 9 em TDATA[309:288]
```

Os dez bits superiores de cada lane são padding. Não interpretar a lane inteira
diretamente como `int32`, pois valores negativos possuem sinal no bit 21, não no
bit 31.

O wrapper deve produzir:

```text
m_axis_tdata[319:0]   = saída do IP
m_axis_tdata[511:320] = zero
m_axis_tlast          = 1 durante o beat válido
```

Buffer PYNQ:

```python
output_buffer = allocate(shape=(16,), dtype=np.uint32)
logits = fixed_bits_to_float(output_buffer[:10])
padding = output_buffer[10:16]  # deve ser zero
```

## 9. Arquitetura alvo do Block Design

Replicar inicialmente a topologia já comprovada:

```text
                              PROCESSING SYSTEM
                    +--------------------------------+
                    | ARM / Linux / PYNQ             |
                    |                                |
                    | M_AXI_HPM0_FPD ----+           |
                    | M_AXI_HPM1_FPD ----+-- controle|
                    |                                |
                    | DDR <----- S_AXI_HPC0_FPD      |
                    +----------------------^---------+
                                           |
                              tráfego memory-mapped
                                           |
                    +----------------------|---------+
                    | PROGRAMMABLE LOGIC             |
                    |                                |
   HPM0/HPM1 ------>| axi_smc (controle)             |
                    |          |                     |
                    |          v S_AXI_LITE          |
                    |      +---------+               |
                    |      | AXI DMA |               |
                    |      +---------+               |
                    |       |       ^                |
                    | MM2S  |128 512| S2MM           |
                    |       v       |                |
                    | +---------------------------+  |
                    | | Accel_dma_wrapper         |  |
                    | | 128→96; 320→512; TLAST    |  |
                    | +-------------+-------------+  |
                    |               |96  ^320        |
                    |               v    |           |
                    | +---------------------------+  |
                    | | ResNet8 hls4ml IP 1.1     |  |
                    | +---------------------------+  |
                    |                                |
                    | DMA MM2S/S2MM -> axi_smc_1 ----+--> HPC0
                    +--------------------------------+
```

Blocos:

```text
zynq_ultra_ps_e_0
rst_ps8_0_100M
axi_smc          # controle: 2 SI, 1 MI
axi_smc_1        # memória: 2 SI, 1 MI
axi_dma_0
Accel_dma_wrapper_0
myproject_0      # IP ResNet8 1.1
```

Interrupções do DMA ficam desconectadas na primeira réplica; o PYNQ usa polling.

## 10. Configuração obrigatória dos blocos

### 10.1 Zynq UltraScale+ MPSoC

Aplicar o board preset da ZCU104 e conferir:

```text
M_AXI_HPM0_FPD = enabled
M_AXI_HPM1_FPD = enabled
S_AXI_HPC0_FPD = enabled
pl_clk0        = 100 MHz
pl_resetn0     = enabled
```

Os dois HPM reproduzem o Block Design funcional anterior. Essa redundância pode
ser simplificada no futuro, mas não antes do primeiro bitstream funcional.

### 10.2 Processor System Reset

```text
pl_clk0    -> slowest_sync_clk
pl_resetn0 -> ext_reset_in
```

Usar `peripheral_aresetn` para DMA, SmartConnects, wrapper e IP. Não adicionar
inversor na réplica inicial; o Tcl comprovado conectava diretamente esses sinais.

### 10.3 SmartConnect de controle

```text
NUM_SI = 2
NUM_MI = 1

HPM0 -> S00_AXI
HPM1 -> S01_AXI
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

### 10.5 AXI DMA

Configuração replicada do sistema comprovado:

```text
Scatter Gather                 OFF
Micro DMA                      OFF
Address Width                  32
Buffer Length Register Width   26

MM2S memory width              128
MM2S stream width              128
MM2S burst                     64
MM2S DRE                       ON

S2MM memory width              512
S2MM stream width              512
S2MM burst                     64
S2MM DRE                       ON
```

O DMA suporta essas larguras. O S2MM recebe um pacote de um beat de 512 bits com
`TLAST=1`.

## 11. Wrapper VHDL recomendado

Criar no novo projeto, por exemplo:

```text
hardware/src/Accel_dma_wrapper.vhd
```

Conteúdo comprovado e compatível com as portas do IP 1.1:

```vhdl
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity Accel_dma_wrapper is
    Port (
        ap_clk   : in  std_logic;
        ap_rst_n : in  std_logic;

        s_axis_tdata  : in  std_logic_vector(127 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        m_axis_tdata  : out std_logic_vector(511 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic;
        m_axis_tlast  : out std_logic;

        ap_start : out std_logic;
        ap_done  : in  std_logic;
        ap_idle  : in  std_logic;
        ap_ready : in  std_logic;

        accel_input_data  : out std_logic_vector(95 downto 0);
        accel_input_valid : out std_logic;
        accel_input_ready : in  std_logic;

        accel_output_data  : in  std_logic_vector(319 downto 0);
        accel_output_valid : in  std_logic;
        accel_output_ready : out std_logic
    );
end Accel_dma_wrapper;

architecture rtl of Accel_dma_wrapper is
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
begin
    ap_start <= '1';

    accel_input_data  <= s_axis_tdata(95 downto 0);
    accel_input_valid <= s_axis_tvalid;
    s_axis_tready     <= accel_input_ready;

    m_axis_tdata(319 downto 0)   <= accel_output_data;
    m_axis_tdata(511 downto 320) <= (others => '0');

    m_axis_tvalid      <= accel_output_valid;
    accel_output_ready <= m_axis_tready;

    -- Existe um único beat de saída por inferência.
    -- Manter TLAST alto enquanto TVALID estiver alto é correto mesmo sob
    -- backpressure; a transferência só ocorre quando TVALID && TREADY.
    m_axis_tlast <= accel_output_valid;
end rtl;
```

Não colocar registradores adicionais no wrapper durante a primeira réplica. Se
timing exigir pipeline posteriormente, o handshake deve ser preservado com um
AXI Register Slice ou skid buffer correto; não se pode atrasar somente `TDATA` ou
somente `TVALID`.

## 12. Sequência recomendada de construção

### Fase 0 — auditoria antes de abrir o Vivado

No shell Linux:

```bash
cd "/home/miguel/Downloads/Plano testes TCC/resnet8"

sha256sum \
  hls4ml/ip_exports/xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip

test -f hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1/component.xml
test -f hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1/hdl/verilog/resnet8_resource_fifo_opt.v

rg -n "<spirit:(vendor|library|name|version)>" \
  hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1/component.xml
```

Esperado:

```text
xilinx.com:hls:resnet8_resource_fifo_opt:1.1
```

Recomenda-se criar o projeto Vivado em um caminho sem espaços, por exemplo:

```text
/home/miguel/resnet8_vivado_ip11
```

O IP repository pode permanecer no workspace original; Tcl deve usar
`file normalize` e chaves para proteger espaços.

Antes de avançar para a Fase 2, criar o diretório `hardware/src` no workspace
e salvar nele, com o nome `Accel_dma_wrapper.vhd`, exatamente o código da
seção 11. O Tcl da próxima fase verifica esse arquivo e interrompe a construção
imediatamente se ele estiver ausente.

### Fase 1 — abrir o Vivado 2024.2

Shell:

```bash
/opt/Xilinx/Vivado/2024.2/bin/vivado
```

Não executar comandos Tcl do Vivado no terminal bash.

### Fase 2 — criar projeto e adicionar IP/wrapper

Os blocos abaixo são comandos para o **Vivado Tcl Console**.

```tcl
set workspace_root [file normalize {/home/miguel/Downloads/Plano testes TCC/resnet8}]
set project_root   [file normalize {/home/miguel/resnet8_vivado_ip11}]
set ip_repo        [file normalize [file join $workspace_root hls4ml ip_repo resnet8_resource_fifo_opt_v1_1]]
set wrapper_src    [file normalize [file join $workspace_root hardware src Accel_dma_wrapper.vhd]]
set expected_vlnv  {xilinx.com:hls:resnet8_resource_fifo_opt:1.1}

foreach required [list \
    [file join $ip_repo component.xml] \
    [file join $ip_repo hdl verilog resnet8_resource_fifo_opt.v] \
    $wrapper_src] {
    if {![file exists $required]} {
        error "Arquivo obrigatório ausente: $required"
    }
}

if {[llength [get_projects -quiet]] == 0} {
    create_project resnet8_zcu104 $project_root -part xczu7ev-ffvc1156-2-e
    set_property BOARD_PART xilinx.com:zcu104:part0:1.1 [current_project]
}

# Usar somente o repo atual durante a réplica evita selecionar o IP antigo 2.2.
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog -rebuild

set defs [get_ipdefs -all -quiet $expected_vlnv]
if {[llength $defs] != 1} {
    error "Esperava exatamente um IP $expected_vlnv; encontrado: $defs"
}
puts "IP_ATUAL_OK=$defs"

add_files -norecurse $wrapper_src
update_compile_order -fileset sources_1

if {![can_resolve_reference Accel_dma_wrapper]} {
    error "Accel_dma_wrapper não compila/não pode ser resolvido"
}
puts "WRAPPER_OK"
```

Enviar ao ChatGPT a saída de:

```tcl
version -short
get_property PART [current_project]
get_property BOARD_PART [current_project]
get_property IP_REPO_PATHS [current_project]
get_ipdefs -all -quiet *:hls:resnet8_resource_fifo_opt:*
```

### Fase 3 — criar o Block Design e os blocos

Vivado Tcl Console:

```tcl
set design_name system
if {[get_files -quiet */${design_name}.bd] ne ""} {
    error "O BD $design_name já existe. Não sobrescrever sem decisão explícita."
}

create_bd_design $design_name
current_bd_design $design_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps

# Reafirmar as interfaces usadas pela arquitetura comprovada.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__MAXIGP1__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps8_0_100M]

set smc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $smc_ctrl

set smc_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_1]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $smc_mem

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_m_axi_s2mm_data_width {512} \
    CONFIG.c_s_axis_s2mm_tdata_width {512} \
    CONFIG.c_s2mm_burst_size {64} \
    CONFIG.c_sg_length_width {26} \
] $dma

set adapt [create_bd_cell -type module -reference Accel_dma_wrapper Accel_dma_wrapper_0]
set accel [create_bd_cell -type ip -vlnv $expected_vlnv myproject_0]

puts "ACCEL_VLNV=[get_property VLNV [get_bd_cells myproject_0]]"
```

Esperado:

```text
ACCEL_VLNV=xilinx.com:hls:resnet8_resource_fifo_opt:1.1
```

Se as propriedades PS não existirem após o board preset, não inventar nomes.
Executar:

```tcl
report_property [get_bd_cells zynq_ultra_ps_e_0]
```

e enviar a saída relevante ao ChatGPT.

### Fase 4 — conectar interfaces AXI

Vivado Tcl Console:

```tcl
connect_bd_intf_net \
    [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_smc/S00_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD] \
    [get_bd_intf_pins axi_smc/S01_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins axi_smc/M00_AXI] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

connect_bd_intf_net \
    [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_smc_1/S00_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins axi_smc_1/S01_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins axi_smc_1/M00_AXI] \
    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

connect_bd_intf_net \
    [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins Accel_dma_wrapper_0/s_axis]

connect_bd_intf_net \
    [get_bd_intf_pins Accel_dma_wrapper_0/m_axis] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
```

Não usar Connection Automation depois disso sem revisar o que ela pretende criar.

### Fase 5 — conectar wrapper ao IP HLS

Vivado Tcl Console:

```tcl
connect_bd_net \
    [get_bd_pins Accel_dma_wrapper_0/accel_input_data] \
    [get_bd_pins myproject_0/input_layer_TDATA]

connect_bd_net \
    [get_bd_pins Accel_dma_wrapper_0/accel_input_valid] \
    [get_bd_pins myproject_0/input_layer_TVALID]

connect_bd_net \
    [get_bd_pins myproject_0/input_layer_TREADY] \
    [get_bd_pins Accel_dma_wrapper_0/accel_input_ready]

connect_bd_net \
    [get_bd_pins myproject_0/layer31_out_TDATA] \
    [get_bd_pins Accel_dma_wrapper_0/accel_output_data]

connect_bd_net \
    [get_bd_pins myproject_0/layer31_out_TVALID] \
    [get_bd_pins Accel_dma_wrapper_0/accel_output_valid]

connect_bd_net \
    [get_bd_pins Accel_dma_wrapper_0/accel_output_ready] \
    [get_bd_pins myproject_0/layer31_out_TREADY]

connect_bd_net \
    [get_bd_pins Accel_dma_wrapper_0/ap_start] \
    [get_bd_pins myproject_0/ap_start]

connect_bd_net \
    [get_bd_pins myproject_0/ap_done] \
    [get_bd_pins Accel_dma_wrapper_0/ap_done]

connect_bd_net \
    [get_bd_pins myproject_0/ap_idle] \
    [get_bd_pins Accel_dma_wrapper_0/ap_idle]

connect_bd_net \
    [get_bd_pins myproject_0/ap_ready] \
    [get_bd_pins Accel_dma_wrapper_0/ap_ready]
```

Conferir larguras:

```tcl
foreach pin {
    myproject_0/input_layer_TDATA
    myproject_0/layer31_out_TDATA
    Accel_dma_wrapper_0/accel_input_data
    Accel_dma_wrapper_0/accel_output_data
} {
    puts "$pin LEFT=[get_property LEFT [get_bd_pins $pin]] RIGHT=[get_property RIGHT [get_bd_pins $pin]]"
}
```

Esperado: `95:0` na entrada e `319:0` na saída.

### Fase 6 — clock e reset

Vivado Tcl Console:

```tcl
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins rst_ps8_0_100M/slowest_sync_clk] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm1_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk] \
    [get_bd_pins axi_smc/aclk] \
    [get_bd_pins axi_smc_1/aclk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins Accel_dma_wrapper_0/ap_clk] \
    [get_bd_pins myproject_0/ap_clk]

connect_bd_net \
    [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
    [get_bd_pins rst_ps8_0_100M/ext_reset_in]

connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] \
    [get_bd_pins axi_smc/aresetn] \
    [get_bd_pins axi_smc_1/aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins Accel_dma_wrapper_0/ap_rst_n] \
    [get_bd_pins myproject_0/ap_rst_n]
```

Conferir frequência e nets:

```tcl
get_bd_pins -of_objects [get_bd_nets -of_objects [get_bd_pins myproject_0/ap_clk]]
get_bd_pins -of_objects [get_bd_nets -of_objects [get_bd_pins myproject_0/ap_rst_n]]
```

### Fase 7 — Address Editor

O projeto funcional anterior usou:

```text
DMA S_AXI_LITE = 0xA0000000 / 64 KiB
DMA MM2S DDR   = 0x00000000 / 2 GiB
DMA S2MM DDR   = 0x00000000 / 2 GiB
```

Primeiro descobrir os nomes presentes:

```tcl
get_bd_addr_spaces -hier
get_bd_addr_segs -hier
```

Se forem iguais ao projeto de referência:

```tcl
assign_bd_address -offset 0xA0000000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force

assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW] -force

assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW] -force
```

OCM e QSPI não são necessários. Se aparecerem no address space do DMA, podem ser
excluídos como no projeto antigo:

```tcl
exclude_bd_addr_seg -offset 0xFF000000 -range 0x01000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_LPS_OCM]

exclude_bd_addr_seg -offset 0xC0000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_QSPI]

exclude_bd_addr_seg -offset 0xFF000000 -range 0x01000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_LPS_OCM]

exclude_bd_addr_seg -offset 0xC0000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_QSPI]
```

Não executar os `exclude` se os segmentos não existirem. O ChatGPT deve montar
uma versão defensiva usando `get_bd_addr_segs -quiet` após ver a saída real.

### Fase 8 — validar o BD

```tcl
regenerate_bd_layout
save_bd_design
validate_bd_design
```

Também registrar:

```tcl
puts "IP_NO_BD=[get_property VLNV [get_bd_cells myproject_0]]"
report_ip_status
```

Não avançar se houver erro de:

- interface/largura;
- clock sem associação;
- reset sem associação;
- endereço não atribuído;
- IP locked/unresolved;
- Module Reference ausente.

Warnings conhecidos de OCM/QSPI ou USER width devem ser avaliados, não ignorados
automaticamente.

### Fase 9 — gerar targets e HDL wrapper

```tcl
set bd_file [get_files -quiet */system.bd]
if {[llength $bd_file] != 1} {
    error "system.bd não encontrado de forma única: $bd_file"
}

generate_target all $bd_file
make_wrapper -files $bd_file -top

set generated_wrapper [file normalize \
    [file join [get_property DIRECTORY [current_project]] \
     resnet8_zcu104.gen sources_1 bd system hdl system_wrapper.v]]

if {![file exists $generated_wrapper]} {
    # O nome do diretório .gen depende do nome do projeto.
    puts "Procure o wrapper com: get_files -all *system_wrapper.v"
} else {
    add_files -norecurse $generated_wrapper
}

set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1
save_bd_design
```

Forma mais robusta após `make_wrapper`:

```tcl
set wrappers [get_files -all -quiet *system_wrapper.v]
puts "WRAPPERS=$wrappers"
```

Se `make_wrapper` criar o arquivo mas ele não estiver no fileset, adicionar o
caminho retornado pelo console.

### Fase 10 — síntese

```tcl
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
puts "SYNTH_STATUS=[get_property STATUS [get_runs synth_1]]"

open_run synth_1
file mkdir [file join $project_root reports]
report_utilization -file [file join $project_root reports utilization_synth.rpt]
report_timing_summary -file [file join $project_root reports timing_synth.rpt]
```

Após síntese, conferir:

```tcl
get_property STATUS [get_runs synth_1]
get_property PROGRESS [get_runs synth_1]
report_drc
```

Não rejeitar o projeto apenas porque a estimativa HLS dizia 160% de LUT. A
referência correta é a síntese Vivado do sistema atual.

### Fase 11 — implementação e bitstream

Começar com a estratégia padrão:

```tcl
set_property strategy {Vivado Implementation Defaults} [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "IMPL_STATUS=[get_property STATUS [get_runs impl_1]]"
puts "IMPL_PROGRESS=[get_property PROGRESS [get_runs impl_1]]"
```

Abrir o run e gerar relatórios:

```tcl
open_run impl_1

set report_dir [file join $project_root reports]
file mkdir $report_dir

report_utilization -hierarchical \
    -file [file join $report_dir utilization_post_route.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -file [file join $report_dir timing_post_route.rpt]
report_route_status \
    -file [file join $report_dir route_status.rpt]
report_drc \
    -file [file join $report_dir drc_post_route.rpt]
report_power \
    -file [file join $report_dir power_post_route.rpt]

write_bd_tcl -force [file join $project_root system_bd_ip11.tcl]
write_hw_platform -fixed -include_bit -force \
    [file join $project_root system_wrapper_ip11.xsa]
```

Critérios mínimos:

```text
failed nets       = 0
unrouted nets     = 0
partially routed  = 0
WNS               >= 0
TNS               = 0
WHS               >= 0
THS               = 0
DRC errors        = 0
bitstream         gerado
```

O projeto antigo atingiu WNS `+0,257 ns` e 77,42% de LUT pós-route, mas esses
valores são apenas referência. O IP atual usa 11 URAM e pode apresentar outra
distribuição física.

Se timing/route falhar, preservar o primeiro resultado e só então explorar:

1. `phys_opt_design` padrão;
2. placement directives moderadas;
3. AXI Register Slice apenas nas fronteiras do wrapper/DMA;
4. floorplanning baseado no relatório de congestionamento;
5. alterações HLS apenas como última etapa.

Não iniciar diretamente com `Congestion_SpreadLogic_high`: ele foi pior no
projeto anterior.

## 13. Exportação para PYNQ

Localizar o bitstream no run:

```tcl
set impl_dir [get_property DIRECTORY [get_runs impl_1]]
puts "IMPL_DIR=$impl_dir"
puts "BITS=[glob -nocomplain -directory $impl_dir *.bit]"
puts "HWHS=[get_files -all -quiet *.hwh]"
```

Criar uma pasta de overlay e copiar o `.bit` e `.hwh` com o mesmo nome:

```text
hardware/overlay/system_wrapper.bit
hardware/overlay/system_wrapper.hwh
```

Preservar também:

```text
hardware/system_bd_ip11.tcl
hardware/src/Accel_dma_wrapper.vhd
hardware/reports/
hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1/
hls4ml/ip_exports/xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip
```

## 14. Smoke test PYNQ

Sequência obrigatória por inferência:

1. preencher os buffers;
2. armar `recvchannel` primeiro;
3. iniciar `sendchannel`;
4. aguardar MM2S;
5. aguardar S2MM;
6. verificar status do DMA;
7. decodificar logits;
8. verificar padding;
9. executar `argmax`.

Código mínimo:

```python
from pathlib import Path
import numpy as np
from pynq import Overlay, allocate

FIXED_W = 22
FIXED_I = 12
FIXED_F = FIXED_W - FIXED_I
FIXED_SCALE = 1 << FIXED_F
FIXED_MASK = (1 << FIXED_W) - 1
FIXED_SIGN = 1 << (FIXED_W - 1)
RAW_MIN = -(1 << (FIXED_W - 1))
RAW_MAX = (1 << (FIXED_W - 1)) - 1
N_PIXELS = 32 * 32

def normalize_image(img):
    x = np.asarray(img, dtype=np.float32)
    if x.shape != (32, 32, 3):
        raise ValueError(f"shape inválido: {x.shape}")
    if np.max(x) > 1.5:
        x = x / 255.0
    return x

def float_to_fixed_bits(x):
    # np.rint usa arredondamento para o par mais próximo em casos exatos,
    # compatível com a intenção de AP_RND_CONV para o packing de software.
    raw = np.rint(x * FIXED_SCALE).astype(np.int64)
    raw = np.clip(raw, RAW_MIN, RAW_MAX)
    return (raw & FIXED_MASK).astype(np.uint32)

def fixed_bits_to_float(words):
    words = np.asarray(words, dtype=np.uint32)
    raw = words & FIXED_MASK
    signed = raw.astype(np.int64)
    negative = (raw & FIXED_SIGN) != 0
    signed[negative] -= (1 << FIXED_W)
    return signed.astype(np.float32) / FIXED_SCALE

bit = Path("hardware/overlay/system_wrapper.bit").resolve()
hwh = bit.with_suffix(".hwh")
if not bit.exists() or not hwh.exists():
    raise FileNotFoundError(f"Overlay incompleto: {bit}, {hwh}")

ol = Overlay(str(bit))
print("IPs:", list(ol.ip_dict))
if "axi_dma_0" not in ol.ip_dict:
    raise RuntimeError("axi_dma_0 ausente no HWH")
dma = ol.axi_dma_0

x_test = np.load("data/cifar10_x_test.npy", mmap_mode="r")
y_test = np.load("data/cifar10_y_test.npy").reshape(-1)

input_buffer = allocate(shape=(N_PIXELS, 4), dtype=np.uint32)
output_buffer = allocate(shape=(16,), dtype=np.uint32)

try:
    fixed = float_to_fixed_bits(normalize_image(x_test[0])).reshape(N_PIXELS, 3)
    input_buffer[:, 0:3] = fixed
    input_buffer[:, 3] = 0
    output_buffer[:] = 0xDEADBEEF

    # Armar recepção antes de liberar a entrada.
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.transfer(input_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    logits = fixed_bits_to_float(output_buffer[:10])
    padding = np.asarray(output_buffer[10:16]).copy()
    pred = int(np.argmax(logits))

    print("ground truth:", int(y_test[0]))
    print("prediction  :", pred)
    print("logits      :", logits)
    print("padding     :", padding)

    if np.any(padding != 0):
        raise RuntimeError(f"Padding 320->512 não é zero: {padding}")
finally:
    input_buffer.close()
    output_buffer.close()
```

O acelerador não aparecerá necessariamente como periférico Python em
`ol.ip_dict`, pois não possui AXI-Lite. `axi_dma_0` deve aparecer.

## 15. Validação progressiva em placa

### Etapa A — uma imagem

Verificar:

- overlay carrega;
- DMA existe;
- `recvchannel.wait()` não trava;
- MM2S e S2MM retornam ao estado idle;
- nenhum erro DMA;
- padding `[10:16]` é zero;
- logits são finitos e plausíveis;
- classe prevista é coerente.

### Etapa B — repetição da mesma imagem

Executar a mesma imagem 100 vezes e verificar determinismo bit a bit. Diferenças
indicam reset, packing, buffer/cache ou protocolo incorreto.

### Etapa C — comparação contra referência

Comparar os logits do FPGA com:

```text
hls4ml/builds/fifo_opt_q22_12_rf288/csim/build/tb_data/csim_results.log
```

para as imagens preservadas no testbench, ou gerar uma referência C++ coerente.

### Etapa D — 100 imagens

Medir:

- acurácia;
- travamentos;
- erros DMA;
- estabilidade dos logits.

### Etapa E — 10.000 imagens

O projeto antigo obteve 74,92%. O novo sistema deve repetir o benchmark completo,
mas esse número não deve ser imposto artificialmente: qualquer diferença deve ser
investigada por packing, versão do modelo, quantização ou IP.

## 16. Benchmark final

Usar como base o script do projeto anterior:

<https://github.com/MiguelBertonzin/Resnet8-Acc-FPGA/blob/main/benchmark/benchmark_zcu104.py>

Métricas obrigatórias:

- acurácia top-1 em 10.000 imagens;
- latência média, mediana, p95 e p99;
- vazão/FPS;
- métricas inference-only;
- métricas end-to-end;
- potência idle e ativa;
- potência dinâmica;
- energia total e dinâmica por inferência;
- recursos pós-route;
- frequência atingida e slack.

Definições:

```text
inference-only:
programação DMA + DDR/MM2S + acelerador + S2MM/DDR + polling

end-to-end:
normalização + quantização/packing + inference-only + decode + argmax

potência dinâmica = potência ativa - potência idle
energia/inferência = potência / FPS
```

No sistema de referência, o INA226 estava em:

```text
/sys/class/hwmon/hwmon2/power1_input
```

O caminho deve ser redescoberto na imagem PYNQ atual; não assumir `hwmon2`.

## 17. Diagnóstico de falhas

### IP não aparece no catálogo

```tcl
get_property IP_REPO_PATHS [current_project]
update_ip_catalog -rebuild
get_ipdefs -all -quiet *:hls:resnet8_resource_fifo_opt:*
```

Confirmar que o caminho aponta para a pasta que contém `component.xml`, não para
o ZIP nem para a pasta pai errada.

### IP 2.2 aparece junto com 1.1

Remover o repo antigo de `IP_REPO_PATHS`, reconstruir catálogo e instanciar o VLNV
completo 1.1. Nunca usar `get_ipdefs` com padrão ambíguo para `create_bd_cell`.

### Module Reference não aparece

```tcl
get_files *Accel_dma_wrapper.vhd
update_compile_order -fileset sources_1
can_resolve_reference Accel_dma_wrapper
```

Revisar erros de compilação VHDL antes de criar o BD.

### `validate_bd_design` acusa largura incompatível

Conferir:

```text
DMA MM2S stream = 128
wrapper s_axis  = 128
wrapper accel input = 96
IP input = 96

IP output = 320
wrapper accel output = 320
wrapper m_axis = 512
DMA S2MM stream = 512
```

### `recvchannel.wait()` trava

Ordem de investigação:

1. `recvchannel.transfer()` foi chamado antes do send;
2. `m_axis_tlast` está conectado e vale 1 no beat válido;
3. `m_axis_tvalid && m_axis_tready` ocorreu;
4. saída do IP não ficou bloqueada por `TREADY=0`;
5. `ap_start=1` chegou ao IP;
6. IP saiu de reset;
7. S2MM address/length estão válidos;
8. DMA não reporta `DMAIntErr`, `DMASlvErr` ou `DMADecErr`;
9. inserir ILA temporário em `TVALID/TREADY/TLAST`, se necessário.

### Send trava

Verificar `input_layer_TREADY`, reset, `ap_start`, clock e exatamente 1.024 beats.
O IP não usa `TLAST` na entrada; o TLAST produzido pelo MM2S pode ser descartado
pelo wrapper.

### Logits absurdos, mas DMA termina

Possíveis causas:

- float32 enviado diretamente;
- escala diferente de 1024;
- ordem CHW em vez de HWC;
- ordem BGR em vez de RGB;
- 22 bits colocados no topo da lane, não nos bits inferiores;
- leitura de logit negativo como `int32` sem estender o bit 21;
- modelo/dataset normalizados de forma diferente.

### Padding não é zero

Indica wrapper incorreto, bitstream/HWH desencontrados ou buffer não realmente
sobrescrito pelo DMA.

### Síntese cabe, mas route falha

Coletar antes de alterar:

```tcl
report_design_analysis -congestion
report_route_status
report_timing_summary
report_utilization -hierarchical
```

Comparar a hierarquia do acelerador, wrapper e interconexões. O projeto antigo
fechou com estratégia padrão; o atual possui URAM e FIFOs maiores, portanto pode
exigir decisões físicas diferentes.

### Timing falha em caminhos de URAM

O OOC atual emitiu recomendações de pipeline para três URAMs grandes. Não alterar
o RTL empacotado impulsivamente. Primeiro confirmar o endpoint e o slack pós-route.
Se o problema estiver realmente nessas memórias, avaliar uma nova síntese HLS com
pipeline apropriado ou clock menor como experimento controlado.

## 18. Warnings: política de tratamento

Warnings não devem ser eliminados apenas por estética, mas todos devem ser
classificados.

Aceitáveis somente após análise:

- OCM/QSPI excluídos do address map;
- adaptações USER em HPC0;
- portas de metadados HLS não utilizadas;
- sinais de status HLS usados apenas para observação;
- recomendações de pipeline que não violam timing.

Bloqueadores:

- IP locked/unresolved;
- clock ou reset sem conexão;
- endereço do DMA não atribuído;
- largura/protocolo AXIS incompatível;
- critical warning ligado a timing, CDC ou constraints;
- nets não roteadas;
- WNS/WHS negativos;
- DRC error;
- mismatch de part/board;
- uso do IP errado.

## 19. Checklist antes da síntese

- [ ] Vivado 2024.2.
- [ ] Board `xilinx.com:zcu104:part0:1.1`.
- [ ] Part `xczu7ev-ffvc1156-2-e`.
- [ ] IP repo atual adicionado.
- [ ] VLNV da instância é exatamente 1.1.
- [ ] Wrapper VHDL compila.
- [ ] HPM0 e HPM1 habilitados.
- [ ] HPC0 habilitado.
- [ ] `pl_clk0 = 100 MHz`.
- [ ] Um único domínio principal de clock.
- [ ] Reset ativo-baixo coerente.
- [ ] SmartConnect controle 2 SI/1 MI.
- [ ] SmartConnect memória 2 SI/1 MI.
- [ ] DMA SG OFF e Micro OFF.
- [ ] DMA DRE ON em ambos os canais.
- [ ] MM2S 128/128, burst 64.
- [ ] S2MM 512/512, burst 64.
- [ ] Wrapper entrada 128→96.
- [ ] Wrapper saída 320→512.
- [ ] `ap_start=1`.
- [ ] `TLAST=output_valid`.
- [ ] DMA S_AXI_LITE acessível pelo PS.
- [ ] MM2S/S2MM acessam DDR por HPC0.
- [ ] Endereços atribuídos.
- [ ] `validate_bd_design` sem erros.
- [ ] `myproject_0` não está locked.
- [ ] `system_wrapper` é o top.

## 20. Checklist antes do bitstream

- [ ] Síntese concluída sem erros.
- [ ] Utilização do sistema cabe no dispositivo.
- [ ] Implementation Defaults testada primeiro.
- [ ] Place concluído.
- [ ] Route concluído.
- [ ] Zero failed/unrouted/partial nets.
- [ ] WNS e WHS não negativos.
- [ ] TNS e THS iguais a zero.
- [ ] DRC sem erros.
- [ ] Bitstream gerado.
- [ ] `.bit` e `.hwh` correspondem ao mesmo build.
- [ ] `system_bd_ip11.tcl` exportado.
- [ ] Relatórios preservados.

## 21. Checklist de validação física

- [ ] Overlay carrega sem erro.
- [ ] `axi_dma_0` aparece no HWH/PYNQ.
- [ ] Buffers possuem endereço compatível com DMA 32-bit.
- [ ] Receive é armado antes do send.
- [ ] MM2S termina.
- [ ] S2MM termina.
- [ ] Nenhum erro DMA.
- [ ] Padding de saída é zero.
- [ ] Logits são decodificados com sinal no bit 21.
- [ ] Uma imagem coincide com referência.
- [ ] 100 repetições são determinísticas.
- [ ] Smoke test de 100 imagens é coerente.
- [ ] Teste de 10.000 imagens concluído.
- [ ] Latência/FPS medidos após warm-up.
- [ ] Potência e energia usam metodologia registrada.

## 22. Resultados antigos: referência, não aceitação do novo projeto

O sistema antigo 2.2 comprovou a viabilidade desta arquitetura:

```text
100 MHz
LUT pós-route       = 178.378 (77,42%)
Registers           = 197.981 (42,96%)
BRAM tiles          = 104,5 (33,49%)
DSP48E2             = 1.174 (67,94%)
URAM                 = 0
WNS                  = +0,257 ns
WHS                  = +0,009 ns
routing errors       = 0
accuracy CIFAR-10    = 74,92% em 10.000 imagens
latência média FPGA = 0,8416 ms
throughput FPGA      = 1.196,39 FPS
```

Esses números devem aparecer em comparações como “projeto anterior”. O novo IP
1.1 só recebe valores próprios depois que os relatórios e benchmarks atuais forem
gerados.

## 23. Artefatos a entregar ao final

Estrutura recomendada:

```text
hardware/
├── overlay/
│   ├── system_wrapper.bit
│   └── system_wrapper.hwh
├── reports/
│   ├── utilization_synth.rpt
│   ├── utilization_post_route.rpt
│   ├── timing_post_route.rpt
│   ├── route_status.rpt
│   ├── drc_post_route.rpt
│   └── power_post_route.rpt
├── src/
│   └── Accel_dma_wrapper.vhd
├── system_bd_ip11.tcl
└── system_wrapper_ip11.xsa

benchmark/
├── smoke_test_zcu104.py
└── benchmark_zcu104_ip11.py

results/
└── metrics_ip11_<data>.json
```

Também preservar os manifests do IP e os checksums dos modelos, bitstream e
relatórios.

## 24. Referências técnicas

- Projeto anterior funcional:
  <https://github.com/MiguelBertonzin/Resnet8-Acc-FPGA>.
- AXI DMA Product Guide PG021:
  <https://docs.amd.com/r/en-US/pg021_axi_dma/Core-Overview>.
- Requisito de término S2MM por `TLAST`:
  <https://docs.amd.com/r/en-US/pg021_axi_dma/S2MM>.
- Direct Register/Simple DMA:
  <https://docs.amd.com/r/en-US/pg021_axi_dma/Direct-Register-Mode-Simple-DMA>.
- hls4ml FIFO depth optimization:
  <https://fastmachinelearning.org/hls4ml/advanced/fifo_depth.html>.

## 25. Próxima ação exata

Ao iniciar a conversa no ChatGPT Web, não peça de uma vez “gere o bitstream”.
Comece pela auditoria:

```text
1. validar caminhos e checksums;
2. criar projeto limpo;
3. registrar somente o IP repo 1.1;
4. adicionar/compilar o wrapper;
5. confirmar VLNV no catálogo;
6. retornar os outputs ao ChatGPT;
7. só então criar o Block Design.
```

Essa sequência reduz o risco de construir durante horas com o IP errado, wrapper
ausente, endereço incorreto ou contrato AXI incompatível.
