# Estado do projeto em 31/08/2026

## Resumo executivo

Este documento registra o estado consolidado do trabalho realizado nesta pasta até 31/08/2026.

O projeto começou com a avaliação de uma ResNet8 para classificação do CIFAR-10 em CPU e GPU, com e sem a camada softmax, e evoluiu até uma implementação funcional na FPGA da AMD/Xilinx ZCU104 usando hls4ml, Vitis HLS, Vivado, ponto fixo e comunicação por AXI DMA.

O foco final passou a ser a variante sem softmax, porque a classificação top-1 pode ser obtida diretamente por `argmax(logits)` e a remoção da softmax simplifica o hardware sem alterar a classe prevista.

Estado alcançado:

- modelo sem softmax criado e funcionalmente validado;
- benchmarks controlados concluídos em CPU e GPU;
- conversão hls4ml concluída;
- síntese HLS e RTL co-simulation concluídas;
- profundidades FIFO perfiladas e otimizadas;
- pacote IP corrigido e validado no Vivado;
- IP 1.1 integrado em um sistema Zynq + AXI DMA;
- bitstream e HWH gerados e executados fisicamente na ZCU104;
- acurácia validada nas 10.000 imagens do CIFAR-10;
- campanha final de 1.000.000 de inferências concluída;
- latência, throughput e consumo energético caracterizados.

O resultado físico principal da ZCU104 foi:

```text
Acurácia top-1       = 74,9200%
Latência média       = 0,849300 ms
Mediana              = 0,849360 ms
p95                  = 0,862090 ms
p99                  = 0,875650 ms
FPS efetivo global   = 899,2063 FPS
Inferências medidas  = 1.000.000
Clock do acelerador  = 100 MHz
```

Não foram alterados nem removidos outliers das campanhas.

## 1. Organização geral da pasta

Os principais grupos de arquivos são:

```text
resnet8/
├── resnet8_cifar10_keras3.h5
├── resnet8_cifar10_keras3_no_softmax.h5
├── metodologia-benchmark.txt
├── info.txt
├── CPU/
│   ├── resnet8-softmax/
│   └── resnet8-no-softmax/
├── GPU/
│   ├── resnet8-softmax/
│   └── resnet8-no-softmax/
├── hls4ml/
├── hardware/src/Accel_dma_wrapper.vhd
├── scripts modelo/remove_softmax.py
├── RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt
├── RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt
└── Vitis AI/
```

A pasta não é um repositório Git utilizável. A reconstrução cronológica deste documento foi feita a partir de datas, metadados, relatórios, scripts, manifests, hashes e arquivos de resultados.

A pasta `Vitis AI/` permanece vazia. O fluxo efetivamente desenvolvido e concluído foi o fluxo hls4ml/Vitis HLS/Vivado.

## 2. Modelos avaliados

Existem dois modelos Keras:

### 2.1 Modelo original com softmax

Arquivo:

```text
resnet8_cifar10_keras3.h5
```

Características:

- saída com 10 probabilidades;
- última ativação: `softmax`;
- 78.714 parâmetros;
- 31 camadas Keras;
- SHA-256:

```text
9cc7cd3ea1c9603501f0c77dc0848b41d0bd136f80cc1d328881ed89cbb9cb1b
```

### 2.2 Modelo sem softmax

Arquivo:

```text
resnet8_cifar10_keras3_no_softmax.h5
```

Características:

- saída com 10 logits;
- última ativação: `linear`;
- 78.714 parâmetros;
- 31 camadas Keras;
- SHA-256:

```text
047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e
```

O modelo foi gerado por:

```text
scripts modelo/remove_softmax.py
```

O script clona o modelo, troca somente a ativação da última camada Dense por `linear`, copia exatamente os pesos e verifica a reconstrução da softmax.

### 2.3 Validação da remoção da softmax

A remoção da softmax foi validada separadamente na CPU e na GPU:

- 47 arrays de pesos exatamente iguais;
- erro absoluto máximo nos pesos: zero;
- mesma quantidade de parâmetros;
- mesma forma de entrada e saída;
- `argmax(softmax)` e `argmax(logits)` iguais em 10.000/10.000 imagens dentro de cada dispositivo;
- erro máximo ao reconstruir `softmax(logits)` de aproximadamente `2,384 × 10^-7`.

Para classificação top-1:

```text
argmax(softmax(z)) = argmax(z)
```

Assim, a softmax não é necessária no acelerador. Caso probabilidades sejam necessárias futuramente, elas podem ser calculadas no ARM/PS após a leitura dos dez logits.

Arquivos de evidência:

```text
CPU/resnet8-no-softmax/resultados/validacao_equivalencia.json
GPU/resnet8-no-softmax/resultados/validacao_equivalencia.json
hls4ml/reports/model_audit.json
hls4ml/docs/DECISAO_SOFTMAX.md
```

## 3. Arquitetura da ResNet8

O modelo recebe imagens CIFAR-10 de `32 × 32 × 3` e possui, em termos gerais:

1. convolução inicial `3 × 3` com 16 canais;
2. Batch Normalization e ReLU;
3. bloco residual com duas convoluções de 16 canais;
4. bloco residual de redução para 32 canais, com atalho `1 × 1`;
5. bloco residual de redução para 64 canais, com atalho `1 × 1`;
6. Global Average Pooling;
7. Dense com 10 saídas lineares.

O código C++ gerado pelo hls4ml contém:

- 9 convoluções, incluindo as projeções `1 × 1` dos atalhos;
- 9 normalizações;
- 7 ativações ReLU;
- 3 somas residuais;
- Global Average Pooling;
- Dense final com 10 logits.

## 4. Metodologia dos benchmarks CPU/GPU

A metodologia consolidada encontra-se em:

```text
metodologia-benchmark.txt
```

### 4.1 Dataset

Foi utilizada a divisão oficial de teste do CIFAR-10:

- 10.000 imagens;
- resolução `32 × 32`;
- três canais RGB;
- 10 classes;
- 1.000 imagens por classe.

Normalização:

```python
test_images = test_images.astype(np.float32) / 255.0
```

Foram construídos conjuntos aninhados, estratificados e balanceados:

- 100 imagens: 10 por classe;
- 1.000 imagens: 100 por classe;
- 10.000 imagens: 1.000 por classe.

Seed:

```text
20260825
```

### 4.2 Configuração comum

- TensorFlow 2.21.0;
- Keras 3.13.2;
- NumPy 1.26.4;
- Python 3.12.7;
- precisão `float32`;
- batch 1;
- execução serial e síncrona;
- `tf.function` com assinatura fixa `(1, 32, 32, 3)`;
- `jit_compile=False`;
- determinismo do TensorFlow habilitado;
- 100 inferências de aquecimento.

Não foram usados:

- XLA;
- TensorRT;
- quantização;
- mixed precision;
- batch maior que 1;
- múltiplas inferências simultâneas.

### 4.3 Repetições

Para cada combinação de dispositivo e modelo:

```text
100 imagens   × 100 ciclos =    10.000 inferências
1.000 imagens × 100 ciclos =   100.000 inferências
10.000 imagens × 100 ciclos = 1.000.000 inferências
Total                          1.110.000 latências
```

As linhas de 10, 20, 50 e 100 ciclos são prefixos cumulativos da mesma execução de 100 ciclos, e não experimentos independentes.

### 4.4 Latência

O intervalo individual foi:

```python
batch = tf.convert_to_tensor(image)
t0 = time.perf_counter_ns()
output = infer(batch)
host_output = output.numpy()
t1 = time.perf_counter_ns()
```

A normalização e a criação do tensor ficaram fora do cronômetro individual. A chamada `.numpy()` força a conclusão da inferência e a disponibilização da saída no host.

### 4.5 Throughput

O throughput foi calculado pelo tempo total de cada ciclo e inclui:

- laço Python;
- seleção da imagem;
- criação do tensor;
- inferência;
- sincronização;
- `argmax`;
- armazenamento da predição.

Foi preservada tanto a média dos FPS dos ciclos quanto o FPS efetivo global:

```text
FPS efetivo global = total de inferências / soma dos tempos dos ciclos
```

### 4.6 Estatística

Nenhum outlier foi removido.

Foram calculados:

- média;
- mediana;
- desvio-padrão amostral;
- p95;
- mínimo;
- máximo;
- IC95% da média, usando os ciclos como unidades;
- IC95% de Wilson para acurácia.

## 5. Resultados em CPU e GPU

Hardware utilizado:

- CPU: Intel Core i7-13700, 16 núcleos físicos e 24 CPUs lógicas;
- GPU: NVIDIA GeForce RTX 3050 OEM, 8 GB, driver 580.173.02.

### 5.1 Resultados sem softmax

Resultado recomendado: 10.000 imagens × 100 ciclos.

| Métrica | CPU i7-13700 | GPU RTX 3050 |
|---|---:|---:|
| Inferências cronometradas | 1.000.000 | 1.000.000 |
| Acurácia | 74,9000% | 74,8900% |
| Acertos | 7.490 | 7.489 |
| Latência média | 0,9760 ms | 0,6912 ms |
| Mediana | 0,9028 ms | 0,5122 ms |
| Desvio-padrão | 0,2447 ms | 0,2609 ms |
| p95 | 1,5704 ms | 1,1219 ms |
| FPS médio dos ciclos | 975,04 | 1.268,49 |
| FPS efetivo global | 975,01 | 1.223,84 |
| Potência medida | 57,41 W | 35,18 W |
| Energia total/inferência | 58,36 mJ | 28,46 mJ |

Escopo de potência:

- CPU: potência do pacote estimada por RAPL/turbostat;
- GPU: potência da placa medida pelo sensor NVIDIA.

Esses escopos não são fisicamente idênticos.

### 5.2 Diferença CPU/GPU de uma imagem

A CPU acertou 7.490 imagens e a GPU acertou 7.489. Isso é compatível com pequenas diferenças numéricas entre oneDNN e CUDA/cuDNN em uma imagem próxima à fronteira de decisão.

Não houve divergência entre o modelo com softmax e o modelo com logits dentro do mesmo dispositivo.

### 5.3 Comparação com a variante softmax

Resultado de 10.000 imagens × 100 ciclos:

| Dispositivo | Com softmax | Sem softmax |
|---|---:|---:|
| CPU, latência média | 0,9551 ms | 0,9760 ms |
| GPU, latência média | 0,6110 ms | 0,6912 ms |

A variante sem softmax não apareceu mais rápida nessas sessões. Isso não demonstra que a softmax acelera o modelo. As coletas foram sequenciais, não pareadas ou intercaladas, e podem ter sido afetadas por:

- DVFS;
- temperatura;
- escalonamento;
- cache;
- estado do runtime;
- escolhas internas de oneDNN/cuDNN.

A conclusão cientificamente defensável é apenas que cada série é consistente internamente. Para isolar causalmente o custo da softmax seriam necessárias rodadas alternadas e pareadas.

## 6. Fluxo hls4ml

O fluxo está documentado em:

```text
hls4ml/README.md
```

### 6.1 Ambiente

- Python 3.12.7;
- TensorFlow 2.21.0;
- Keras 3.13.2;
- hls4ml 1.3.0;
- NumPy 1.26.4;
- Vitis HLS 2024.2;
- Vivado 2024.2;
- FPGA `xczu7ev-ffvc1156-2-e`.

### 6.2 Configuração do acelerador

```text
Backend                 = Vitis
IOType                  = io_stream
Strategy                = Resource
ConvImplementation      = LineBuffer
Clock solicitado        = 10 ns / 100 MHz
Precisão                 = ap_fixed<22,12,AP_RND_CONV,AP_SAT>
ReuseFactor máximo       = 288
Modelo                   = sem softmax
```

`ap_fixed<22,12>` possui:

- 22 bits totais;
- 12 bits inteiros, incluindo sinal;
- 10 bits fracionários;
- escala equivalente de 1024.

### 6.3 Reuse factors efetivos

O plano solicitado foi ajustado para fatores matematicamente válidos:

| Camada | RF solicitado | RF efetivo |
|---|---:|---:|
| conv2d | 8 | 9 |
| conv2d_1 | 32 | 36 |
| conv2d_2 | 32 | 36 |
| conv2d_3 | 64 | 72 |
| conv2d_4 | 128 | 144 |
| conv2d_5 | 64 | 64 |
| conv2d_6 | 288 | 288 |
| conv2d_7 | 288 | 288 |
| conv2d_8 | 128 | 128 |
| dense | 64 | 64 |

Evidência:

```text
hls4ml/builds/fifo_opt_q22_12_rf288/reuse_plan_resolved.csv
```

### 6.4 Validação C++ inicial

O baseline foi validado em 20 imagens:

- concordância top-1 Keras/hls4ml: 100%;
- acurácia Keras no pequeno subconjunto: 95%;
- acurácia hls4ml no pequeno subconjunto: 95%;
- MAE dos logits: 0,081272;
- erro absoluto máximo: 0,319565.

Esse conjunto de 20 imagens foi apenas uma validação rápida do fluxo. A validação física posterior utilizou as 10.000 imagens oficiais.

### 6.5 Problema de caminhos com espaços

O caminho do projeto contém espaços. O Vitis HLS 2024.2 não aceita esses caminhos em determinadas etapas. Os scripts criam aliases temporários sem espaços em `/tmp`, mantendo os arquivos reais dentro desta pasta.

## 7. Otimização das FIFOs

Como o projeto usa `io_stream`, as camadas são interligadas por FIFOs.

O fluxo utilizado foi:

```text
vitis:fifo_depth_optimization
```

Configuração:

- profundidade temporária de profiling: 4096;
- amostras mínimas do testbench: 2;
- total de FIFOs: 42;
- maior profundidade otimizada: 2100;
- FIFOs saturadas no limite: nenhuma;
- profiling considerado válido.

Profundidades grandes relevantes:

```text
2100
1464
1351
69
45
42
30
23
21
14
```

Grande parte dos demais canais foi reduzida para profundidade 1.

Evidências:

```text
hls4ml/builds/fifo_opt_q22_12_rf288/fifo_depths.json
hls4ml/builds/fifo_opt_q22_12_rf288/fifo_profile_summary.json
hls4ml/builds/fifo_opt_q22_12_rf288/fifo_rtl_audit.json
```

## 8. Síntese HLS

Resultados da C synthesis final:

| Métrica | Resultado |
|---|---:|
| Clock solicitado | 10,000 ns |
| Clock estimado | 7,298 ns |
| Latência estimada | 85.026 a 85.313 ciclos |
| Latência a 100 MHz | 0,850 a 0,853 ms |
| BRAM18 estimada | 722 |
| DSP estimado | 1.350 |
| FF estimado | 192.641 |
| LUT estimada | 370.925 |
| URAM estimada | 0 |

O HLS estimou recursos acima da capacidade da ZCU104, especialmente LUT e BRAM. Essa estimativa não representou o resultado final do mapeamento no Vivado.

A RTL co-simulation foi aprovada e as saídas CSim/Cosim das amostras foram iguais.

## 9. Problema do export padrão e criação do IP 1.1

O export padrão do Vitis HLS 2024.2 reintroduziu profundidade 4096 nos 42 FIFOs, apesar de o top sintetizado usar as profundidades otimizadas.

Consequências:

- o pacote 1.0 gerado diretamente pelo `export_design` foi rejeitado;
- o RTL correto continuava presente em `solution1/syn/verilog`;
- foi criado um script para reconstruir o pacote usando o scaffold IP-XACT do Vitis e o RTL final correto.

Script:

```text
hls4ml/scripts/05_make_correct_ip.py
```

O script:

- copia o RTL final otimizado;
- remove VHDL e artefatos residuais do scaffold;
- remove módulos FIFO `d4096` não utilizados;
- atualiza os file sets do `component.xml`;
- verifica módulos FIFO referenciados;
- verifica o top por hash;
- gera o ZIP corrigido.

IP final:

```text
xilinx.com:hls:resnet8_resource_fifo_opt:1.1
```

Pacote válido:

```text
hls4ml/ip_exports/xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip
```

SHA-256 do ZIP:

```text
4e148c361116d19430aabd95e00b0dc3ca6789821e83602f1cd5d70313095191
```

O ZIP 1.0 produzido diretamente pelo Vitis não deve ser usado.

## 10. Validação Vivado do IP 1.1

O IP corrigido passou por:

- registro no catálogo Vivado;
- `validate_ip`;
- geração de todos os targets;
- síntese Out-of-Context do XCI;
- progresso OOC de 100%;
- ausência de instâncias `d4096` no top.

### 10.1 Recursos OOC do IP 1.1

| Recurso | Uso | Disponível | Ocupação |
|---|---:|---:|---:|
| CLB LUT | 165.564 | 230.400 | 71,86% |
| Registradores | 178.894 | 460.800 | 38,82% |
| BRAM tiles | 74,5 | 312 | 23,88% |
| URAM | 11 | 96 | 11,46% |
| DSP48E2 | 1.174 | 1.728 | 67,94% |

O Vivado reduziu substancialmente as estimativas do HLS. As FIFOs maiores foram mapeadas em URAM, o que explica o uso de 11 URAMs e a redução de BRAM.

Relatório consolidado:

```text
hls4ml/reports/VITIS_HLS_VS_VIVADO_IP_V1_1.md
```

## 11. Diferença entre o IP antigo 2.2 e o IP atual 1.1

Existem dois contextos históricos que não devem ser misturados.

### 11.1 Projeto antigo

O arquivo:

```text
RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt
```

documenta um projeto anterior funcional que usava:

```text
xilinx.com:hls:resnet8_resource_fifo_opt:2.2
```

Esse projeto antigo obteve:

```text
CLB LUT pós-route = 178.378 = 77,42%
Registradores     = 197.981 = 42,96%
BRAM tiles        = 104,5   = 33,49%
DSP48E2           = 1.174   = 67,94%
URAM              = 0
WNS               = +0,257 ns
TNS               = 0,000 ns
```

Esses valores são apenas referência histórica da arquitetura antiga.

### 11.2 Projeto atual

O projeto atual usa:

```text
xilinx.com:hls:resnet8_resource_fifo_opt:1.1
```

Ele possui FIFOs diferentes e utiliza 11 URAMs no resultado OOC.

O HWH preservado nas campanhas físicas confirma explicitamente que o bloco integrado é o IP 1.1.

Portanto:

- não usar os números pós-route do IP 2.2 como resultados do 1.1;
- não substituir o IP 1.1 pelo 2.2;
- não usar padrões ambíguos ao localizar o VLNV no Vivado;
- sempre confirmar o VLNV completo da instância `myproject_0`.

O documento `RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt` foi escrito antes da integração física do IP 1.1. Ele é útil como handoff e procedimento de construção, mas seu estado de “ainda não implementado” ficou desatualizado depois das campanhas na placa.

## 12. Arquitetura do sistema na ZCU104

A arquitetura física utilizada foi:

```text
CPU ARM / PYNQ
       |
       | programa o AXI DMA por AXI-Lite
       v
DDR <-> AXI DMA <-> Accel_dma_wrapper <-> ResNet8 hls4ml IP 1.1
```

Blocos principais:

- `zynq_ultra_ps_e_0`;
- `rst_ps8_0_100M`;
- `axi_smc` para controle;
- `axi_smc_1` para memória;
- `axi_dma_0`;
- `Accel_dma_wrapper_0`;
- `myproject_0`, o IP ResNet8 1.1.

### 12.1 Interfaces do Processing System

Foram utilizadas:

- `M_AXI_HPM0_FPD`;
- `M_AXI_HPM1_FPD`;
- `S_AXI_HPC0_FPD`;
- `pl_clk0` a 100 MHz.

O DMA acessa a DDR pela porta HPC0.

### 12.2 AXI DMA

Configuração essencial:

- Scatter Gather: desabilitado;
- Micro DMA: desabilitado;
- MM2S: habilitado;
- S2MM: habilitado;
- stream MM2S: 128 bits;
- stream S2MM: 512 bits;
- controle AXI-Lite em `0xA0000000`, faixa de 64 KiB;
- DDR acessível de `0x00000000` a `0x7FFFFFFF`;
- interrupções desconectadas;
- software operando por polling.

### 12.3 Wrapper VHDL

Arquivo:

```text
hardware/src/Accel_dma_wrapper.vhd
```

Funções do wrapper:

- adaptar entrada DMA de 128 bits para entrada hls4ml de 96 bits;
- preservar `TVALID/TREADY`;
- adaptar saída hls4ml de 320 bits para saída DMA de 512 bits;
- preencher os 192 bits superiores com zero;
- gerar `TLAST` no único beat de saída;
- manter `ap_start='1'`;
- ligar os sinais de controle e handshake do HLS.

O wrapper não implementa FIFO ou buffering adicional.

## 13. Packing numérico e buffers

### 13.1 Entrada

Cada pixel possui R, G e B quantizados para `ap_fixed<22,12>` e armazenados em três palavras de 32 bits.

Formato de um beat MM2S de 128 bits:

```text
bits  31:0   = R
bits  63:32  = G
bits  95:64  = B
bits 127:96  = padding zero
```

O wrapper envia somente os 96 bits inferiores ao acelerador.

Buffer PYNQ:

```python
input_buffer = allocate(shape=(1024, 4), dtype=np.uint32)
```

Uma imagem possui 1.024 pixels e ocupa:

```text
1024 beats × 16 bytes = 16.384 bytes
```

### 13.2 Saída

O acelerador produz 10 logits, cada um transportado em uma lane de 32 bits:

```text
10 × 32 bits = 320 bits
```

O wrapper completa a saída para 512 bits:

```text
bits 319:0   = 10 logits
bits 511:320 = zero
```

Buffer PYNQ:

```python
output_buffer = allocate(shape=(16,), dtype=np.uint32)
```

O DMA recebe um único beat de 64 bytes e `TLAST=1`.

## 14. Evidência da integração atual

As campanhas intermediária e final preservaram o mesmo bitstream e HWH.

SHA-256 do bitstream:

```text
4e7f65c0fe6cd3356f22a0823b9f39557c550e7150d0c52fb155d406f8344e59
```

SHA-256 do HWH:

```text
e9de495557b715ff5036e9202aa76e79b8d0681491e3345b9e3d2d8993f95ca3
```

O HWH registra:

- `myproject_0` com VLNV `xilinx.com:hls:resnet8_resource_fifo_opt:1.1`;
- clock de 100 MHz;
- DMA MM2S de 128 bits;
- DMA S2MM de 512 bits;
- endereço AXI-Lite do DMA `0xA0000000`;
- conexões do wrapper ao DMA e ao acelerador;
- DDR pela `S_AXI_HPC0_FPD`.

Isso comprova que os resultados físicos foram obtidos com o IP atual 1.1.

## 15. Campanha intermediária na ZCU104

Diretório:

```text
hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/
```

Ambiente:

- ZCU104;
- PYNQ 3.1.1;
- Python 3.10.4;
- NumPy 1.21.5;
- kernel Xilinx 2024.1;
- clock do acelerador: 100 MHz;
- potência medida no sensor PMBus `12V_power`;
- intervalo definitivo de potência: 1 segundo.

O relógio do sistema embarcado estava incorreto. Alguns metadados internos mostram 2025, mas a data real da coleta foi 28/08/2026, conforme `ENVIRONMENT.json`.

### 15.1 Cenário inference batch 1

Foram utilizadas 5.000 imagens estratificadas e 20 ciclos, totalizando 100.000 inferências.

Fronteira da latência individual:

```text
entrada já presente no PynqBuffer
→ DMA MM2S
→ ResNet8 FPGA
→ DMA S2MM
→ wait()
```

Fronteira do throughput:

```text
cópia da imagem já empacotada para PynqBuffer
→ DMA/FPGA/DMA
→ decode dos logits
→ argmax
→ próxima imagem
```

Resultados:

| Métrica | Resultado |
|---|---:|
| Latência média | 0,8451 ms |
| Mediana | 0,8456 ms |
| p95 | 0,8592 ms |
| p99 | 0,8759 ms |
| FPS efetivo | 901,08 |
| Potência idle | 10,6094 W |
| Potência ativa | 12,2283 W |
| Potência dinâmica | 1,6189 W |
| Energia total/inferência | 13,4994 mJ |
| Energia dinâmica/inferência | 1,7871 mJ |

### 15.2 Cenário end-to-end

Fronteira:

```text
imagem uint8 em RAM
→ float32
→ normalização /255
→ quantização ap_fixed<22,12>
→ packing
→ PynqBuffer
→ DMA/FPGA/DMA
→ decode
→ argmax
```

Resultados:

| Métrica | Resultado |
|---|---:|
| Latência média | 1,5977 ms |
| Mediana | 1,5930 ms |
| p95 | 1,6155 ms |
| p99 | 1,6699 ms |
| FPS efetivo | 620,70 |
| Potência idle | 10,6270 W |
| Potência ativa | 11,8088 W |
| Potência dinâmica | 1,1818 W |
| Energia total/inferência | 19,0992 mJ |
| Energia dinâmica/inferência | 1,9114 mJ |

### 15.3 Caminho acelerado com entrada pronta

Esse cenário foi chamado de “saturado”, mas ainda é serial, síncrono e batch 1. Não foram usadas inferências simultâneas, double buffering ou várias requisições pendentes.

Fronteira:

```text
mesma entrada já presente no PynqBuffer
→ DMA MM2S
→ FPGA
→ DMA S2MM
→ wait()
→ repetir
```

Resultados:

| Métrica | Resultado |
|---|---:|
| Latência média, 50.000 inferências | 0,832636 ms |
| Mediana | 0,833510 ms |
| p95 | 0,845190 ms |
| p99 | 0,850360 ms |
| Taxa equivalente pela latência | 1.201,01 FPS |
| Throughput sustentado medido | 1.200,94 FPS |
| Potência idle | 10,6170 W |
| Potência ativa | 12,6580 W |
| Potência dinâmica | 2,0410 W |
| Energia total/inferência | 10,7266 mJ |
| Energia dinâmica/inferência | 1,7295 mJ |

A diferença entre a taxa derivada da latência e o throughput medido foi aproximadamente 0,005%, fornecendo validação cruzada da medição.

## 16. Campanha final ZCU104, 10.000 × 100

Diretório:

```text
hls4ml/Resultados_ZCU104/resnet8_ip11_final_10k100_2026-08-28/
```

O objetivo foi aumentar a paridade experimental com CPU/GPU:

- 10.000 imagens oficiais;
- 100 ciclos completos;
- 1.000.000 de inferências;
- batch 1;
- execução serial e síncrona;
- 100 inferências de warm-up;
- seed `20260825`;
- nenhum outlier removido.

### 16.1 Acurácia

A validação de acurácia foi feita separadamente da medição de desempenho:

```text
Imagens únicas = 10.000
Acertos         = 7.492
Acurácia        = 74,9200%
IC95% Wilson    = [74,0609%; 75,7599%]
```

Foram preservados:

```text
predictions_accuracy_10000.npy
logits_accuracy_10000.npy
confusion_matrix_10000.npy
confusion_matrix_10000.csv
accuracy_10000.json
```

### 16.2 Latência inference-only

Fronteira:

```text
entrada já copiada para PynqBuffer
→ DMA MM2S
→ ResNet8 hls4ml
→ DMA S2MM
→ wait()
```

Ficam fora:

- carregamento do dataset;
- conversão `uint8 -> float32`;
- normalização;
- quantização;
- packing;
- cópia para o PynqBuffer;
- decode dos logits;
- `argmax`.

Resultados:

| Métrica | Resultado |
|---|---:|
| Inferências | 1.000.000 |
| Média | 0,849300 ms |
| Mediana | 0,849360 ms |
| Desvio-padrão | 0,015460 ms |
| CV | 1,8204% |
| p95 | 0,862090 ms |
| p99 | 0,875650 ms |
| Mínimo | 0,825990 ms |
| Máximo | 4,762210 ms |
| Taxa equivalente | 1.177,4402 FPS |

O valor máximo foi mantido nos dados.

### 16.3 Throughput efetivo

Fronteira:

```text
imagem previamente empacotada
→ cópia para PynqBuffer
→ DMA/FPGA/DMA
→ decode dos logits
→ argmax
→ próxima imagem
```

Resultados:

| Métrica | Resultado |
|---|---:|
| Média dos FPS dos ciclos | 899,2073 FPS |
| FPS efetivo global | 899,2063 FPS |
| Desvio entre ciclos | 0,9882 FPS |
| CV | 0,1099% |
| IC95% | [899,0136; 899,4010] FPS |
| Drift entre primeiros/últimos 20 ciclos | +0,1496% |
| Divergências de predição entre ciclos | 0 |

### 16.4 Convergência

| Ciclos | Inferências | Latência média | p95 | FPS efetivo global |
|---:|---:|---:|---:|---:|
| 10 | 100.000 | 0,849460 ms | 0,862230 ms | 899,230 |
| 20 | 200.000 | 0,849442 ms | 0,862120 ms | 899,136 |
| 50 | 500.000 | 0,849526 ms | 0,862200 ms | 898,905 |
| 100 | 1.000.000 | 0,849300 ms | 0,862090 ms | 899,206 |

Os resultados permaneceram altamente estáveis.

## 17. Comparação CPU × GPU × FPGA

Comparação principal sem softmax, usando 10.000 imagens × 100 ciclos:

| Métrica | CPU i7-13700 | GPU RTX 3050 | ZCU104 |
|---|---:|---:|---:|
| Acurácia | 74,90% | 74,89% | 74,92% |
| Latência média | 0,9760 ms | 0,6912 ms | 0,8493 ms |
| Mediana | 0,9028 ms | 0,5122 ms | 0,8494 ms |
| p95 | 1,5704 ms | 1,1219 ms | 0,8621 ms |
| FPS efetivo global | 975,01 | 1.223,84 | 899,21 |
| Potência ativa | 57,41 W | 35,18 W | 12,23 W* |
| Energia total/inferência | 58,36 mJ | 28,46 mJ | 13,50 mJ* |

`*` A potência e a energia da ZCU104 vieram da campanha intermediária de 5.000 imagens × 20 ciclos. A campanha final de 10.000 × 100 não repetiu a telemetria energética.

### 17.1 Interpretação

- A GPU teve a menor latência média.
- A ZCU104 apresentou latência aproximadamente 13,0% menor que a CPU.
- O FPS efetivo da ZCU104 ficou aproximadamente 7,8% abaixo da CPU.
- O FPS efetivo da ZCU104 ficou aproximadamente 26,5% abaixo da GPU.
- A ZCU104 teve p95 muito menor e comportamento temporal muito mais concentrado.
- A energia total por inferência da ZCU104 foi aproximadamente 76,9% menor que a CPU.
- A energia total por inferência da ZCU104 foi aproximadamente 52,6% menor que a GPU.

As fronteiras não são fisicamente idênticas:

- CPU/GPU usam seus runtimes TensorFlow;
- a latência FPGA mede DMA-FPGA-DMA com entrada já no PynqBuffer;
- o throughput FPGA inclui cópia da entrada empacotada, DMA, decode, `argmax` e Python;
- a potência CPU é do pacote;
- a potência GPU é da placa;
- a potência FPGA é do rail de 12 V da ZCU104.

Portanto, os resultados devem ser apresentados com essas ressalvas.

## 18. Acurácia e efeito da quantização

Resultados em 10.000 imagens:

```text
CPU float32  = 7.490 acertos = 74,90%
GPU float32  = 7.489 acertos = 74,89%
FPGA Q22.12  = 7.492 acertos = 74,92%
```

A diferença de poucas imagens é compatível com pequenas perturbações numéricas próximas às fronteiras de decisão.

Não se deve interpretar os dois ou três acertos adicionais da FPGA como melhoria comprovada de generalização. Todos os resultados são estatisticamente compatíveis e usam o mesmo modelo treinado.

## 19. Pontos fortes do trabalho

- Modelos-fonte preservados com hashes.
- Remoção da softmax validada numericamente.
- Scripts CPU/GPU reproduzíveis e retomáveis.
- Dados brutos de latência preservados em NPY.
- Resultados por ciclo preservados em CSV.
- Metadados de ambiente preservados em JSON.
- Seed e amostragem estratificada documentadas.
- Nenhum outlier removido.
- IC95% e Wilson calculados.
- Telemetria de CPU, GPU e FPGA preservada.
- Conversão hls4ml automatizada.
- Reuse factors resolvidos e registrados.
- FIFO profiling documentado.
- RTL final auditado.
- Pacote IP corrigido com hashes.
- Validação de catálogo e OOC concluída.
- Bitstream e HWH preservados.
- Validação física em todas as 10.000 imagens.
- Campanha final com 1.000.000 de inferências.
- Ausência de divergências de predição entre ciclos na FPGA.

## 20. Lacunas e cuidados para trabalho futuro

### 20.1 Relatórios pós-route do IP 1.1

O bitstream e os benchmarks físicos provam que o sistema 1.1 foi integrado e executado. Entretanto, não foram encontrados nesta pasta os relatórios pós-route do sistema completo atual:

- `utilization_post_route.rpt`;
- `timing_post_route.rpt`;
- `route_status.rpt`;
- `drc_post_route.rpt`;
- `power_post_route.rpt`;
- projeto `.xpr` ou Tcl completo da implementação atual.

Por isso, não há nesta pasta evidência documental suficiente para publicar como resultados do IP 1.1:

- utilização total pós-route;
- WNS/TNS/WHS/THS atuais;
- frequência máxima atual;
- estimativa Vivado de potência atual.

Os números pós-route do IP 2.2 não devem ser reutilizados para preencher essa lacuna.

### 20.2 Documentação temporalmente desatualizada

- `RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt` descreve o projeto antigo 2.2.
- `RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt` foi escrito antes da integração física do 1.1.
- Este arquivo, `ESTADO-31-08.md`, deve ser usado como referência de estado consolidado até 31/08/2026.

### 20.3 Organização da pasta

- não existe controle de versão Git nesta pasta;
- existem arquivos `.orig` e `__pycache__`;
- os scripts softmax/no-softmax e CPU/GPU possuem duplicação;
- `hls4ml/artifacts/` e `hls4ml/logs/` estão vazias;
- `hls4ml/models/` contém apenas documentação, enquanto os modelos reais estão na raiz;
- `Vitis AI/` está vazia.

### 20.4 Comparações futuras

Para uma comparação ainda mais rigorosa:

- repetir potência FPGA no protocolo final 10.000 × 100;
- coletar potência do sistema completo CPU/GPU com o mesmo instrumento externo;
- medir CPU/GPU end-to-end a partir de `uint8` para comparar com o cenário end-to-end FPGA;
- realizar sessões softmax/no-softmax intercaladas e pareadas;
- preservar o projeto Vivado atual e todos os relatórios pós-route;
- testar double buffering ou múltiplas inferências em voo em um cenário separado;
- manter sempre separadas as métricas de batch 1 e de capacidade máxima sustentada.

## 21. Arquivos principais para continuidade

### Modelos e decisão softmax

```text
resnet8_cifar10_keras3.h5
resnet8_cifar10_keras3_no_softmax.h5
scripts modelo/remove_softmax.py
hls4ml/docs/DECISAO_SOFTMAX.md
hls4ml/reports/model_audit.json
```

### Benchmarks CPU/GPU

```text
metodologia-benchmark.txt
CPU/resnet8-no-softmax/README.md
CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py
CPU/resnet8-no-softmax/resultados/
GPU/resnet8-no-softmax/README.md
GPU/resnet8-no-softmax/benchmark_no_softmax_gpu.py
GPU/resnet8-no-softmax/resultados/
```

### Fluxo hls4ml

```text
hls4ml/README.md
hls4ml/configs/design_zcu104.json
hls4ml/configs/reuse_plan_zcu104_max288.json
hls4ml/scripts/hls_common.py
hls4ml/scripts/00_check_environment.py
hls4ml/scripts/01_inspect_models.py
hls4ml/scripts/02_prepare_cifar10.py
hls4ml/scripts/03_build_baseline.py
hls4ml/scripts/04_optimize_fifo.py
hls4ml/scripts/05_make_correct_ip.py
```

### Build e FIFO

```text
hls4ml/builds/baseline_q22_12_rf288/
hls4ml/builds/fifo_opt_q22_12_rf288/
hls4ml/builds/fifo_opt_q22_12_rf288/build_return.json
hls4ml/builds/fifo_opt_q22_12_rf288/fifo_depths.json
hls4ml/builds/fifo_opt_q22_12_rf288/reuse_plan_resolved.csv
```

### IP final

```text
hls4ml/ip_repo/resnet8_resource_fifo_opt_v1_1/
hls4ml/ip_exports/xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip
hls4ml/ip_validation/resnet8_resource_fifo_opt_v1_1/
hls4ml/reports/VITIS_HLS_VS_VIVADO_IP_V1_1.md
```

### Hardware e integração

```text
hardware/src/Accel_dma_wrapper.vhd
RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt
RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt
```

### Resultados físicos

```text
hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/
hls4ml/Resultados_ZCU104/resnet8_ip11_final_10k100_2026-08-28/
```

## 22. Conclusão final em 31/08/2026

A ResNet8 sem softmax foi implementada com sucesso na ZCU104 usando hls4ml e ponto fixo Q22.12.

O fluxo passou por:

```text
modelo Keras
→ remoção validada da softmax
→ preparação CIFAR-10
→ conversão hls4ml
→ validação C++
→ síntese HLS
→ profiling e otimização FIFO
→ correção do IP exportado
→ validação Vivado OOC
→ integração PS + DMA + wrapper
→ bitstream
→ execução física
→ validação de acurácia
→ benchmark de desempenho e energia
```

Resultado final comprovado:

```text
Modelo               ResNet8 CIFAR-10 sem softmax
Plataforma           AMD/Xilinx ZCU104
IP                   resnet8_resource_fifo_opt:1.1
Precisão             ap_fixed<22,12,AP_RND_CONV,AP_SAT>
Clock                100 MHz
Acurácia             74,9200%
Latência média       0,849300 ms
p95                  0,862090 ms
FPS efetivo global   899,2063 FPS
Inferências finais   1.000.000
Erros de DMA         nenhum registrado
Divergências/ciclos  zero
```

A implementação está funcional e cientificamente bem documentada em desempenho. A principal pendência é recuperar ou regenerar e preservar os relatórios pós-route do sistema completo atual com IP 1.1, evitando usar como substitutos os números do projeto antigo 2.2.
