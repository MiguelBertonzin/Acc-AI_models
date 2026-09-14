# Relatório consolidado ResNet8 — CPU, GPU, hls4ml e Vitis AI

**Atualização:** 03/09/2026  
**Modelo:** ResNet8 para CIFAR-10  
**Plataforma FPGA:** AMD/Xilinx ZCU104  
**Objetivo:** registrar, em um único documento, tudo que existe no repositório,
o que foi executado, o que foi coletado, a qualidade dos artefatos e quais
comparações são metodologicamente válidas.

## 1. Resultado geral

O repositório contém quatro fluxos executados:

1. TensorFlow/Keras em CPU;
2. TensorFlow/Keras em GPU NVIDIA;
3. hls4ml implementado na ZCU104;
4. Vitis AI executado no DPU da ZCU104.

Os resultados são internamente consistentes e os aceleradores foram validados
funcionalmente. A acurácia pode ser comparada diretamente entre todos os fluxos.
Latência e vazão podem ser comparadas quando se usa a mesma fronteira e, no
Vitis AI, uma thread. Potência, energia e recursos lógicos não permitem um
ranking direto porque os sensores e os domínios físicos são diferentes.

### 1.1 Números principais

| Fluxo comum sem softmax | Acurácia | Latência média | P95 | FPS efetivo global |
|---|---:|---:|---:|---:|
| CPU Intel Core i7-13700 | 74,90% | 0,97597 ms | 1,57039 ms | 975,01 |
| GPU NVIDIA RTX 3050 OEM | 74,89% | 0,69116 ms | 1,12186 ms | 1.223,84 |
| hls4ml ZCU104, serial | 74,92% | 0,84930 ms | 0,86209 ms | 899,21 |
| Vitis AI ZCU104, 1 thread | 73,96% | **0,25255 ms** | **0,25457 ms** | **3.092,49** |

Condições desta tabela: batch 1, modelo sem softmax, 10.000 imagens, 100 ciclos,
1.000.000 de inferências e uma requisição em voo. O FPS é calculado como total
de inferências dividido pelo tempo total, não como média simples de FPS.

### 1.2 Interpretação imediata

- O Vitis AI apresenta a maior vazão e a menor latência.
- A PTQ INT8 do Vitis AI reduz a acurácia em 0,94 ponto percentual.
- O hls4ml mantém praticamente a acurácia float, mas tem menor vazão efetiva.
- A GPU supera a CPU, porém apresentou maior variação entre os ciclos.
- O DPU reproduziu exatamente a referência quantizada do host.
- O hls4ml foi implementado, completamente roteado e fechou timing a 100 MHz.

## 2. Inventário do repositório

```text
resnet8/
├── resnet8_cifar10_keras3.h5
├── resnet8_cifar10_keras3_no_softmax.h5
├── CPU/
│   ├── resnet8-softmax/
│   └── resnet8-no-softmax/
├── GPU/
│   ├── resnet8-softmax/
│   └── resnet8-no-softmax/
├── hls4ml/
│   ├── builds/
│   ├── configs/
│   ├── data/
│   ├── docs/
│   ├── ip_exports/
│   ├── ip_repo/
│   ├── ip_validation/
│   ├── reports/
│   ├── Resultados_ZCU104/
│   ├── scripts/
│   └── tcl/
├── Vitis AI/
│   ├── artifacts/
│   ├── backups/
│   ├── board/
│   ├── config/
│   ├── data/
│   ├── docker/
│   ├── logs/
│   ├── manifests/
│   ├── models/
│   ├── reports/
│   ├── results/
│   └── scripts/
├── hardware/src/
├── scripts modelo/
├── README-METODOLOGIA.md
└── RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt
```

### 2.1 Documentos principais

| Documento | Finalidade |
|---|---|
| [Metodologia](README-METODOLOGIA.md) | contrato experimental e definições gerais |
| [CPU com softmax](CPU/resnet8-softmax/resultados/RESULTADOS.md) | resultados CPU do modelo original |
| [CPU sem softmax](CPU/resnet8-no-softmax/resultados/RESULTADOS.md) | resultados CPU do modelo em logits |
| [GPU com softmax](GPU/resnet8-softmax/resultados/RESULTADOS.md) | resultados GPU do modelo original |
| [GPU sem softmax](GPU/resnet8-no-softmax/resultados/RESULTADOS.md) | resultados GPU do modelo em logits |
| [hls4ml](hls4ml/README.md) | conversão, síntese, IP e resultados da FPGA |
| [hls4ml final na ZCU104](hls4ml/Resultados_ZCU104/resnet8_ip11_final_10k100_2026-08-28/README.md) | campanha final 10k × 100 |
| [hls4ml pós-route](hls4ml/RESULTADOS_VIVADO_IP11_POST_ROUTE.txt) | recursos, timing, route e DRC |
| [Vitis AI](Vitis%20AI/README.md) | fluxo completo de PTQ, DPU e benchmark |

## 3. Modelo e contrato experimental

### 3.1 Modelo

| Propriedade | Valor |
|---|---|
| Arquitetura | ResNet8 |
| Dataset | CIFAR-10 |
| Entrada | NHWC, `32 × 32 × 3` |
| Saída | 10 classes |
| Parâmetros | 78.714 |
| Camadas Keras | 31 |
| Decisão comum | `argmax` |
| Modelo original | saída com softmax |
| Modelo comum aos FPGAs | saída linear, 10 logits |

| Arquivo | SHA-256 |
|---|---|
| `resnet8_cifar10_keras3.h5` | `9cc7cd3ea1c9603501f0c77dc0848b41d0bd136f80cc1d328881ed89cbb9cb1b` |
| `resnet8_cifar10_keras3_no_softmax.h5` | `047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e` |

A remoção da softmax preservou os 47 arrays de pesos, com erro máximo zero.
CPU e GPU confirmaram igualdade do `argmax` nas 10.000 imagens. A reconstrução
da softmax a partir dos logits teve erro máximo de aproximadamente
`2,38 × 10⁻⁷`.

### 3.2 Dataset

| Item | Configuração |
|---|---|
| Conjunto | teste oficial do CIFAR-10 |
| Total | 10.000 imagens |
| Distribuição | 1.000 por classe |
| Entrada armazenada | `uint8`, NHWC |
| Normalização | conversão para `float32` e divisão por 255 |
| Tamanhos | 100, 1.000 e 10.000 |
| Amostragem menor | estratificada e aninhada |
| Semente | 20260825 |

Foi verificado diretamente que as imagens e os rótulos armazenados nos pacotes
hls4ml e Vitis AI são exatamente iguais e estão na mesma ordem. No conjunto
completo, `test_indices` do Vitis AI corresponde a `0..9999`.

### 3.3 Protocolo comum

- batch 1;
- uma imagem por chamada;
- 100 inferências de aquecimento;
- sincronização antes de encerrar o cronômetro;
- nenhuma remoção de outliers;
- 100 ciclos nas campanhas finais normais;
- acurácia calculada apenas com imagens únicas;
- repetições de benchmark não são novas amostras estatísticas de acurácia.

## 4. Significado das métricas

### 4.1 Latência de inferência

A entrada já está preparada para o backend. O intervalo cobre submissão,
execução e sincronização até a saída estar disponível:

- CPU/GPU: chamada do grafo TensorFlow e materialização da saída;
- hls4ml: DMA MM2S → FPGA → DMA S2MM → `wait()`;
- Vitis AI: `execute_async()` + `wait()`, com INT8 já copiado.

Normalização, seleção da próxima imagem e carregamento do dataset ficam fora.

### 4.2 Vazão efetiva batch 1

É o tempo de parede do laço que processa imagens novas. Inclui o trabalho
necessário para alimentar o backend e obter a classe. O valor recomendado é:

```text
FPS_global = total_de_inferências / soma_dos_tempos_dos_ciclos
```

### 4.3 Ponta a ponta

Começa em uma imagem CIFAR-10 `uint8` já na RAM e termina na classe:

```text
uint8 → float32 → /255 → quantização/packing → acelerador → decode → argmax
```

Foi coletado separadamente apenas em hls4ml e Vitis AI.

### 4.4 Saturado

Reutiliza uma entrada já preparada e carregada para medir o teto do caminho
acelerado. Não representa uma aplicação completa.

### 4.5 Potência e energia

```text
P_dinâmica = P_ativa - P_ociosa
E_total    = P_ativa / FPS
E_dinâmica = P_dinâmica / FPS
```

As fórmulas são comuns, mas os sensores não cobrem o mesmo domínio físico.

## 5. Fluxo CPU

### 5.1 Ambiente

| Item | Valor |
|---|---|
| CPU | 13th Gen Intel Core i7-13700 |
| Núcleos / CPUs lógicas | 16 / 24 |
| TensorFlow / Keras | 2.21.0 / 3.13.2 |
| Python | 3.12.7 |
| Execução | `tf.function`, `jit_compile=False`, `/CPU:0` |
| Threads TensorFlow | seleção automática |
| Afinidade | 24 CPUs lógicas |
| Telemetria | turbostat/RAPL, 100 ms |

### 5.2 Resultado principal, N=10.000 e 100 ciclos

| Variante | Acurácia | Média | Mediana | P95 | FPS médio | FPS global |
|---|---:|---:|---:|---:|---:|---:|
| Com softmax | 74,90% | 0,95509 ms | 0,88431 ms | 1,52904 ms | 997,55 | 997,44 |
| Sem softmax | 74,90% | 0,97597 ms | 0,90283 ms | 1,57039 ms | 975,04 | 975,01 |

### 5.3 Potência principal

| Variante | Pacote (W) | Dinâmica (W) | Energia total (mJ/inf) | Dinâmica (mJ/inf) |
|---|---:|---:|---:|---:|
| Com softmax | 57,3368 | 44,5380 | 56,9807 | 44,1491 |
| Sem softmax | 57,4144 | 44,5866 | 58,3599 | 45,2034 |

O baseline RAPL foi aproximadamente 12,8 W. Esses valores representam o pacote
do processador, não a máquina completa.

### 5.4 Consistência

- Os seis arrays têm forma `100 × N`, sem NaN, infinito ou latência negativa.
- Os CSVs possuem 100 passagens por N.
- Média, P95 e FPS foram reproduzidos a partir dos dados brutos.
- No modelo sem softmax, o CV de FPS foi 0,51% e o drift +0,46%.
- No modelo com softmax, o CV foi 1,02% e o drift +2,02%.

O modelo sem softmax apareceu 2,3% mais lento, embora remova uma operação.
As campanhas foram sequenciais, não alternadas, e o baseline térmico era
diferente. Esse resultado não prova que a softmax aumenta o desempenho.

## 6. Fluxo GPU

### 6.1 Ambiente

| Item | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 OEM, 8 GiB |
| Driver | 580.173.02 |
| Limite de potência | 120 W |
| TensorFlow / Keras | 2.21.0 / 3.13.2 |
| Python | 3.12.7 |
| Execução | `tf.function`, sem XLA |
| Telemetria | `nvidia-smi`, 100 ms |
| Potência ociosa | 28,729 W softmax; 29,063 W logits |

### 6.2 Resultado principal, N=10.000 e 100 ciclos

| Variante | Acurácia | Média | Mediana | P95 | FPS médio | FPS global |
|---|---:|---:|---:|---:|---:|---:|
| Com softmax | 74,89% | 0,61097 ms | 0,51044 ms | 1,07685 ms | 1.400,91 | 1.365,32 |
| Sem softmax | 74,89% | 0,69116 ms | 0,51225 ms | 1,12186 ms | 1.268,49 | 1.223,84 |

### 6.3 Potência principal

| Variante | Total GPU (W) | Dinâmica (W) | Energia total (mJ/inf) | Dinâmica (mJ/inf) |
|---|---:|---:|---:|---:|
| Com softmax | 35,9181 | 7,1895 | 26,1127 | 5,0710 |
| Sem softmax | 35,1824 | 6,1194 | 28,4617 | 4,7143 |

É potência da placa NVIDIA reportada pelo driver, não potência do computador.
A precisão declarada do sensor é da ordem de ±5 W, especialmente relevante em
janelas curtas.

### 6.4 Consistência e instabilidade

- Arrays e CSVs têm dimensões e contagens corretas.
- Não existem latências inválidas.
- Softmax e logits têm 100% de igualdade de classes na GPU.
- A GPU classificou uma imagem diferente da CPU: 7.489 contra 7.490 acertos.
- O CV entre ciclos foi 14,26% com softmax e 18,42% sem softmax.
- O drift de FPS foi −2,71% e −3,77%, respectivamente.
- O FPS global é 2,6%–3,7% menor que a média dos FPS por ciclo.

A variação é alta e os resultados por N não são monotônicos. Para tabelas do
TCC, usar o FPS global de N=10.000/100 ciclos e apresentar a GPU com ressalva.
Não usar essas campanhas para estimar o custo isolado da softmax.

## 7. Fluxo hls4ml na ZCU104

### 7.1 Conversão e implementação

| Propriedade | Valor |
|---|---|
| Modelo de origem | mesmo Keras sem softmax, SHA `047c...761e` |
| hls4ml | 1.3.0 |
| Backend | Vitis |
| Ferramentas AMD/Xilinx | Vitis HLS/Vivado 2024.2 |
| Dispositivo | XCZU7EV-FFVC1156-2-E |
| Estratégia | Resource |
| IO | `io_stream` |
| Convolução | LineBuffer |
| Precisão | `ap_fixed<22,12,AP_RND_CONV,AP_SAT>` |
| Reuse Factor máximo | 288 |
| FIFO optimization | habilitada |
| Clock | 100 MHz |
| Interface da placa | AXI DMA |

O fluxo realizou auditoria do modelo, preparação do CIFAR-10, geração baseline,
otimização de FIFO, correção/empacotamento do IP, síntese OOC, integração no
Vivado, implementação, geração de bitstream e validação na placa.

### 7.2 Validação

| Item | Resultado |
|---|---:|
| Imagens | 10.000 |
| Acertos | 7.492 |
| Acurácia | 74,92% |
| IC95% Wilson | 74,0609%–75,7599% |
| Divergências durante os ciclos | 0 |

### 7.3 Campanha final 10k × 100

| Métrica | Resultado |
|---|---:|
| Inferências | 1.000.000 |
| Latência média | 0,849300 ms |
| Mediana | 0,849360 ms |
| Desvio-padrão | 0,015460 ms |
| P95 / P99 | 0,862090 / 0,875650 ms |
| FPS global | 899,2063 |
| Taxa equivalente pela latência | 1.177,4412 FPS |
| CV entre ciclos | 0,1099% |
| Drift | +0,1496% |

### 7.4 Campanha intermediária: três cenários

Usou 5.000 imagens × 20 ciclos para inferência e ponta a ponta; saturado usou
oito janelas de 10 s.

| Métrica | Inferência | Ponta a ponta | Saturado serial |
|---|---:|---:|---:|
| Latência média | 0,8451 ms | 1,5977 ms | — |
| P95 | 0,8592 ms | 1,6155 ms | — |
| FPS | 901,08 | 620,70 | 1.200,94 |
| Potência ociosa | 10,6094 W | 10,6270 W | 10,6170 W |
| Potência ativa | 12,2283 W | 11,8088 W | 12,6580 W |
| Potência dinâmica | 1,6189 W | 1,1818 W | 2,0410 W |
| Energia total | 13,4994 mJ | 19,0992 mJ | 10,7266 mJ |
| Energia dinâmica | 1,7871 mJ | 1,9114 mJ | 1,7295 mJ |

O sensor usado foi o PMBus `12V_power`, com aquisição final a cada 1 s.

### 7.5 Recursos pós-route

| Recurso | IP ResNet8 | Sistema completo |
|---|---:|---:|
| CLB LUTs | 162.352 (70,47%) | 178.658 (77,54%) |
| Registradores | 179.101 (38,87%) | 197.484 (42,86%) |
| BRAM tiles | 74,5 (23,88%) | 85,5 (27,40%) |
| URAM | 11 (11,46%) | 11 (11,46%) |
| DSP48E2 | 1.174 (67,94%) | 1.174 (67,94%) |

O sistema de integração acrescentou 16.306 LUTs, 18.383 registradores e 11
BRAM tiles. DSPs e URAMs pertencem integralmente ao acelerador ResNet8.

### 7.6 Timing, route e DRC

| Verificação | Resultado |
|---|---:|
| Clock | 100 MHz / 10 ns |
| WNS / TNS | +0,122 ns / 0 ns |
| WHS / THS | +0,009 ns / 0 ns |
| Redes roteáveis completas | 368.791/368.791 |
| Erros de roteamento | 0 |
| DRC | 3.008 warnings; zero erros críticos reportados |

Os warnings são principalmente recomendações de pipeline de DSP e não impediram
timing, roteamento ou funcionamento na placa.

### 7.7 Consistência

- A campanha intermediária deu 0,8451 ms/901,08 FPS.
- A campanha final deu 0,8493 ms/899,21 FPS.
- Diferenças: +0,49% em latência e −0,21% em FPS.
- A acurácia permaneceu exatamente 74,92%.
- Todos os 19 itens do manifesto da coleta final passaram no SHA-256.
- Um milhão de latências são finitas e positivas.

## 8. Fluxo Vitis AI na ZCU104

### 8.1 Host: conversão, PTQ e compilação

| Etapa | Resultado |
|---|---|
| Keras 3 → Keras 2.12 | 32/32 argmax iguais |
| Erro médio / máximo | `3,525 × 10⁻⁶` / `1,287 × 10⁻⁵` |
| Float Keras 2 | 74,90% |
| Calibração | 1.000 imagens, 100 por classe |
| Quantização | PTQ INT8 `pof2s` |
| Host quantizado | 73,96% |
| Concordância INT8 × float | 93,75% |
| Compilador | sucesso, retorno 0 |
| Alvo | `DPUCZDX8G_ISA1_B4096` |

XModel:

| Propriedade | Valor |
|---|---|
| Tamanho | 249.251 bytes |
| SHA-256 | `bfacc85783958ff0d734e5c7e44f126a6fd4af9b6b1f20e4d9b2f7ae96fa7802` |
| Subgrafos | USER entrada, DPU com 36 operações e CPU dequant |
| Entrada | INT8 `[1,32,32,3]`, `fix_point=6` |
| Saída | INT8 `[1,1,1,10]`, `fix_point=2` |

### 8.2 Ambiente da placa

| Item | Valor |
|---|---|
| CPU/OS | AArch64, Linux Xilinx 2022.2 |
| Python / NumPy | 3.9.9 / 1.21.2 |
| VART | 3.0 |
| XRT | 2.14 / 2022.2 |
| DPU | 2 núcleos |
| Clock DPU/XRT | 300 MHz |
| Fingerprint | `0x101000056010407` |
| CPUs ARM | 4 |

Apesar de o compilador ser Vitis AI 3.5 e o runtime da placa ser VART 3.0, o
XModel foi aceito e sua compatibilidade foi demonstrada para este experimento.

### 8.3 Validação DPU

| Verificação | Resultado |
|---|---:|
| Imagens | 10.000 |
| Acertos / acurácia | 7.396 / 73,96% |
| Concordância DPU × host INT8 | 10.000/10.000 |
| MAE / RMSE / máximo dos logits | 0 / 0 / 0 |
| Faixa INT8 observada | −109 a 84 |
| Saturação em −128 ou +127 | nenhuma |

### 8.4 Protocolo

- batch 1;
- 1, 2, 3 e 4 threads;
- runner VART e buffers privados por worker;
- threads fixadas, sem erros de afinidade;
- 100 warm-ups por runner;
- N=100, 1.000 e 10.000, 100 ciclos;
- latência saturada: 30 blocos × 2.000;
- vazão saturada: 20 janelas × 10 s;
- nenhuma remoção de outliers.

### 8.5 Inferência, N=10.000

| Threads | Média | Mediana | P95 | P99 | FPS por ciclo |
|---:|---:|---:|---:|---:|---:|
| 1 | **0,252552 ms** | 0,252100 | 0,254570 | 0,264000 | 3.092,50 |
| 2 | 0,264278 ms | 0,260120 | 0,279330 | 0,290420 | 5.889,36 |
| 3 | 0,340574 ms | 0,351310 | 0,397550 | 0,406090 | **6.707,23** |
| 4 | 0,519735 ms | 0,525890 | 0,558600 | 0,699660 | 6.545,64 |

Uma thread minimiza a latência; três maximizam a vazão normal. O FPS global
recalculado para uma thread é 3.092,489, praticamente igual à média.

### 8.6 Ponta a ponta, N=10.000

| Threads | Média | Mediana | P95 | P99 | FPS por ciclo |
|---:|---:|---:|---:|---:|---:|
| 1 | **0,681568 ms** | 0,679280 | 0,693230 | 0,709620 | 1.444,74 |
| 2 | 0,954863 ms | 0,934430 | 1,050910 | 1,071540 | 2.070,73 |
| 3 | 1,429096 ms | 1,423760 | 1,593330 | 1,628230 | **2.081,76** |
| 4 | 1,923526 ms | 1,900700 | 2,696090 | 3,156770 | 2.054,89 |

### 8.7 Saturado

| Threads | Média | P95 | P99 | FPS | IC95% FPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,246675 ms | 0,248450 | 0,257490 | 3.991,11 | 3.988,56–3.993,70 |
| 2 | 0,261533 ms | 0,272530 | 0,281540 | 7.517,01 | 7.509,11–7.524,95 |
| 3 | 0,273826 ms | 0,287460 | 0,301660 | 10.725,94 | 10.694,83–10.753,21 |
| 4 | 0,307111 ms | 0,340680 | 0,369240 | **12.814,81** | 12.734,74–12.893,41 |

Quatro threads são superiores somente quando a entrada é reutilizada e o host
faz trabalho mínimo.

### 8.8 Potência e energia

O sensor foi o INA226 em
`/sys/class/hwmon/hwmon0/power1_input`, mas sem rótulo. A potência ociosa
média em 120 s foi 14,4788595 W.

| Cenário | T | Ativa (W) | Dinâmica (W) | FPS | Total (mJ) | Dinâmica (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| Inferência | 1 | 15,2359 | 0,7571 | 3.134,29 | 4,8611 | 0,2415 |
| Inferência | 2 | 15,9118 | 1,4329 | 5.880,78 | 2,7064 | 0,2437 |
| Inferência | 3 | 16,2043 | 1,7254 | 6.743,45 | 2,4040 | 0,2559 |
| Inferência | 4 | 16,1586 | 1,6797 | 6.566,47 | 2,4610 | 0,2558 |
| Ponta a ponta | 1 | 14,9590 | 0,4802 | 1.448,53 | 10,3270 | 0,3315 |
| Ponta a ponta | 2 | 15,2439 | 0,7651 | 2.069,51 | 7,3667 | 0,3697 |
| Ponta a ponta | 3 | 15,2784 | 0,7995 | 2.049,41 | 7,4558 | 0,3902 |
| Ponta a ponta | 4 | 15,3047 | 0,8258 | 2.036,52 | 7,5154 | 0,4055 |
| Saturado | 1 | 15,3896 | 0,9107 | 3.981,75 | 3,8650 | 0,2287 |
| Saturado | 2 | 16,1958 | 1,7169 | 7.483,46 | 2,1644 | 0,2295 |
| Saturado | 3 | 16,9033 | 2,4244 | 10.692,85 | 1,5811 | 0,2268 |
| Saturado | 4 | 17,2803 | 2,8014 | 12.554,48 | **1,3767** | **0,2232** |

### 8.9 Overhead da telemetria

Foram feitos seis pares ON/OFF por cenário/thread. A maioria dos IC95% inclui
zero. A exceção relevante foi saturado com três threads, com overhead médio de
2,30% e IC95% de 0,52% a 4,31%. Desempenho e potência foram, portanto,
coletados em campanhas separadas.

### 8.10 Integridade

| Auditoria | Resultado |
|---|---:|
| Manifesto do host | 45/45 |
| Deploy | 7/7 |
| Validação | 8/8 |
| Manifesto visível de desempenho | 102/102 |
| JSON / NPY / NPZ analisados | 48 / 35 / 2 |
| NaN ou infinito | nenhum |
| Latências normais / saturadas | 8.880.000 / 240.000 |
| Inferências em janelas saturadas | 7.010.073 |
| Erros ao recalcular estatísticas | 0 |
| Erros de afinidade | 0 |

Pacote canônico:

```text
Vitis AI/resnet8_vitis_ai_COMPLETE_2026-09-02.tar.gz
SHA-256: 604a8f397bd451172ce6b3aedd12586a1660a08f077de1c6271f4db0ef71ceee
```

## 9. Validação cruzada de acurácia

### 9.1 Acurácia e intervalos

| Backend | Corretas | Acurácia | IC95% Wilson |
|---|---:|---:|---:|
| CPU float | 7.490 | 74,90% | 74,04%–75,74% |
| GPU float | 7.489 | 74,89% | 74,03%–75,73% |
| hls4ml | 7.492 | 74,92% | 74,06%–75,76% |
| Vitis AI INT8 | 7.396 | 73,96% | 73,09%–74,81% |

### 9.2 Predições pareadas

| Comparação | Iguais | Concordância | Primeiro apenas correto | Segundo apenas correto |
|---|---:|---:|---:|---:|
| hls4ml × float | 9.907 | 99,07% | 34 | 32 |
| INT8 × float | 9.375 | 93,75% | 182 | 276 |
| hls4ml × Vitis AI | 9.348 | 93,48% | 289 | 193 |
| DPU × host INT8 | 10.000 | 100,00% | 0 | 0 |

O hls4ml é funcionalmente quase idêntico ao float. Vitis AI difere por causa da
PTQ INT8; a diferença pareada hls4ml × Vitis tem McNemar exato
`p ≈ 1,421065 × 10⁻⁵`.

## 10. Comparação de desempenho

### 10.1 Relações contra o Vitis AI de uma thread

| Referência | Razão de latência | Razão de vazão |
|---|---:|---:|
| CPU sem softmax | Vitis 3,86× menor | Vitis 3,17× maior |
| GPU sem softmax | Vitis 2,74× menor | Vitis 2,53× maior |
| hls4ml serial | Vitis 3,36× menor | Vitis 3,44× maior |

Essas razões são de plataforma/sistema, não de kernel puro. CPU pode usar
paralelismo interno do TensorFlow; GPU usa o dispositivo inteiro; hls4ml inclui
DMA; Vitis inclui o runtime VART.

### 10.2 hls4ml × Vitis AI

| Métrica | hls4ml | Vitis, 1 thread | Relação |
|---|---:|---:|---:|
| Acurácia | 74,92% | 73,96% | hls4ml +0,96 p.p. |
| Latência inference-only | 0,8493 ms | 0,2526 ms | Vitis 3,36× menor |
| FPS efetivo | 899,21 | 3.092,49 | Vitis 3,44× maior |
| Latência ponta a ponta | 1,5977 ms | 0,6816 ms | Vitis 2,34× menor |
| FPS ponta a ponta | 620,70 | 1.444,74 | Vitis 2,33× maior |
| FPS saturado serial | 1.200,94 | 3.991,11 | Vitis 3,32× maior |

Ponta a ponta hls4ml usa 5.000 × 20, enquanto Vitis usa 10.000 × 100.
Saturado hls4ml usa oito janelas; Vitis usa vinte. Os conceitos são próximos,
mas o tamanho da campanha precisa acompanhar a tabela.

## 11. Matriz de comparabilidade

| Métrica | Grau | Regra |
|---|---|---|
| Acurácia top-1 | direta | mesmas imagens, rótulos e `argmax` |
| Predição por imagem | direta | comparar pelo índice |
| Latência inference-only | boa, sistêmica | entrada preparada e saída sincronizada |
| FPS batch 1 | boa | usar FPS global; Vitis com 1 thread |
| Ponta a ponta | hls4ml × Vitis | CPU/GPU não coletaram |
| Saturado serial | hls4ml × Vitis 1 thread | declarar janelas diferentes |
| Vitis 2–4 threads | apenas escalabilidade interna | dois DPUs e concorrência |
| Potência total | não direta | sensores medem domínios distintos |
| Energia dinâmica | indicativa | baselines e rails distintos |
| LUT/FF/DSP/BRAM | apenas hls4ml | DPU é overlay fixo sem relatório equivalente |
| Softmax × logits, acurácia | direta | equivalência comprovada |
| Softmax × logits, desempenho | não causal | campanhas não pareadas |

## 12. Por que potência não deve virar ranking

| Plataforma | Sensor | Domínio |
|---|---|---|
| CPU | RAPL/turbostat | pacote do processador |
| GPU | `nvidia-smi` | placa NVIDIA |
| hls4ml | PMBus `12V_power` | alimentação de 12 V da ZCU104 |
| Vitis AI | INA226 sem rótulo | rail não identificado |

Também diferem baseline, intervalo de aquisição, duração das janelas, sistema
operacional e overlay. É válido dizer “dentro do Vitis, quatro threads saturadas
reduzem energia por imagem”; não é rigoroso dizer “Vitis gasta X vezes menos que
GPU” com esses sensores.

## 13. Problemas, ressalvas e pendências documentais

### 13.1 Documentação antiga

O `README-METODOLOGIA.md`, consolidado em 31/08, ainda descreve o Vitis AI
como roteiro não executado. Este documento de 03/09 e o
`Vitis AI/README.md` substituem essa informação histórica.

### 13.2 FPS médio versus global

Os READMEs CPU/GPU exibem a média aritmética dos FPS por ciclo. Em CPU a
diferença é pequena; em GPU chega a 3,7%. Para comparação consolidada, usar
`throughput_effective_fps`.

### 13.3 Relógio da placa

Há timestamps incompatíveis com a data real:

- hls4ml intermediário registra `2025-05-04`, mas a coleta real é 28/08/2026;
- validação Vitis registra `2021-11-19`, mas a campanha é 01–02/09/2026.

Isso indica RTC não sincronizado. Não altera arrays ou estatísticas, mas deve ser
explicado na proveniência.

### 13.4 JSON não estrito na CPU

Os metadados CPU usam o literal `NaN` para frequência inválida. Python aceita,
mas JSON estrito não. Em uma publicação ou pipeline externo, substituir por
`null` acompanhado da justificativa já presente nos metadados.

### 13.5 Manifesto Vitis

`SHA256SUMS_COMPLETE.txt` menciona o transitório
`power_run_2026-09-02.pid`, não empacotado. Todos os demais itens presentes
foram validados. O log textual de potência está truncado, mas CSVs e resumos
estão completos. O arquivo de temperatura final não possui valor numérico.

### 13.6 Benchmark C/C++

O roteiro Vitis mencionou uma alternativa C/C++, mas a coleta preservada usa
Python/VART. Não existe benchmark C/C++ completo no diretório.

## 14. Veredito por fluxo

| Fluxo | Funcional | Dados brutos | Estabilidade | Comparação |
|---|---|---|---|---|
| CPU | aprovado | completos | boa | válida no cenário comum |
| GPU | aprovado | completos | moderada/baixa | válida com ressalva |
| hls4ml | aprovado | completos e com hashes | excelente | válida nas fronteiras declaradas |
| Vitis AI | aprovado | completos no pacote canônico | excelente | usar 1 thread para comparação |

## 15. Recomendações para as tabelas do TCC

1. Usar o modelo sem softmax como variante comum.
2. Usar N=10.000, 100 ciclos e batch 1 na tabela principal.
3. Publicar `throughput_effective_fps`, não apenas média dos FPS.
4. Usar Vitis AI com uma thread na comparação CPU/GPU/hls4ml.
5. Apresentar 2–4 threads como estudo separado de escalabilidade.
6. Separar as tabelas de inference-only, ponta a ponta e saturado.
7. Apresentar acurácia com número de acertos e IC95%.
8. Explicar que a perda Vitis vem da PTQ, não do DPU.
9. Não ranquear energia entre plataformas com os sensores atuais.
10. Colocar recursos/timing hls4ml em uma seção de implementação, não na tabela
    comum de software/aceleradores.
11. Informar as datas reais e a falha de RTC da placa.
12. Manter os máximos de latência; nenhum outlier foi removido.

## 16. O que está comprovado

- Os modelos com e sem softmax preservam a mesma decisão.
- CPU e GPU executaram o mesmo modelo e dataset.
- hls4ml e Vitis usaram exatamente as mesmas imagens/rótulos.
- hls4ml foi gerado a partir do mesmo modelo sem softmax.
- hls4ml reproduz quase integralmente a referência float.
- O XModel Vitis foi compilado, carregado e executado no DPU real.
- DPU e referência INT8 coincidem em todas as 10.000 amostras.
- Os resultados de desempenho podem ser recalculados dos dados brutos.
- O hls4ml fecha timing e route no dispositivo alvo.
- O pacote completo Vitis possui hash externo válido.

## 17. O que não está comprovado

- Que remover softmax melhora o tempo nos hosts: as campanhas não são pareadas.
- Que GPU seria estável sob repetição em outro momento: houve alto CV.
- Que a energia dos quatro dispositivos mede a mesma fronteira elétrica.
- Que o rail Vitis corresponde à placa inteira ou apenas ao DPU.
- Que Vitis 3.5/VART 3.0 funciona para qualquer XModel; apenas este foi validado.
- Que o teto saturado representa uma aplicação real.
- Que recursos hls4ml podem ser comparados aos recursos internos do overlay DPU.

## 18. Conclusão final

O conjunto de resultados é adequado para um TCC desde que as fronteiras sejam
mantidas explícitas. A cadeia funcional está bem demonstrada: modelo float,
remoção de softmax, duas quantizações diferentes, implementação hls4ml e
execução INT8 no DPU.

O hls4ml prioriza fidelidade ao modelo float e alcança 74,92% de acurácia.
O Vitis AI prioriza desempenho e alcança 3.092,49 FPS no modo comparável de uma
thread, com 73,96% de acurácia. O DPU não adiciona erro ao modelo quantizado.

Para operação:

- **menor latência Vitis:** uma thread;
- **maior vazão Vitis de aplicação:** três threads;
- **maior teto saturado Vitis:** quatro threads;
- **melhor preservação da acurácia em FPGA:** hls4ml;
- **resultado host mais cauteloso:** GPU, pela variabilidade temporal;
- **comparação energética:** somente qualitativa entre plataformas.

Este README é a consolidação de referência em 03/09/2026. Os READMEs específicos
continuam sendo as fontes detalhadas de cada fluxo e os CSV/NPY/NPZ/JSON são as
fontes numéricas primárias.
