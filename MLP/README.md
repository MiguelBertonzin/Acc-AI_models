# MLP Iris — dados, benchmarks e IP FPGA

Este repositório reúne o modelo MLP treinado para o conjunto Iris, os benchmarks em CPU e GPU e duas implementações para a ZCU104. A especificação normativa de coleta é [`METODOLOGIA_BENCHMARK_MLP_IRIS.md`](METODOLOGIA_BENCHMARK_MLP_IRIS.md); o arquivo `metodologia-benchmark.txt` é apenas uma referência histórica de outro modelo.

## Estado atual

| Etapa | Estado | Resultado principal |
|---|---:|---|
| Modelo Keras e pré-processamento | Concluído | MLP 4→8→8→3, 139 parâmetros |
| Validação funcional | Concluída | 29/30 amostras corretas, 96,67% de acurácia |
| Benchmark CPU | Concluído | Campanhas de 30 mil e 100 mil inferências, com 5 repetições e RAPL |
| Benchmark GPU | Concluído | 5 campanhas de 30 mil + 5 de 100 mil, com telemetria NVIDIA |
| Quantização e compilação Vitis AI | Concluída | INT8 com 96,67% e XModel para DPUCZDX8G B4096 |
| Conversão e validação hls4ml | Concluída | C simulation e C/RTL co-simulation aprovadas |
| Síntese Vitis HLS | Concluída | Latência de 5 ciclos, II=1, clock estimado de 6,586 ns |
| Síntese e implementação Vivado OOC | Concluída | Timing atendido a 100 MHz e recursos pós-route coletados |
| Integração no block design da ZCU104 | Pendente | Necessário wrapper/interface e integração com PS/AXI |
| Medição física na placa | Pendente | Ainda não há latência, throughput ou potência medidos na ZCU104 |

## Modelo e conjunto de dados

- Dataset: Iris, com 150 amostras, 4 atributos e 3 classes.
- Divisão: 120 amostras para treino e 30 para teste.
- Configuração da divisão: `test_size=0.2`, `random_state=42` e estratificação.
- Arquitetura: entrada com 4 atributos, duas camadas densas de 8 neurônios com ReLU e saída de 3 classes com softmax.
- Total: 139 parâmetros treináveis.
- Modelo original: `float32`.

Arquivos principais:

- [`iris_mlp_clean.h5`](iris_mlp_clean.h5): modelo Keras treinado.
- [`iris_scaler.joblib`](iris_scaler.joblib): normalizador usado antes da inferência.
- [`hls4ml/docs/descricao_completa_MLP_Iris_FPGA.txt`](hls4ml/docs/descricao_completa_MLP_Iris_FPGA.txt): descrição do modelo e do fluxo proposto.

## Validação funcional de referência

| Métrica | Resultado |
|---|---:|
| Acurácia no conjunto de teste | 96,67% (29/30) |
| Intervalo de Wilson de 95% | 83,33% a 99,41% |
| Macro F1 | 0,966583 |
| MCC | 0,951587 |
| ROC AUC macro | 0,996667 |
| Concordância de classes CPU × GPU | 100% |
| Maior diferença de probabilidade CPU × GPU | 1,192 × 10⁻⁷ |
| Determinismo GPU | 100% |

## Benchmark em CPU

Plataforma: Intel Core i7-13700, 16 núcleos físicos e 24 processadores lógicos, TensorFlow 2.21 com oneDNN. As inferências foram feitas com batch 1, execução serial e síncrona, após 200 warm-ups. A potência do pacote foi coletada por RAPL, com baseline de 5 s e telemetria a cada 100 ms.

Os resultados consolidados abaixo correspondem a cinco campanhas independentes para cada tamanho:

| Métrica | 30 mil inferências | 100 mil inferências |
|---|---:|---:|
| Latência somente da inferência | 0,112886 ms | 0,112399 ms |
| IC 95% da latência | 0,110670–0,115102 ms | 0,107717–0,117082 ms |
| Latência efetiva ponta a ponta | 0,137473 ms | 0,137069 ms |
| Throughput | 7.275,55 inf/s | 7.301,88 inf/s |
| Potência média do pacote | 53,444 W | 55,245 W |
| Energia total do pacote | 7,685 mJ/inf | 7,927 mJ/inf |
| Energia dinâmica do pacote | 5,974 mJ/inf | 6,121 mJ/inf |
| Temperatura média | 68,8 °C | 73,6 °C |
| Pico médio de temperatura | 74,6 °C | 78,8 °C |

A campanha de **100 mil inferências** é a referência recomendada por cobrir uma janela de medição maior. Não foi encontrada diferença estatisticamente significativa entre as campanhas de 30 mil e 100 mil para energia, potência, latência de inferência, latência ponta a ponta ou throughput.

Relatórios e dados:

- [`CPU/README.md`](CPU/README.md): índice e metodologia do benchmark.
- [`CPU/VALIDACAO_ENERGIA_RAPL.md`](CPU/VALIDACAO_ENERGIA_RAPL.md): validação das medições de energia.
- [`CPU/COMPARACAO_CPU_30000_100000_E_GPU.md`](CPU/COMPARACAO_CPU_30000_100000_E_GPU.md): comparação entre campanhas e com a GPU.
- [`CPU/resultados_30000_rapl/`](CPU/resultados_30000_rapl/), [`CPU/resultados_100000_rapl/`](CPU/resultados_100000_rapl/) e [`CPU/validacao_rapl/`](CPU/validacao_rapl/): dados brutos, telemetria e repetições independentes.

## Benchmark em GPU

Plataforma: NVIDIA GeForce RTX 3050 OEM de 8 GB, driver 580.173.02, TensorFlow 2.21, CUDA 12.5.1 e cuDNN 9. Foram concluídas **cinco campanhas independentes** por tamanho, batch 1 serial e síncrono, 200 warm-ups, baseline de 5 s, telemetria de 100 ms e nenhum outlier removido.

| Métrica (média entre campanhas) | 30 mil × 5 | 100 mil × 5 |
|---|---:|---:|
| Inference-only | 0,176134 ms | 0,168663 ms |
| IC95% t | 0,168712–0,183557 ms | 0,157735–0,179591 ms |
| End-to-end efetiva | 0,264565 ms | 0,255899 ms |
| Throughput | 3.784,46 inf/s | 3.911,81 inf/s |
| Energia total da placa GPU | 7,8758 mJ/inf | 7,5211 mJ/inf |
| Acurácia em todas as campanhas | 29/30 | 29/30 |

A unidade experimental desses intervalos é a campanha (`n=5`). A energia usa o sensor de potência total da placa via `nvidia-smi`; a energia dinâmica por subtração de baseline continua sendo de baixa confiança para esta carga curta.

Relatórios e dados:

- [`GPU/replicas/RESUMO_5_CAMPANHAS.md`](GPU/replicas/RESUMO_5_CAMPANHAS.md): consolidação estatística das dez campanhas.
- [`GPU/replicas/campanhas_gpu.csv`](GPU/replicas/campanhas_gpu.csv) e [`GPU/replicas/resumo_5_campanhas.json`](GPU/replicas/resumo_5_campanhas.json): dados por campanha e agregados.
- [`GPU/resultados/`](GPU/resultados/) e [`GPU/resultados_100000/`](GPU/resultados_100000/): réplicas históricas `rep01`.
- [`GPU/replicas/30000/`](GPU/replicas/30000/) e [`GPU/replicas/100000/`](GPU/replicas/100000/): `rep02` a `rep05`.

## Comparação CPU × GPU

Para esta rede muito pequena e com batch 1, a CPU apresentou aproximadamente o dobro do throughput da GPU. Isso é coerente com a sobrecarga fixa de lançamento e sincronização das operações na GPU, que domina o tempo de uma inferência tão curta.

As medições de energia não são diretamente equivalentes:

- CPU: energia do pacote medida por RAPL.
- GPU: potência total da placa reportada pelo sensor NVIDIA.

Portanto, os números permitem caracterizar cada plataforma no protocolo utilizado, mas não representam uma comparação elétrica perfeitamente homogênea de todo o sistema.

## XModel com Vitis AI

Foi gerado um segundo acelerador pelo fluxo Vitis AI 3.5, independente do IP hls4ml. O AI Quantizer utilizou PTQ INT8 `pof2s` com todas as 120 amostras de treino, sem usar as 30 amostras de teste. O AI Compiler produziu um XModel para a DPU `DPUCZDX8G_ISA1_B4096` da ZCU104.

| Métrica | Resultado |
|---|---:|
| Acurácia float | 96,67% (29/30) |
| Acurácia INT8 simulada | 96,67% (29/30) |
| Concordância INT8 × float | 100% |
| Subgrafos DPU | 1 |
| Entrada DPU | `int8[1,4]`, `fix_point=5` |
| Saída DPU | `int8[1,3]`, `fix_point=3` |
| Tamanho do XModel | 109.221 bytes |

O softmax foi retirado do grafo DPU, mantendo `argmax(logits)` equivalente ao classificador original. Ele pode ser aplicado no ARM quando forem necessárias probabilidades. O fluxo, os dados de calibração, os relatórios e o pacote para a placa estão documentados em [`Vitis AI/README.md`](<Vitis AI/README.md>).
O roteiro interativo para implantação e coleta física está em [`Vitis AI/README_CHATGPT_WEB_ZCU104_XMODEL.md`](<Vitis AI/README_CHATGPT_WEB_ZCU104_XMODEL.md>).

## IP gerado com hls4ml

Configuração consolidada:

| Parâmetro | Valor |
|---|---|
| Dispositivo alvo | ZCU104 — `xczu7ev-ffvc1156-2-e` |
| Ferramenta | Vitis HLS 2024.2 |
| hls4ml | 1.3.0 |
| Clock solicitado | 100 MHz (10 ns) |
| Precisão padrão | `ap_fixed<16,6>` |
| QKeras | Não utilizado |
| Reuse factor | 1 |
| Estratégia | `Latency` |
| I/O | `io_parallel` |
| Softmax | Implementado no hardware |
| Nome e versão do IP | `mlp_iris` v1.0.0 |

As tabelas auxiliares do softmax usam `ap_fixed<18,8>`. A interface gerada é paralela e usa controle `ap_ctrl_hs`: entrada `features` de 64 bits, correspondente a quatro valores de 16 bits, e três saídas de 16 bits com sinais `ap_vld`. Ela ainda não é uma interface AXI completa.

### Validação HLS

| Verificação | Resultado |
|---|---:|
| Acurácia C++ HLS | 96,67% (29/30) |
| Concordância de classes HLS × Keras | 100% |
| Erro absoluto máximo nas saídas | 0,1776121259 |
| Erro absoluto médio nas saídas | 0,0167446900 |
| C simulation | Aprovada |
| C/RTL co-simulation Verilog | Aprovada |
| Igualdade C simulation × RTL | Exata para os vetores testados |

O erro numérico nas probabilidades é causado principalmente pela aproximação de softmax em ponto fixo; ele não alterou a classe prevista no conjunto de teste.

### Resultado do Vitis HLS

| Métrica | Resultado |
|---|---:|
| Período solicitado | 10,000 ns |
| Período estimado | 6,586 ns |
| Frequência estimada | 151,85 MHz |
| Latência | 5 ciclos / 50 ns a 100 MHz |
| Intervalo de iniciação | 1 ciclo |
| LUT | 4.058 |
| FF | 413 |
| DSP | 108 |
| BRAM18K | 3 |
| URAM | 0 |

## Vitis HLS × Vivado

O Vivado foi executado em modo out-of-context, primeiro após síntese e depois após place-and-route. Esse resultado é mais fiel ao uso real da lógica do IP que a estimativa inicial do HLS, embora ainda não inclua o sistema completo da ZCU104.

| Recurso | Vitis HLS | Vivado pós-síntese OOC | Vivado pós-route OOC |
|---|---:|---:|---:|
| LUT | 4.058 | 2.098 | 1.944 |
| Registradores/FF | 413 | 377 | 377 |
| DSP | 108 | 108 | 108 |
| BRAM18 | 3 | 3 | 3 |
| URAM | 0 | 0 | 0 |

No pós-route, o IP ocupa aproximadamente:

- 1.944 LUTs, ou 0,84% do dispositivo;
- 377 registradores, ou 0,08%;
- 108 DSPs, ou 6,25%;
- 3 blocos RAMB18, ou 0,48%;
- 330 CLBs, ou 1,15%.

Em relação à estimativa HLS, o pós-route apresenta 2.114 LUTs a menos (`−52,09%`) e 36 registradores a menos (`−8,72%`), mantendo DSP e BRAM iguais.

### Timing do Vivado a 100 MHz

| Métrica | Resultado |
|---|---:|
| WNS | +2,886 ns |
| TNS | 0 ns |
| WHS | +0,064 ns |
| Caminho crítico aproximado | 7,114 ns |
| Frequência equivalente aproximada | 140,6 MHz |
| Restrições atendidas | Sim |
| Falhas de roteamento | Nenhuma |

Há avisos consultivos sobre pipelining interno de DSPs e um sinal sem carga, mas nenhum deles impediu a implementação ou o fechamento de timing. Como a análise é OOC, os ports de fronteira não possuem atrasos externos de entrada/saída e existe um aviso relacionado a `HD.CLK_SRC`. O timing deve ser revalidado após a integração no block design completo.

Relatórios e artefatos:

- [`hls4ml/RESULTADOS_IP.md`](hls4ml/RESULTADOS_IP.md): resultado completo do IP.
- [`hls4ml/COMPARACAO_RECURSOS_VITIS_VIVADO.md`](hls4ml/COMPARACAO_RECURSOS_VITIS_VIVADO.md): comparação detalhada de recursos e timing.
- [`hls4ml/README_CHATGPT_WEB_ZCU104_IP.md`](hls4ml/README_CHATGPT_WEB_ZCU104_IP.md): roteiro interativo para Vivado, block design, PYNQ e medições na ZCU104.
- [`hls4ml/README.md`](hls4ml/README.md): instruções do fluxo hls4ml.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/): projeto HLS gerado.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip): pacote do IP para o catálogo do Vivado.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/reports/vivado/`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/reports/vivado/): utilização, timing, DRC e checkpoints do Vivado.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/reports/resource_comparison.json`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/reports/resource_comparison.json): comparação em formato estruturado.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/configuration_manifest.json`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/configuration_manifest.json): manifesto da configuração.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/validation_summary.json`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/validation_summary.json): resumo da validação funcional.
- [`hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/build_report.json`](hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/build_report.json): resumo da síntese HLS.

## Inventário do repositório

| Diretório | Conteúdo | Estado aproximado atual |
|---|---|---:|
| [`CPU/`](CPU/) | Benchmarks, telemetria, energia RAPL e relatórios CPU | 196 arquivos / 17 MB |
| [`GPU/`](GPU/) | Benchmarks, telemetria NVIDIA e relatórios GPU | 159 arquivos / 13 MB |
| [`hls4ml/`](hls4ml/) | Código gerado, simulação, síntese, IP e relatórios Vivado | 750 arquivos / 86 MB |
| [`scripts/`](scripts/) | Scripts de benchmark, análise, auditoria e manifest | 11 arquivos / aproximadamente 350 KB |
| [`Vitis AI/`](<Vitis AI/>) | Fluxo INT8, XModel, relatórios e pacote ZCU104 | 72 arquivos / 788 KB |

Os dados brutos preservados incluem arquivos NumPy com latências individuais, CSVs de passagens e telemetria, JSONs de metadados e validação, relatórios textuais, logs, checkpoints e o pacote exportado do IP.

## Scripts principais

- [`scripts/benchmark_mlp_iris_cpu.py`](scripts/benchmark_mlp_iris_cpu.py): benchmark de CPU.
- [`scripts/analyze_cpu_rapl.py`](scripts/analyze_cpu_rapl.py): consolidação e análise das campanhas RAPL.
- [`scripts/benchmark_mlp_iris_gpu.py`](scripts/benchmark_mlp_iris_gpu.py): benchmark GPU de 30 mil inferências.
- [`scripts/benchmark_mlp_iris_gpu_100k.py`](scripts/benchmark_mlp_iris_gpu_100k.py): benchmark GPU de 100 mil inferências.
- [`scripts/analyze_gpu_replicas.py`](scripts/analyze_gpu_replicas.py): consolidação de cinco campanhas GPU por tamanho.
- [`scripts/audit_board_readiness.py`](scripts/audit_board_readiness.py): auditoria reproduzível de prontidão, hashes e documentação.
- [`hls4ml/scripts/generate_mlp_iris_hls4ml_ip.py`](hls4ml/scripts/generate_mlp_iris_hls4ml_ip.py): conversão, validação e geração do IP.
- [`hls4ml/scripts/run_mlp_iris_hls4ml_ip.sh`](hls4ml/scripts/run_mlp_iris_hls4ml_ip.sh): execução automatizada do fluxo hls4ml/Vitis HLS.
- [`hls4ml/vivado_ooc_post_route.tcl`](hls4ml/vivado_ooc_post_route.tcl): síntese e implementação OOC no Vivado.

## Integridade dos artefatos principais

| Arquivo | SHA-256 |
|---|---|
| `iris_mlp_clean.h5` | `60d4e49c7082d97cac029ba919ffa8c88514ac9109d0bd69bafb23fa663fbb94` |
| `iris_scaler.joblib` | `beef6de7823db389c06565f0eb199ba71a353d4c2050e425ef6efd425bb20e31` |
| IP `xilinx_com_hls_mlp_iris_1_0.zip` | `88a11380ea472636af4b4b72348aeda9e0b3b2421456b71b0b30222b2a774ba3` |
| XModel Vitis AI `iris_mlp.xmodel` | `47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce` |


## Snapshot host-ready

O estado liberado para iniciar a ZCU104 é congelado por `manifests/host_ready_manifest.json`, `manifests/HOST_READY_SHA256SUMS.txt` e pelo arquivo `snapshots/MLP_HOST_READY_2026-09-03.tar.gz`. O hash externo do arquivo compactado fica em `snapshots/MLP_HOST_READY_2026-09-03.tar.gz.sha256`. O snapshot exclui somente a própria pasta `snapshots/`, metadados `.git` e caches Python; inclui fontes, dados, resultados, XModel, IP, golden e documentação.

Para verificar uma cópia restaurada:

```bash
sha256sum -c snapshots/MLP_HOST_READY_2026-09-03.tar.gz.sha256
sha256sum -c manifests/HOST_READY_SHA256SUMS.txt
python3 scripts/audit_board_readiness.py
```

## Liberação para a etapa de placa

A auditoria automatizada em [`HOST_READINESS_REPORT.md`](HOST_READINESS_REPORT.md) foi aprovada: hashes do IP/XModel, manifests, pacote golden, validações, links e as cinco campanhas CPU/GPU por tamanho estão consistentes. O projeto está liberado para iniciar as duas implementações físicas.

Isso não significa que a placa já esteja concluída. Permanecem necessariamente dependentes da ZCU104:

1. Vitis AI: validar imagem, fingerprint da DPU e VART; executar os 30 vetores; coletar 5 × 30k e 5 × 100k.
2. HLS4ML: construir wrapper AXI e block design; fechar timing do sistema completo; gerar bit/HWH; executar o golden; coletar 5 × 30k e 5 × 100k.
3. Energia: confirmar unidade, escala e escopo dos sensores da placa ou usar o mesmo wattímetro externo nas plataformas.

Use os dois guias ChatGPT Web de cada pasta. Todos os novos artefatos de um fluxo devem voltar para sua respectiva pasta (`Vitis AI/` ou `hls4ml/`).

> [`metodologia-benchmark.txt`](metodologia-benchmark.txt) descreve ResNet-8/CIFAR-10 e é apenas histórico. Para este projeto, prevalece [`METODOLOGIA_BENCHMARK_MLP_IRIS.md`](METODOLOGIA_BENCHMARK_MLP_IRIS.md).
