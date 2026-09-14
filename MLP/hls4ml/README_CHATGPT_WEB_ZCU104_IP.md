# Instruções para o ChatGPT Web — integrar o IP hls4ml no Vivado e medir na ZCU104

> Entregue este arquivo ao ChatGPT Web junto com a pasta `hls4ml/`, ou
> forneça os arquivos solicitados por ele. Este texto define o estado atual, a
> arquitetura recomendada, as etapas do block design e o protocolo de medição.
> O chat deve conduzir o usuário interativamente, com comandos de terminal e Tcl
> baseados nas saídas reais das ferramentas.

## 1. Missão do chat

Ajude o usuário a:

1. importar no Vivado 2024.2 o IP hls4ml já exportado;
2. construir um wrapper AXI4-Lite para a interface paralela do núcleo;
3. criar integralmente por Tcl um block design para a ZCU104;
4. conectar Processing System, clock, reset, AXI e o acelerador;
5. validar endereços, interfaces e handshake;
6. sintetizar, implementar, fechar timing e gerar bitstream;
7. exportar `.bit`, `.hwh`, relatórios e, se útil, `.xsa`;
8. carregar o overlay em uma imagem PYNQ da ZCU104;
9. validar as 30 amostras Iris;
10. medir desempenho, potência, energia e temperatura com o mesmo contrato
    estatístico usado no fluxo Vitis AI, CPU e GPU.

O IP já foi gerado, validado por C simulation, síntese HLS e co-simulação
C/RTL. Não regenere o hls4ml nem altere precisão, reuse factor ou clock sem
autorização explícita.

## 2. Como o chat deve trabalhar

Regras obrigatórias:

- Trabalhe por fases, enviando poucos comandos por vez.
- Espere o usuário colar stdout/stderr integral antes de avançar.
- Comece com inspeção somente de leitura.
- Antes de `sudo`, instalações, exclusões ou sobrescritas, explique a ação.
- Não use cliques manuais como fonte única da configuração; toda alteração
  relevante deve existir em Tcl ou fonte versionável.
- Não invente VLNV, board part, nomes de ports, endereços ou versões.
- Descubra os valores com Tcl e use exatamente as saídas do Vivado.
- Preserve o ZIP original do IP e seu SHA-256.
- Não edite RTL gerado pelo HLS dentro do pacote.
- Coloque wrapper, Tcl, projeto publicável, relatórios e outputs dentro de
  `hls4ml/`.
- Se precisar de um caminho temporário sem espaços, use uma pasta em `/tmp`
  somente durante a execução e publique todos os resultados de volta em
  `hls4ml/`.
- Gere logs com `tee` ou redirecionamento controlado.
- Falhe imediatamente em erro de DRC, timing, endereço, port ou conexão.
- Não trate relatório OOC como resultado do sistema completo.
- Não afirme que a placa foi medida sem execução física.
- Ao consultar a internet, use prioritariamente documentação oficial AMD/Xilinx,
  Vivado, Zynq UltraScale+ MPSoC, ZCU104 e PYNQ.

## 3. Estado imutável do IP

### Modelo

- MLP Iris `4 → 8 ReLU → 8 ReLU → 3 softmax`.
- 139 parâmetros.
- Acurácia Keras: 29/30 = 96,67%.
- Acurácia HLS C++: 29/30 = 96,67%.
- Concordância de classes Keras × HLS: 30/30.
- Co-simulação C/RTL Verilog: aprovada.
- O softmax faz parte do núcleo hls4ml.

### Configuração HLS

- hls4ml 1.3.0.
- Vitis HLS 2024.2.
- Parte: `xczu7ev-ffvc1156-2-e`.
- Clock: 100 MHz, período de 10 ns.
- Precisão padrão: `ap_fixed<16,6>`.
- Tabelas do softmax: `ap_fixed<18,8>`.
- Sem QKeras.
- `ReuseFactor=1`.
- `Strategy=Latency`.
- `IOType=io_parallel`.
- Nome/versionamento: `mlp_iris` v1.0.0.

### Artefato principal

```text
mlp_iris_apfixed16_6_rf1_100mhz/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip
```

SHA-256 esperado:

```text
88a11380ea472636af4b4b72348aeda9e0b3b2421456b71b0b30222b2a774ba3
```

O diretório que contém o `component.xml` também pode ser usado como
repositório de IP:

```text
mlp_iris_apfixed16_6_rf1_100mhz/mlp_iris_prj/solution1/impl/ip/
```

### Interface RTL real

Ports do top `mlp_iris`:

- `ap_clk`: entrada;
- `ap_rst`: reset ativo em nível alto;
- `ap_start`: entrada;
- `ap_done`: saída;
- `ap_idle`: saída;
- `ap_ready`: saída;
- `features_ap_vld`: entrada;
- `features[63:0]`: entrada;
- `layer7_out_0[15:0]`: saída;
- `layer7_out_0_ap_vld`: saída;
- `layer7_out_1[15:0]`: saída;
- `layer7_out_1_ap_vld`: saída;
- `layer7_out_2[15:0]`: saída;
- `layer7_out_2_ap_vld`: saída.

A entrada é bloqueante: iniciar sem apresentar `features_ap_vld` pode manter o
núcleo aguardando dados.

### Formato numérico e empacotamento

`ap_fixed<16,6>` é um número de 16 bits em complemento de dois, com 6 bits
inteiros incluindo o sinal e 10 bits fracionários.

Conversão de float para hardware:

```text
q = aplicar AP_TRN/AP_WRAP a (x × 2^10)
```

Os typedefs gerados usam os modos padrão `AP_TRN` e `AP_WRAP`. Na primeira validação, confirme bit a bit a mesma política usada
pela biblioteca `ap_fixed`; não suponha arredondamento diferente.

Empacotamento confirmado no RTL:

```text
features[15:0]   = x[0]
features[31:16]  = x[1]
features[47:32]  = x[2]
features[63:48]  = x[3]
```

As três saídas também são `ap_fixed<16,6>` e devem ser interpretadas com escala
`2^-10`.

### Resultados já obtidos

Síntese Vitis HLS:

- latência: 5 ciclos;
- II: 1;
- estimativa de período: 6,586 ns;
- LUT: 4.058;
- FF: 413;
- DSP: 108;
- BRAM18K: 3.

Vivado OOC pós-route:

- LUT: 1.944;
- registradores: 377;
- DSP: 108;
- RAMB18: 3;
- CLB: 330;
- WNS: +2,886 ns a 100 MHz;
- TNS: 0;
- WHS: +0,064 ns;
- redes não roteadas: zero.

Esses números cobrem apenas o núcleo, não o block design completo.

## 4. Arquitetura de integração recomendada

A interface do IP não é AXI. A solução recomendada é criar um wrapper RTL
AXI4-Lite dedicado que:

- instancia o IP `mlp_iris` sem editar o RTL gerado;
- recebe dois registradores de 32 bits e forma `features[63:0]`;
- gera um pulso `ap_start`;
- apresenta `features_ap_vld` no ciclo correto;
- mantém entrada estável enquanto o núcleo precisar;
- captura as três saídas quando os sinais `ap_vld` forem válidos;
- cria `done` sticky para o software não perder um pulso;
- permite limpar `done` por escrita;
- expõe `idle` e `ready`;
- inclui um contador de ciclos do start aceito ao done;
- opcionalmente inclui contador total de invocações;
- implementa reset determinístico;
- documenta integralmente o mapa de registradores.

Mapa inicial sugerido, que deve ser confirmado e congelado no wrapper:

| Offset | Acesso | Conteúdo |
|---:|---|---|
| `0x00` | R/W | control/status: start, done-sticky, idle, ready |
| `0x04` | R/W1C | interrupções/status adicional, se implementado |
| `0x10` | R/W | `features[31:0]` |
| `0x14` | R/W | `features[63:32]` |
| `0x20` | R | saída 0 em 16 bits |
| `0x24` | R | saída 1 em 16 bits |
| `0x28` | R | saída 2 em 16 bits |
| `0x2C` | R | bits de validade das saídas |
| `0x30` | R | ciclos da última inferência |
| `0x34` | R | contador de invocações |

O chat deve revisar esse mapa contra o RTL criado. Não continue se driver e
hardware discordarem.

Para quatro entradas e três saídas pequenas, AXI4-Lite é preferível a adicionar
DMA apenas por conveniência. Se o usuário quiser uma variante AXI-Stream/DMA,
trate-a como outro experimento e não misture seus resultados com a variante
MMIO.

## 5. Block design recomendado

O block design deve conter, no mínimo:

- Zynq UltraScale+ MPSoC;
- configuração de DDR e Fixed IO da ZCU104;
- uma porta master HPM do PS habilitada;
- clock PL de 100 MHz;
- `proc_sys_reset`;
- AXI SmartConnect ou interconnect apropriado;
- wrapper AXI4-Lite da MLP;
- núcleo `mlp_iris` instanciado pelo wrapper;
- clock e reset conectados coerentemente;
- endereço AXI atribuído e salvo;
- opcionalmente interrupção, somente se implementada e validada.

O clock do núcleo deve ser exatamente o clock AXI de 100 MHz na primeira
versão, evitando crossing desnecessário.

Não use `ap_rst` diretamente com polaridade errada. O wrapper deve converter
`aresetn` AXI, ativo baixo, para o reset ativo alto esperado pelo HLS.

O chat deve obter o VLNV real com Tcl, por exemplo consultando
`get_ipdefs`, antes de `create_bd_cell`.

## 6. Estrutura de arquivos exigida

Crie dentro desta pasta:

```text
hls4ml/vivado_zcu104_mlp/
├── README.md
├── src/
│   ├── mlp_iris_axi_wrapper.v
│   └── arquivos auxiliares
├── tb/
│   └── testbench do wrapper
├── tcl/
│   ├── 00_check_environment.tcl
│   ├── 01_create_project.tcl
│   ├── 02_create_block_design.tcl
│   ├── 03_build_bitstream.tcl
│   └── 04_export_artifacts.tcl
├── logs/
├── reports/
│   ├── utilization_post_synth.rpt
│   ├── utilization_post_route.rpt
│   ├── utilization_hierarchical.rpt
│   ├── timing_post_route.rpt
│   ├── drc_post_route.rpt
│   ├── power_estimated.rpt
│   └── address_map.txt
├── output/
│   ├── mlp_iris_overlay.bit
│   ├── mlp_iris_overlay.hwh
│   └── mlp_iris_overlay.xsa
├── pynq/
│   ├── mlp_iris_driver.py
│   ├── validate_mlp_iris.py
│   └── benchmark_mlp_iris.py
├── results_zcu104/
└── manifests/
    └── SHA256SUMS.txt
```

Se o Vivado não tolerar o caminho pai com espaços, mantenha fontes/Tcl em
`hls4ml/vivado_zcu104_mlp/`, execute o projeto temporário em uma pasta
`/tmp` sem espaços e copie automaticamente logs, relatórios e outputs de volta.
Não deixe o único projeto válido fora de `hls4ml/`.

## 7. Fases do Vivado

### Fase 0 — inventário

Antes de criar o projeto, colete:

- caminho do executável Vivado;
- `vivado -version`;
- licença disponível;
- board parts instalados para ZCU104;
- part `xczu7ev-ffvc1156-2-e`;
- VLNV encontrado para `mlp_iris`;
- SHA-256 do ZIP;
- presença e leitura de `component.xml`;
- espaço em disco;
- conteúdo de alterações existentes.

### Fase 1 — validação isolada do repositório de IP

Crie um Tcl somente de leitura que:

- configure `ip_repo_paths`;
- execute `update_ip_catalog`;
- liste o VLNV exato;
- consulte ports/interfaces;
- falhe se o IP não for encontrado;
- grave a saída em log.

Não avance para o block design até o catálogo reconhecer o IP.

### Fase 2 — criação e simulação do wrapper

O chat deve fornecer o RTL completo, não trechos incompletos. Depois:

- lint/síntese do wrapper;
- testbench do mapa AXI;
- teste de escrita das quatro entradas;
- teste de pulso start/vld;
- teste de captura de três saídas;
- teste de done sticky e clear;
- teste do contador de ciclos;
- reset em meio a uma operação;
- back-to-back apenas como teste secundário.

A simulação deve usar pelo menos vetores de
`mlp_iris_apfixed16_6_rf1_100mhz/tb_data/tb_input_features.dat` e comparar com
`mlp_iris_apfixed16_6_rf1_100mhz/tb_data/rtl_cosim_results.log` ou `mlp_iris_apfixed16_6_rf1_100mhz/tb_data/tb_output_predictions.dat`.
`mlp_iris_apfixed16_6_rf1_100mhz/validation_summary.json` contém os rótulos e as classes de referência.

Não aceite apenas “simulou sem erro”; salve as diferenças numéricas.

### Fase 3 — criação do projeto e block design por Tcl

O Tcl deve ser idempotente ou trabalhar sempre em diretório novo. Ele deve:

1. criar projeto para a parte correta;
2. configurar board part somente se disponível;
3. adicionar o repositório do IP;
4. criar o block design;
5. instanciar e configurar o Zynq MPSoC;
6. aplicar automação de board apenas quando a saída for revisada;
7. habilitar HPM master e FCLK de 100 MHz;
8. instanciar reset e interconnect;
9. adicionar o wrapper/acelerador;
10. conectar AXI, clock e reset;
11. executar `assign_bd_address`;
12. salvar o mapa de endereços;
13. executar `validate_bd_design`;
14. gerar targets;
15. criar o HDL wrapper do BD;
16. definir o top;
17. salvar o projeto.

Todo comando Tcl crítico deve testar a existência do objeto retornado.

### Fase 4 — síntese, implementação e bitstream

Execute em batch e salve logs. Gere:

- síntese do sistema completo;
- implementação/place-and-route;
- bitstream;
- `.hwh` correspondente ao mesmo bitstream;
- `.xsa`, se útil;
- relatório de utilização total;
- relatório hierárquico separando PS, AXI, wrapper e MLP;
- timing summary;
- DRC;
- clocks;
- power report estimado;
- mapa de endereços.

Critérios mínimos:

- runs concluídas;
- zero erros;
- zero nets não roteadas;
- WNS ≥ 0;
- TNS = 0;
- hold atendido;
- DRC crítico resolvido;
- clock do domínio MLP confirmado em 100 MHz;
- bit e hwh provenientes da mesma execução;
- endereço AXI consistente com o driver.

O `report_power` do Vivado é estimativa e não substitui medição física.

### Fase 5 — manifesto

Calcule SHA-256 de:

- IP ZIP original;
- wrapper;
- Tcl;
- bit;
- hwh;
- xsa;
- relatórios;
- driver PYNQ;
- scripts de validação e benchmark.

Registre versão do Vivado, parte, board part, commit/estado dos arquivos e data.

## 8. Implantação no PYNQ

Antes de carregar o overlay, o chat deve identificar:

- versão da imagem PYNQ;
- versão do Python e pacote `pynq`;
- modelo real da placa;
- espaço em disco;
- clocks e sensores disponíveis;
- usuário e caminho do Jupyter;
- endereço base extraído do HWH.

Copie `.bit` e `.hwh` com o mesmo basename:

```text
mlp_iris_overlay.bit
mlp_iris_overlay.hwh
```

Carregue com `pynq.Overlay`, confirme `is_loaded()` e inspecione
`ip_dict`. Não hardcode endereço antes de comparar com o HWH.

O driver deve:

- usar `MMIO` ou o objeto IP do overlay;
- normalizar entradas previamente com o mesmo StandardScaler;
- converter os quatro valores para `ap_fixed<16,6>`;
- empacotar na ordem confirmada;
- escrever os dois registradores;
- iniciar uma única inferência;
- aguardar done sticky com timeout;
- ler três saídas;
- converter complemento de dois;
- dividir por `2^10`;
- executar argmax;
- limpar status;
- detectar timeout e estados inválidos.

Inclua testes unitários em software para números positivos, negativos, zero,
overflow/wrap e empacotamento 64 bits.

## 9. Validação funcional na placa

Antes do benchmark:

- use as mesmas 30 amostras do holdout;
- pré-normalize fora da região cronometrada;
- compare cada saída da placa com a referência HLS/RTL;
- compare classes com Keras;
- salve saída bruta de 16 bits e saída float;
- calcule erro absoluto máximo e médio;
- confirme 29/30 acertos;
- confirme 30/30 classes iguais ao HLS C++;
- calcule matriz de confusão e IC95% Wilson;
- repita várias passagens e aborte em não determinismo.

O erro numérico esperado não é zero em relação ao Keras. A referência correta do
hardware é a co-simulação RTL da mesma configuração `ap_fixed<16,6>`.

## 10. Contrato comum de benchmark

O protocolo em `../METODOLOGIA_BENCHMARK_MLP_IRIS.md` foi escrito para
ResNet-8/CIFAR-10. Adapte para Iris:

- “imagem” vira “amostra”;
- “FPS” vira “inferências/s”;
- a acurácia usa somente 30 flores únicas;
- repetições medem desempenho e determinismo.

Use:

- batch 1;
- fluxo serial e síncrono;
- uma inferência em voo;
- dados normalizados previamente;
- 200 warm-ups;
- baseline de potência de 5 s após aquecimento;
- telemetria de 100 ms;
- 30 amostras por passagem;
- seed `20260831`;
- 30.000 e 100.000 inferências exatas;
- cinco campanhas independentes de cada tamanho;
- nenhum outlier removido;
- nenhum print no trecho medido.

Para 30k: 1.000 passagens completas.

Para 100k: 3.333 passagens completas e uma passagem parcial de 10.

## 11. Janelas de medição obrigatórias

### 11.1 Ciclos do núcleo

Use o contador implementado no wrapper, iniciado quando start/vld forem aceitos
e encerrado em `ap_done`.

Reporte:

```text
kernel_cycles
kernel_latency_ns = kernel_cycles × 10 ns
```

A síntese HLS estimou 5 ciclos. O wrapper pode observar diferença de handshake;
se ocorrer, explique exatamente a fronteira do contador.

### 11.2 Chamada MMIO

Começa imediatamente antes da primeira escrita necessária e termina após:

- escrita da entrada;
- start;
- polling/espera de done;
- leitura das três saídas.

Nome sugerido:

```text
latency_mmio_call_ms
```

### 11.3 Aplicação end-to-end

Começa antes de converter/empacotar os quatro valores já normalizados e termina
depois de:

- MMIO;
- desempacotamento;
- conversão para float;
- argmax;
- materialização e armazenamento da classe.

Nome sugerido:

```text
latency_application_end_to_end_ms
```

Esta é a janela primária comparável ao fluxo Vitis AI. O núcleo hls4ml já inclui
softmax; registre essa diferença.

### 11.4 Passagem

Cronometre externamente cada grupo de 30 amostras. Throughput efetivo:

```text
throughput_effective = total_inferences / sum(passage_duration_seconds)
```

Reporte também throughput médio das passagens, sem substituí-lo pela razão dos
totais.

Não use `1000 / média_da_latência` como único cálculo de vazão.

## 12. Estatística obrigatória

Para cada tipo de latência, preserve todos os valores e reporte:

- média;
- mediana;
- desvio-padrão amostral;
- CV;
- mínimo;
- máximo;
- p90;
- p95;
- p99.

IC95%:

- use médias das passagens completas como observações;
- use t de Student;
- informe o número de passagens;
- exclua apenas a passagem parcial do IC, nunca dos agregados globais;
- registre autocorrelação como limitação.

Acurácia:

- somente 30 flores únicas;
- acertos/30;
- Wilson 95%;
- F1 macro;
- MCC;
- kappa;
- matriz de confusão.

Nas cinco campanhas, a unidade independente é a campanha. Calcule média,
mediana, desvio, CV e IC95%. Não use teste pareado se as condições não forem
pareadas.

CPU e GPU já possuem cinco campanhas independentes de 30k e cinco de 100k. Não repita os testes de host; colete cinco campanhas por tamanho na ZCU104 e trate a campanha como unidade experimental.

## 13. Potência e energia na ZCU104

Use exatamente a mesma política do documento do Vitis AI para permitir
comparação.

Primeiro descubra sensores reais:

- wattímetro externo, se disponível;
- INA226/PMBus;
- `/sys/class/hwmon`;
- APIs PYNQ de rails;
- temperatura do PS/PL;
- frequência do clock PL;
- utilização do ARM.

Não invente rails e não some domínios sobrepostos.

Baseline de 5 s:

- overlay carregado;
- driver inicializado;
- depois dos 200 warm-ups;
- sem inferências;
- mesmo sampler e mesma taxa de 100 ms.

Integração:

```text
E_total = integral de P(t) dt
E_por_inferencia = E_total / N
P_dinamica(t) = max(P_ativa(t) - P_baseline, 0)
E_dinamica = integral de P_dinamica(t) dt
```

Reporte separadamente placa, PL, PS e DDR somente quando o hardware permitir.

A energia do Vivado é estimada; a energia do benchmark deve vir dos sensores ou
do wattímetro. Declare o escopo ao comparar com RAPL da CPU e sensor da GPU.

## 14. Campanhas físicas

Execute cinco campanhas de 30k e cinco de 100k, mantendo:

- mesma placa e fonte;
- mesma frequência de 100 MHz;
- mesma imagem PYNQ;
- mesmo bitstream;
- mesmo notebook/script;
- mesmo holdout;
- mesma seed;
- mesma taxa de telemetria;
- temperatura inicial e tempo de resfriamento registrados.

Se possível, alterne a ordem das campanhas hls4ml e Vitis AI ou use um plano
balanceado. Se as duas soluções exigirem imagens de boot diferentes, registre a
reinicialização e não chame os ensaios de pareados.

Cada campanha deve usar diretório novo e hashes próprios.

## 15. Estrutura mínima dos resultados

```text
hls4ml/vivado_zcu104_mlp/results_zcu104/
├── 30000_rep1/
├── 30000_rep2/
├── 30000_rep3/
├── 30000_rep4/
├── 30000_rep5/
├── 100000_rep1/
├── 100000_rep2/
├── 100000_rep3/
├── 100000_rep4/
├── 100000_rep5/
├── campaigns.csv
├── aggregate_statistics.json
├── comparison_vitis_ai_cpu_gpu.md
└── SHA256SUMS.txt
```

Cada campanha:

- `metadata.json`;
- `environment.txt`;
- `overlay_manifest.json`;
- `commands.log`;
- `validation_unique_30.json`;
- `predictions_unique_30.csv`;
- `kernel_cycles.npy`;
- `latencies_mmio_call_ms.npy`;
- `latencies_application_end_to_end_ms.npy`;
- `passages.csv`;
- `telemetry.csv`;
- `telemetry_summary.json`;
- `benchmark_summary.json`;
- `SHA256SUMS.txt`.

## 16. Comparação hls4ml × Vitis AI

A tabela final deve manter linhas separadas para:

- precisão numérica: `ap_fixed<16,6>` versus INT8;
- softmax: hardware hls4ml versus ARM no Vitis AI;
- acurácia nas mesmas 30 amostras;
- ciclos/latência do núcleo;
- latência da chamada ao runtime/MMIO;
- latência completa da aplicação;
- throughput batch 1 serial;
- potência por escopo físico;
- energia total e dinâmica por inferência;
- temperatura;
- frequência;
- recursos do block design hls4ml;
- arquitetura DPU usada pelo Vitis AI;
- número de campanhas.

Não compare os 50 ns estimados do núcleo HLS com o tempo end-to-end do Vitis AI.
Não compare recursos isolados do núcleo HLS com recursos totais de uma DPU
preexistente sem explicar o escopo.

Se for medido modo saturado/pipelined, publique em seção secundária. O resultado
principal deve continuar batch 1 serial e síncrono.

## 17. Critérios de conclusão

O chat só deve declarar o trabalho concluído quando houver:

- wrapper AXI completo e versionado;
- testbench aprovado;
- Tcl reproduzível;
- block design validado;
- endereço documentado;
- síntese e implementação sem erros;
- timing fechado a 100 MHz;
- DRC revisado;
- bit e hwh correspondentes;
- overlay carregado no PYNQ;
- 30 amostras validadas;
- determinismo comprovado;
- dez campanhas físicas coletadas;
- telemetria e energia com escopo documentado;
- relatórios brutos e agregados;
- hashes finais;
- resultados copiados para `hls4ml/`.

## 18. Primeira resposta esperada do ChatGPT Web

Ao receber este arquivo, o chat deve:

1. resumir o IP e a interface encontrada;
2. afirmar que preservará o IP original;
3. explicar a necessidade do wrapper AXI4-Lite;
4. fornecer somente comandos de inventário da Fase 0;
5. incluir comandos Tcl de leitura para descobrir board part e VLNV;
6. pedir a saída completa;
7. não criar o block design antes de validar ambiente e catálogo do IP.

## 19. Referências locais que prevalecem

Leia antes de orientar:

- `README.md` desta pasta;
- `RESULTADOS_IP.md`;
- `COMPARACAO_RECURSOS_VITIS_VIVADO.md`;
- `mlp_iris_apfixed16_6_rf1_100mhz/configuration_manifest.json`;
- `mlp_iris_apfixed16_6_rf1_100mhz/validation_summary.json`;
- `mlp_iris_apfixed16_6_rf1_100mhz/build_report.json`;
- `mlp_iris_apfixed16_6_rf1_100mhz/reports/resource_comparison.json`;
- `../METODOLOGIA_BENCHMARK_MLP_IRIS.md`;
- `../CPU/README.md`;
- `../GPU/README.md`;
- `../Vitis AI/README_CHATGPT_WEB_ZCU104_XMODEL.md`.

Quando uma recomendação externa conflitar com ports, hashes ou relatórios
locais, confie primeiro nos artefatos reais e peça confirmação ao usuário.

## Atualização de prontidão de 2026-09-03

A metodologia normativa é `../METODOLOGIA_BENCHMARK_MLP_IRIS.md`. CPU e GPU já possuem cinco campanhas de 30k e cinco de 100k; a GPU foi completada nesta revisão. Não solicite novas réplicas de host antes da placa. Preserve uma campanha por diretório e trate `n=5` campanhas como unidade experimental.
