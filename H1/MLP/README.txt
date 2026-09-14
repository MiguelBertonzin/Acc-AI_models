H1 — MLP Iris: estudo do ReuseFactor global
===========================================

Data da campanha: 9 de setembro de 2026. Configurações concluídas: RF1, RF4, RF8, RF16 e RF32.

Os cinco projetos e IPs estão salvos nesta pasta, com código C++/RTL, pesos,
vetores de teste, logs, configurações e relatórios. Todos passaram por validação
C++, C simulation, síntese HLS, co-simulação C/RTL Verilog e exportação de IP.
A auditoria funcional foi aprovada. Em 10 de setembro de 2026, os cinco RTLs
foram sintetizados e otimizados no Vivado 2024.2 em modo Out-of-Context (OOC), com
clock de 10 ns. Foram coletados utilização pós-síntese, WNS/TNS, metodologia e
checkpoints. Não houve place-and-route, bitstream ou medição física na ZCU104.

1. Objetivo e escopo experimental
---------------------------------

Avaliar como o ReuseFactor global modifica a relação entre utilização de recursos
e desempenho da MLP, preservando o restante do fluxo anterior.

A única variável experimental alterada foi HLSConfig.Model.ReuseFactor:
1, 4, 8, 16 e 32. Inicialmente foram gerados RF1/4/8; em seguida a série foi
ampliada com RF16/32, preservando os três projetos concluídos.

Referência anterior: MLP/hls4ml (../../MLP/hls4ml/README.md), especificamente
projeto RF1 original (../../MLP/hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/).
O novo RF1 reproduziu integralmente os campos do relatório CSynthesisReport
da referência e as saídas C/RTL das 30 amostras.

A estratégia efetivamente usada nesta campanha é Latency em todos os RFs,
como no fluxo original. As restrições discutidas para Strategy=Resource não
foram usadas para rejeitar RF16/32 nesta série. A conversão confirmou esses
valores nas três densas, inclusive na saída.

2. Modelo, dados e pré-processamento
------------------------------------

Arquivos de entrada preservados:

- iris_mlp_clean.h5 (iris_mlp_clean.h5): modelo Keras em FP32, sem camadas/configurações de quantização.
- iris_scaler.joblib (iris_scaler.joblib): scaler previamente ajustado, usado por transform; não é reajustado nesta campanha.

Arquitetura:

Etapa   | Dimensões     | Ativação | Parâmetros, incluindo bias
--------+---------------+----------+---------------------------
Entrada | 4 atributos   | —        | 0                         
dense1  | 4 → 8         | ReLU     | 40                        
dense2  | 8 → 8         | ReLU     | 72                        
output  | 8 → 3         | Softmax  | 27                        
Total   | 4 → 8 → 8 → 3 | —        | 139                       

O modelo não foi retreinado e os pesos não foram ajustados entre RFs. A Softmax
permanece no hardware. A classificação é obtida por argmax das três saídas.

Os dados vêm de sklearn.datasets.load_iris(). O holdout é reconstruído com
train_test_split(test_size=0.2, random_state=42, stratify=labels):
30 amostras de teste, sempre na mesma ordem, seguidas de scaler.transform
e conversão para float32. O restante do split não é usado para treinamento
nesta campanha. Essa avaliação preserva o procedimento anterior; não constitui
uma nova auditoria da separação treino/teste do treinamento original.

As entradas usadas pelo testbench e as saídas FP32 estão salvas em
RF*/tb_input_features.dat e RF*/tb_output_predictions.dat.
Os rótulos esperados e classes previstas constam em validation_summary.json.
Não houve campanha de warm-up, repetições temporais ou batch de aplicação na placa;
a validação funcional contém 30 transações de inferência.

SHA-256 das entradas, iguais aos arquivos do fluxo anterior:

60d4e49c7082d97cac029ba919ffa8c88514ac9109d0bd69bafb23fa663fbb94  iris_mlp_clean.h5
beef6de7823db389c06565f0eb199ba71a353d4c2050e425ef6efd425bb20e31  iris_scaler.joblib

3. Configuração de hardware mantida constante
---------------------------------------------

Parâmetro                     | Valor                                                                           
------------------------------+---------------------------------------------------------------------------------
Backend                       | Vitis                                                                           
Dispositivo                   | xczu7ev-ffvc1156-2-e — ZCU104                                                   
Clock solicitado              | 10 ns / 100 MHz                                                                 
Incerteza de clock            | 27%, padrão do backend utilizado                                                
Strategy                      | Latency                                                                         
IOType                        | io_parallel                                                                     
Granularidade da configuração | model                                                                           
Precisão padrão               | fixed<16,6> na configuração; ap_fixed<16,6> no C++                              
Saída                         | 3 valores com Softmax                                                           
Projeto/top                   | mlp_iris                                                                        
BramFactor                    | 1000000000                                                                      
TraceOutput                   | false                                                                           
BitExact                      | null, configuração original                                                     
Overrides de RF por camada    | Nenhum                                                                          
Otimização de FIFO            | Nenhuma etapa de otimização; arquitetura io_parallel                            
Vivado OOC pós-síntese        | Concluído externamente para os cinco RFs; `vsynth=False` somente no build hls4ml

O RF global é herdado pelas camadas, inclusive pelas ativações e Softmax. Isso
não significa que todas as operações internas usem RF da mesma maneira. Os
atributos efetivos estão em effective_layers.json; as constantes geradas estão
em firmware/parameters.h. As três densas usam nnet::latency em todos os casos.

Precisão numérica exata
-----------------------

O arquivo Keras permanece FP32. O ponto fixo é aplicado na conversão hls4ml:

- Entrada, pesos, biases, acumuladores padrão e resultados: ap_fixed<16,6>,
  com 16 bits totais e 6 bits inteiros, incluindo sinal; 10 bits fracionários.
- O tipo padrão não explicita modos adicionais: usa os defaults de ap_fixed
  (truncamento e wrap). Não foi substituído pelos modos Q22.12 usados na LeNet.
- Tipos auxiliares de tabela declarados: ap_fixed<18,8>.
- Tabelas exponencial/inversa e entrada da inversa da Softmax:
  ap_fixed<18,8,AP_RND,AP_SAT,0>.
- Normalização interna declarada na configuração da Softmax: ap_ufixed<15,5>.
- Softmax: implementação stable, tabelas exponencial e inversa com 1024 entradas.

As declarações completas estão em firmware/defines.h e firmware/parameters.h.
A auditoria comparou os tipos e os pesos gerados com a referência anterior.

4. Ambiente e ferramentas
-------------------------

Versões registradas nos manifestos e conferidas contra a referência:

Componente                    | Versão
------------------------------+-------
python                        | 3.12.7
tensorflow                    | 2.21.0
keras                         | 3.13.2
hls4ml                        | 1.3.0 
numpy                         | 1.26.4
scikit_learn_runtime          | 1.5.1 
Vitis HLS                     | 2024.2
Vivado usado no empacotamento | 2024.2

O gerador inclui os binários instalados em /opt/Xilinx/Vitis/2024.2/bin,
/opt/Xilinx/Vitis_HLS/2024.2/bin e /opt/Xilinx/Vivado/2024.2/bin no PATH.
Preserva a preparação anterior com LD_PRELOAD de
/usr/lib/x86_64-linux-gnu/libstdc++.so.6 e desabilita GPU no processo de conversão.

O caminho do workspace contém espaços. Como o Vitis HLS não aceita esse caminho
para construir os projetos, cada RF é gerado em diretório exclusivo /tmp/h1_mlp_rfN_*
e depois copiado integralmente para RFN/. Os resultados permanentes estão aqui;
logs e arquivos de projeto ainda podem mencionar os caminhos temporários de origem.

5. Procedimento executado
-------------------------

1. Conferência do modelo/scaler com os hashes do fluxo anterior.
2. Adaptação de uma cópia do gerador original, parametrizando RF e saída.
3. Conversão hls4ml com a mesma precisão, estratégia, IO, modelo e clock.
4. Registro dos atributos efetivos e verificação de RF/Latency nas densas.
5. Compilação C++ e inferência das 30 amostras; comparação com Keras.
6. C simulation do testbench.
7. C synthesis pelo Vitis HLS.
8. Co-simulação Verilog com XSIM e validação das saídas C contra RTL.
9. Exportação no formato ip_catalog, incluindo component.xml e ZIP.
10. Cópia do projeto, logs e resultados para a subpasta do RF.
11. Auditoria dos cinco projetos e consolidação inicial em CSV/JSON.
12. Síntese e opt_design Vivado OOC de cada RTL com clock de 10 ns.
13. Extração automática dos recursos/timing Vivado e inclusão de novas colunas no CSV.

A chamada de build usa reset=True, csim=True, synth=True, cosim=True,
validation=True, export=True e vsynth=False. Esse último parâmetro informa
que o build hls4ml não chamou a síntese Vivado; a etapa Vivado OOC foi executada
depois, de modo externo e uniforme, por run_vivado_ooc.tcl. O RF1 anterior não
foi simplesmente copiado: foi gerado novamente nesta campanha e reproduziu a referência.

Horários registrados pelo executor, fuso America/Sao_Paulo:

RF | Início              | Fim, mesma data | Tempo total do executor (s)
---+---------------------+-----------------+----------------------------
1  | 2026-09-09 15:41:41 | 15:42:43        | 62.3                       
4  | 2026-09-09 15:42:43 | 15:43:37        | 53.6                       
8  | 2026-09-09 15:43:37 | 15:44:30        | 52.9                       
16 | 2026-09-09 15:50:07 | 15:50:58        | 51.4                       
32 | 2026-09-09 15:50:58 | 15:51:50        | 51.9                       

Esses tempos incluem conversão, validação, ferramentas e publicação dos arquivos.
São tempos de construção no host; não são latências de inferência nem um benchmark
controlado de compilação.

6. Resultados funcionais coletados
----------------------------------

RF | Keras: acertos | hls4ml: acertos | Concordância top-1 Keras/HLS | Co-simulação Verilog
---+----------------+-----------------+------------------------------+---------------------
1  | 29/30          | 29/30           | 30/30                        | Pass                
4  | 29/30          | 29/30           | 30/30                        | Pass                
8  | 29/30          | 29/30           | 30/30                        | Pass                
16 | 29/30          | 29/30           | 30/30                        | Pass                
32 | 29/30          | 29/30           | 30/30                        | Pass                

Em todos os RFs:

- Acurácia: 29/30 = 96,67% no holdout.
- Concordância de classes Keras/HLS: 30/30 = 100%.
- Erro absoluto médio das saídas HLS contra Keras: 0.016744690.
- Erro absoluto máximo das saídas HLS contra Keras: 0.177612126.
- Os valores de saída registrados para C simulation e RTL coincidem em todas as
  30 transações, entre os cinco RFs e com a referência HLS anterior.

As probabilidades HLS não são idênticas às probabilidades FP32. A quantização e a
aproximação da Softmax mantiveram as classes neste conjunto, apesar do erro nas
saídas numéricas. A concordância entre C/RTL aqui é verificada nos resultados
registrados dos testbenches; não representa uma prova formal para todas as entradas.

7. Resultados de síntese e co-simulação
---------------------------------------

LUT, FF, DSP, BRAM e URAM abaixo são estimativas da síntese HLS. Latência e II
em ciclos coincidem entre o relatório HLS e a co-simulação Verilog nesta campanha.
Os mínimos e máximos reportados são iguais para cada configuração.

RF global | Latência (ciclos) | Latência a 100 MHz (ns) | II obtido (ciclos) | DSP | LUT   | FF    | BRAM_18K | URAM
----------+-------------------+-------------------------+--------------------+-----+-------+-------+----------+-----
1         | 5                 | 50                      | 1                  | 108 | 4.058 | 413   | 3        | 0   
4         | 14                | 140                     | 4                  | 33  | 5.187 | 1.950 | 3        | 0   
8         | 24                | 240                     | 8                  | 18  | 5.096 | 2.341 | 3        | 0   
16        | 42                | 420                     | 14                 | 11  | 4.848 | 2.885 | 3        | 0   
32        | 80                | 800                     | 28                 | 7   | 5.059 | 3.055 | 3        | 0   

BRAM_18K representa unidades de 18 Kb usadas pelo relatório HLS; não deve ser
misturado diretamente com contagens de tiles BRAM de relatórios Vivado.
Disponibilidades registradas pelo HLS: 1728 DSP, 230400 LUT, 460800 FF,
624 BRAM_18K e 96 URAM.

RF | Clock solicitado (ns) | Período estimado HLS (ns) | 1000/período estimado (MHz)
---+-----------------------+---------------------------+----------------------------
1  | 10,000                | 6.586                     | 151.84                     
4  | 10,000                | 6.586                     | 151.84                     
8  | 10,000                | 5.588                     | 178.95                     
16 | 10,000                | 6.586                     | 151.84                     
32 | 10,000                | 5.588                     | 178.95                     

A última coluna é apenas o inverso do do período estimado, não uma frequência
validada em place-and-route. A operação solicitada permanece em 100 MHz.
WNS/TNS foram coletados na netlist OOC pós-síntese e permanecem separados dos
valores HLS; ainda não há timing pós-route.

Como interpretar as métricas
----------------------------

- Latência em tempo: latência_ciclos × 10 ns.
- II: intervalo obtido entre transações no núcleo, conforme relatórios; é diferente
  da latência de uma transação. O RF solicitado não deve ser substituído pelo II.
- RF16 preservou o RF efetivo 16 e obteve II14; RF32 preservou RF32 e obteve II28.
- Uma projeção ideal de vazão do núcleo seria 100e6 / II inferências/s, condicionada
  a alimentação e recepção contínuas e ao protocolo. Não foi medida na placa e não
  foi registrada como FPS real da aplicação.
- A redução de DSP não implica redução de todos os recursos ou de potência.

Comparando RF32 com RF1, a estimativa de DSP caiu de 108 para 7 (93.52%),
a latência aumentou de 5 para 80 ciclos (16 vezes) e o II de 1 para 28 ciclos.
LUT aumentou de 4058 para 5059 (24.67%) e FF de 413 para 3055.
RF1 apresenta menor latência/II; RF32 apresenta menor DSP; RF16 usa menos LUT/FF
que RF32, mas usa mais DSP. A escolha depende dos recursos limitantes e da meta
de desempenho, a confirmar na implementação física.

8. Validação de utilização no Vivado
------------------------------------

Método
------

Cada caso usa o RTL preservado em RF*/mlp_iris_prj/solution1/syn/verilog/.
O script run_vivado_ooc.tcl lê o Verilog, executa synth_design em modo OOC para
o top mlp_iris e a parte xczu7ev-ffvc1156-2-e, cria ap_clk com período de
10 ns, executa opt_design e salva:

- utilization.rpt e utilization_hierarchical.rpt;
- timing_summary.rpt com WNS/TNS a 100 MHz;
- methodology.rpt;
- checkpoint mlp_iris_RFN_post_synth.dcp;
- status.txt confirmando conclusão.

A execução foi sequencial para RF1, RF4, RF8, RF16 e RF32. Todos terminaram sem
erros de síntese ou otimização. O aviso HD.CLK_SRC é esperado em análise OOC,
pois o buffer e a origem física do clock pertencem ao design de nível superior.
Ele deve ser resolvido na integração e não invalida a contagem de recursos.

O Vitis HLS fornece uma estimativa antes do mapeamento completo. O Vivado mapeia a
netlist para primitivas do dispositivo e elimina ou combina lógica. Estes valores
são mais representativos da implementação que a estimativa HLS, porém ainda não
incluem posicionamento, roteamento, PS, DMA, wrapper ou interconexões.

Tabela consolidada HLS, RTL e Vivado
------------------------------------

RF | Latência/II RTL (ciclos) | Latência a 100 MHz (µs) | Vazão teórica do núcleo (inferências/s) | DSP HLS → Vivado | LUT HLS → Vivado | FF HLS → Vivado | BRAM18 HLS → Vivado | URAM | WNS (ns) | TNS (ns) | Cabe?
---+--------------------------+-------------------------+-----------------------------------------+------------------+------------------+-----------------+---------------------+------+----------+----------+------
1  | 5 / 1                    | 0,05                    | 100.000.000                             | 108 → 108        | 4.058 → 2.098    | 413 → 377       | 3 → 3               | 0    | +5,096   | 0,000    | Sim  
4  | 14 / 4                   | 0,14                    | 25.000.000                              | 33 → 33          | 5.187 → 2.461    | 1.950 → 1.786   | 3 → 3               | 0    | +4,646   | 0,000    | Sim  
8  | 24 / 8                   | 0,24                    | 12.500.000                              | 18 → 18          | 5.096 → 2.562    | 2.341 → 2.105   | 3 → 3               | 0    | +4,827   | 0,000    | Sim  
16 | 42 / 14                  | 0,42                    | 7.142.857                               | 11 → 11          | 4.848 → 2.887    | 2.885 → 2.453   | 3 → 3               | 0    | +4,594   | 0,000    | Sim  
32 | 80 / 28                  | 0,80                    | 3.571.429                               | 7 → 7            | 5.059 → 2.846    | 3.055 → 2.550   | 3 → 3               | 0    | +4,459   | 0,000    | Sim  

A vazão é 100 MHz / II e supõe transações contínuas sem custo de interface. Não
é FPS medido em placa. A latência usa os ciclos da co-simulação RTL, que coincidem
com o relatório HLS nesta MLP.

Recursos Vivado e ocupação da ZCU104
------------------------------------

RF | CLB LUTs      | CLB Registers | DSP48E2     | RAMB18 | BRAM tiles  | URAM | Caminho crítico pós-síntese (ns)
---+---------------+---------------+-------------+--------+-------------+------+---------------------------------
1  | 2.098 (0,91%) | 377 (0,08%)   | 108 (6,25%) | 3      | 1,5 (0,48%) | 0    | 4,904                           
4  | 2.461 (1,07%) | 1.786 (0,39%) | 33 (1,91%)  | 3      | 1,5 (0,48%) | 0    | 5,354                           
8  | 2.562 (1,11%) | 2.105 (0,46%) | 18 (1,04%)  | 3      | 1,5 (0,48%) | 0    | 5,173                           
16 | 2.887 (1,25%) | 2.453 (0,53%) | 11 (0,64%)  | 3      | 1,5 (0,48%) | 0    | 5,406                           
32 | 2.846 (1,24%) | 2.550 (0,55%) | 7 (0,41%)   | 3      | 1,5 (0,48%) | 0    | 5,541                           

Capacidades consideradas: 230.400 LUTs, 460.800 registers, 1.728 DSPs, 312 tiles
de BRAM/624 blocos equivalentes de 18 Kb e 96 URAM. Todas as variantes atendem o
clock de 10 ns na síntese OOC, com WNS positivo e TNS zero, e todas cabem com ampla
folga nessa etapa.

Diferença entre a estimativa HLS e o Vivado
-------------------------------------------

RF | DSP Vivado/HLS | LUT Vivado/HLS | FF Vivado/HLS | BRAM18 Vivado/HLS
---+----------------+----------------+---------------+------------------
1  | 100,00%        | 51,70%         | 91,28%        | 100,00%          
4  | 100,00%        | 47,45%         | 91,59%        | 100,00%          
8  | 100,00%        | 50,27%         | 89,92%        | 100,00%          
16 | 100,00%        | 59,55%         | 85,03%        | 100,00%          
32 | 100,00%        | 56,26%         | 83,47%        | 100,00%          

DSP e BRAM foram previstos exatamente pelo HLS. As LUTs reais ficaram entre
47,45% e 59,55% da estimativa; os registers ficaram entre 83,47% e 91,59%.
A variação mostra que a tendência deve ser confirmada no Vivado, especialmente
para LUT/FF, em vez de comparar apenas as estimativas HLS.

Interpretação da H1 após o Vivado
---------------------------------

O aumento do RF reduziu DSPs de 108 para 7, queda de 93,52%, mas aumentou a
latência de 5 para 80 ciclos e reduziu a vazão teórica em 96,43%. Ao mesmo tempo,
as LUTs Vivado subiram de 2.098 para 2.846 e os registers de 377 para 2.550.
Portanto, serializar mais não reduz todos os tipos de recurso.

Como todas as variantes ocupam pouca área da ZCU104, RF1 é o ponto de maior
desempenho e ainda consome somente 6,25% dos DSPs. RF4 reduz DSPs em 69,44% frente
a RF1 e mantém 25 milhões de inferências/s teóricas, sendo um compromisso útil
quando DSPs precisam ser reservados. RF16 e RF32 priorizam DSP mínimo; entre eles,
RF32 economiza apenas 4 DSPs e 41 LUTs, mas dobra o II de 14 para 28 e acrescenta
97 registers. A decisão final ainda depende da integração e do place-and-route.

9. IPs exportados e interface
-----------------------------

Variante | Pacote para distribuição                                                   | Diretório para catálogo Vivado                          
---------+----------------------------------------------------------------------------+---------------------------------------------------------
RF1      | [ZIP](RF1/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip)  | [IP descompactado](RF1/mlp_iris_prj/solution1/impl/ip/) 
RF4      | [ZIP](RF4/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip)  | [IP descompactado](RF4/mlp_iris_prj/solution1/impl/ip/) 
RF8      | [ZIP](RF8/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip)  | [IP descompactado](RF8/mlp_iris_prj/solution1/impl/ip/) 
RF16     | [ZIP](RF16/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip) | [IP descompactado](RF16/mlp_iris_prj/solution1/impl/ip/)
RF32     | [ZIP](RF32/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip) | [IP descompactado](RF32/mlp_iris_prj/solution1/impl/ip/)

Todos preservam o top mlp_iris e a identidade IP-XACT
xilinx.com:hls:mlp_iris:1.0. Use somente o repositório da variante desejada
em um projeto Vivado para evitar ambiguidade entre IPs com a mesma identidade.
O nome do ZIP é igual nos cinco diretórios; identifique o RF pelo caminho e hash.

Interface observada no RTL:

Sinal                                    | Função                               
-----------------------------------------+--------------------------------------
ap_clk / ap_rst                          | Clock e reset                        
ap_start / ap_done / ap_idle / ap_ready  | Controle ap_ctrl_hs                  
features[63:0] / features_ap_vld         | Quatro entradas de 16 bits e validade
layer7_out_0[15:0] / layer7_out_0_ap_vld | Saída da classe 0 e validade         
layer7_out_1[15:0] / layer7_out_1_ap_vld | Saída da classe 1 e validade         
layer7_out_2[15:0] / layer7_out_2_ap_vld | Saída da classe 2 e validade         

O IP não fornece AXI DMA/AXI-Lite diretamente. A integração com PS/DDR precisa do
wrapper apropriado. Como latência e II variam entre RFs, a lógica de controle deve
respeitar os sinais do protocolo, sem assumir os 5 ciclos do RF1.

10. Organização dos arquivos salvos
-----------------------------------

H1/MLP/
├── README.md
├── iris_mlp_clean.h5
├── iris_scaler.joblib
├── generate_mlp_h1.py
├── run_h1.py
├── audit_h1.py
├── run_vivado_ooc.tcl
├── collect_vivado.py
├── write_readme_txt.py
├── README.txt
├── resultados_hls.csv
├── vivado_ooc_summary.json
├── audit_summary.json
├── IP_SHA256SUMS.txt
├── SHA256SUMS_ALL.txt
├── RF1/
├── RF4/
├── RF8/
├── RF16/
└── RF32/

Conteúdo relevante de cada RFN/:

Arquivo/diretório                   | Conteúdo                                                    
------------------------------------+-------------------------------------------------------------
configuration_manifest.json         | Parâmetros, versões, hashes do modelo/scaler                
effective_layers.json               | RF e estratégia efetivos após conversão                     
validation_summary.json             | Acurácia, erros, classes e rótulos esperados                
build_report.json                   | CSimResults, CosimResults, CSynthesisReport e CosimReport   
exported_artifacts.json             | Inventário dos artefatos exportados pelo gerador            
run_status.json                     | Status, comando, início/fim e staging                       
run.log                             | Log completo do gerador e das ferramentas                   
hls4ml_config.yml                   | Configuração gerada do projeto                              
tb_input_features.dat               | Entradas padronizadas do holdout                            
tb_output_predictions.dat           | Saídas FP32 do Keras para o testbench                       
hls_predictions.dat                 | Saídas da validação C++                                     
tb_data/                            | Dados e resultados usados pela simulação                    
firmware/                           | C++ gerado, tipos, constantes, pesos e biblioteca nnet_utils
mlp_iris_prj/solution1/syn/report/  | Relatórios de síntese, incluindo mlp_iris_csynth.rpt/xml    
mlp_iris_prj/solution1/syn/verilog/ | RTL Verilog sintetizado                                     
mlp_iris_prj/solution1/sim/report/  | Relatórios de co-simulação                                  
mlp_iris_prj/solution1/impl/ip/     | component.xml, HDL empacotado e ZIP do IP                   
vivado_ooc/                         | utilização, timing, metodologia e checkpoint pós-síntese    

Os CSV/JSON consolidados são índices dos relatórios originais, que permanecem
preservados por RF. Não é necessário depender de /tmp para consultar ou importar
os IPs exportados. Reexecutar os projetos HLS copiados pode exigir caminhos sem
espaços; para uma nova síntese, prefira o gerador com saída temporária nova.

11. Auditoria e integridade
---------------------------

audit_summary.json (audit_summary.json) registra a aprovação das verificações:

1. Parâmetros e versões iguais à referência, exceto RF.
2. Modelo/scaler identificados por SHA-256.
3. Três densas presentes, com RF solicitado e Strategy=Latency; RF herdado nas demais camadas.
4. parameters.h idêntico à referência após normalizar somente os valores de RF.
5. defines.h e arquivos de pesos idênticos à referência.
6. 29/30 acertos e concordância top-1 de 100%.
7. 30 resultados C e RTL correspondentes à referência e co-simulação Pass.
8. RF1 com CSynthesisReport idêntico ao original.
9. ZIPs íntegros e Verilog empacotado idêntico ao RTL da síntese, byte a byte.

A auditoria funcional não substitui a síntese Vivado. A coleta OOC agora cobre
todos os cinco RTLs; place-and-route e validação física continuam pendentes. A auditoria usa a referência em ../../MLP/hls4ml/; ao transportar apenas
esta pasta, preserve também a referência para repetir essa comparação.

Hashes dos cinco ZIPs:

6bca2cf4f86faf5c1e22d30cae5b45c9be5507583078bd00db3bd14a395c9bc5  RF1/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip
c4947d44dde5a52dc2f35228640ffb24580da09dedf5fe694305192dece11273  RF4/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip
a5fc3ed0f6cc347d18b4380f8390cc5261b6b95ce4d2eca8836347b38879bce9  RF8/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip
b34b2e595686d9ccce03939961f289a0b6f196e66eb23d8779bef95fb7d7c6ac  RF16/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip
8de742aa196de89209f856da1ad72e1922b95de3a0d8fb39ac31b59b67a989ff  RF32/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip

IP_SHA256SUMS.txt (IP_SHA256SUMS.txt) cobre os cinco IPs.
SHA256SUMS_ALL.txt (SHA256SUMS_ALL.txt) registra um snapshot dos arquivos regulares
desta pasta, incluindo modelos, scripts, README, logs e projetos. Exclui somente
seu próprio arquivo. Alterações posteriores legítimas exigem atualizar o snapshot.
Checksums detectam alterações; não substituem um backup externo.

12. Como conferir e reproduzir
------------------------------

Comandos a partir de H1/MLP:

Conferir os IPs e o snapshot completo, sem executar síntese.
============================================================
sha256sum -c IP_SHA256SUMS.txt
sha256sum -c SHA256SUMS_ALL.txt

Revalidar a campanha funcional e regenerar a base HLS/RTL do CSV.
=================================================================
python3 audit_h1.py

Reexecutar a síntese Vivado OOC para todos os RFs.
==================================================
for rf in 1 4 8 16 32; do
  /opt/Xilinx/Vivado/2024.2/bin/vivado -mode batch -source run_vivado_ooc.tcl -tclargs "$PWD/RF${rf}" "RF${rf}"
done

Recolocar as colunas Vivado no CSV após a auditoria.
====================================================
python3 collect_vivado.py
python3 write_readme_txt.py

Executar a série HLS; os cinco RFs já concluídos são preservados.
=================================================================
python3 run_h1.py

O executor percorre RF1/4/8/16/32, cria pastas ausentes e ignora as marcadas como
concluídas. Recusa sobrescrever uma pasta não vazia sem conclusão. Ele não refaz
uma campanha concluída nem verifica sua integridade ao ignorá-la; para isso use
a auditoria e os checksums. Em caso de falha, o log e o projeto parcial ficam
salvos na pasta do RF para diagnóstico.

Para gerar uma nova cópia independente, sem sobrescrever os projetos desta campanha:

A partir de H1/MLP; a saída nova não contém espaços.
====================================================
mlp_h1_rebuild_dir=$(mktemp -d /tmp/h1_mlp_rf16_rebuild.XXXXXX)
python3 generate_mlp_h1.py --reuse-factor 16 --output-dir "$mlp_h1_rebuild_dir" --cosim

Troque 16 por 1, 4, 8 ou 32 conforme necessário. O uso de --cosim é necessário
para repetir todas as etapas desta campanha. --skip-build executa somente
conversão, compilação C++ e validação numérica, sem síntese/exportação.
O gerador direto pode escrever em um diretório existente: use sempre uma saída
nova. Novos ZIPs podem ter hashes diferentes por metadados de empacotamento;
compare também configurações, RTL e resultados, não apenas o hash do contêiner.

13. Diagnósticos preservados nos logs
-------------------------------------

- config_array_partition -maximum_size não é reconhecido pelo Vitis 2024.2:
  o Tcl gerado já envolve essa chamada em catch, e a execução continuou.
- O scaler foi serializado com scikit-learn 1.7.2 e carregado em 1.5.1,
  gerando aviso de versão. O runtime e o arquivo foram preservados como no fluxo
  anterior; a equivalência numérica foi conferida no holdout desta campanha.
- Há mensagens de inicialização CUDA, apesar de a GPU estar desabilitada para o fluxo.
- Há avisos de pragmas antigos, argumentos não usados e ambiente do compilador.

Essas mensagens permanecem nos logs. Não foram aplicadas mudanças diferentes por
RF para suprimi-las. A conclusão se baseia nos relatórios, co-simulação, saídas
registradas e IPs verificados, e não apenas no código de saída do executor.

14. O que falta coletar para completar a avaliação física da H1
---------------------------------------------------------------

Etapa/métrica                                    | Estado desta campanha                        
-------------------------------------------------+----------------------------------------------
Conversão e validação C++                        | Concluídas para os cinco RFs                 
C simulation e síntese HLS                       | Concluídas para os cinco RFs                 
Co-simulação RTL e exportação IP                 | Concluídas para os cinco RFs                 
Recursos pós-síntese Vivado/OOC                  | Concluídos para os cinco RFs                 
WNS/TNS OOC a 100 MHz                            | Concluídos; todos com WNS positivo e TNS zero
Place-and-route e clock físico                   | Não coletados                                
Wrapper, block design, bitstream/HWH da série    | Não gerados nesta H1                         
Acurácia na ZCU104                               | Não medida nesta série                       
Latência inference-only/end-to-end e FPS físicos | Não medidos                                  
Média, mediana, p95 e desvio temporal na placa   | Não coletados                                
Potência idle/ativa e energia por inferência     | Não medidas                                  

A continuidade é selecionar os RFs a implementar, preservar o clock e o método
de integração/medição entre eles, fechar timing e validar na placa as mesmas
entradas antes do benchmark. Os artefatos de outras campanhas permanecem como
referência; seus resultados físicos não foram atribuídos a esta série H1.
