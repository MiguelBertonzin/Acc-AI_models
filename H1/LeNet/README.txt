H1 — LeNet/MNIST: RF global solicitado 32, 64 e 128
===================================================

Campanha HLS iniciada em 2026-09-09 e análise Vivado concluída em 2026-09-10.
Os três projetos foram validados em C++ nas 10.000 imagens MNIST, sintetizados pelo
Vitis HLS, co-simulados em RTL em 10 imagens representativas, exportados como IP e
sintetizados novamente no Vivado 2024.2 em modo Out-of-Context (OOC). Auditoria
funcional e de integridade: passed.

Resultado principal
-------------------

A variação do RF não alterou os resultados numéricos: os três projetos atingiram
98,99% de acurácia HLS e 99,99% de concordância com o Keras. O custo foi trocado
entre paralelismo e latência:

- RF32: maior vazão, mas usa 103,11% das LUTs. Não cabe na ZCU104.
- RF64: usa 57,42% das LUTs e 40,05% dos DSPs, com aproximadamente 2.105,71 imagens/s. É o melhor ponto de equilíbrio desta H1 para iniciar a integração.
- RF128: usa 35,11% das LUTs e 20,02% dos DSPs, com aproximadamente 1.102,66 imagens/s. Oferece maior folga para o restante do sistema.

Essas vazões são derivadas do intervalo observado na co-simulação RTL a 100 MHz;
não são medições na placa. A análise Vivado desta campanha termina na netlist
otimizada pós-síntese. Não houve place-and-route, bitstream ou benchmark físico.

Objetivo e relação com o fluxo anterior
---------------------------------------

Variar somente a solicitação global de ReuseFactor, preservando modelo, pesos,
precisão, estratégia, I/O, clock e demais parâmetros da LeNet anterior.
A referência foi ../../LeNet/hls4ml/builds/cap64_q22_12_rate_balanced/, com perfil
por camada 5/50/64/60/42. Esse perfil é uma referência numérica e de configuração;
não corresponde a RF64 global e não deve ser misturado com a H1 como se fosse.

Foram removidos todos os overrides LayerName.*.ReuseFactor, para que a alteração
em Model.ReuseFactor seja herdada. A configuração solicitada está em
RF*/hls_config_requested.json; a configuração efetiva em effective_layers.json
e reuse_resolved.json/csv. Os valores válidos por camada estão registrados.

Configuração fixa
-----------------

Parâmetro                           | Valor                                                           
------------------------------------+-----------------------------------------------------------------
Modelo fonte                        | lenet_mnist_final.h5, FP32, sem quantização/QAT no arquivo Keras
Saída                               | 10 logits; sem Softmax; classe por argmax                       
Backend                             | Vitis                                                           
FPGA                                | xczu7ev-ffvc1156-2-e, ZCU104                                    
Clock                               | 10 ns / 100 MHz                                                 
Incerteza                           | 27%, padrão do backend                                          
Strategy                            | Resource                                                        
IOType                              | io_stream                                                       
Convoluções                         | LineBuffer, ParallelizationFactor=1                             
Precisão                            | ap_fixed<22,12,AP_RND_CONV,AP_SAT>                              
Trace de camadas                    | desabilitado                                                    
Otimização FIFO                     | não executada; profundidades preservadas                        
hls4ml                              | 1.3.0                                                           
Vitis HLS / Vivado OOC              | 2024.2                                                          
Python / TensorFlow / Keras / NumPy | 3.12.7 / 2.21.0 / 3.13.2 / 1.26.4                               

A precisão possui 22 bits totais e 12 bits inteiros incluindo sinal, com 10 bits
fracionários. Pesos, biases, acumuladores e resultados preservam os tipos da
referência. A entrada input é renomeada para input_layer, como no fluxo anterior,
sem alterar seus pesos ou significado.

Arquitetura: entrada 28×28×1; Conv 5×5/6 + ReLU; MaxPool 2×2; Conv 5×5/16 + ReLU;
MaxPool 2×2; Flatten 256; Dense120 + ReLU; Dense84 + ReLU; Dense10 linear.
Total: 44.426 parâmetros, incluindo biases. Não houve retreinamento.

RF solicitado e efetivo
-----------------------

RF solicitado | conv1 | conv2 | dense1 | dense2 | output
--------------+-------+-------+--------+--------+-------
32            | 25    | 30    | 32     | 30     | 28    
64            | 75    | 75    | 64     | 60     | 84    
128           | 150   | 150   | 128    | 120    | 168   

O backend aproxima cada RF inválido ao valor válido mais próximo. Por isso, esta
H1 avalia o efeito do parâmetro global solicitado, não a aplicação de um mesmo
RF efetivo a todas as camadas. Todas as densas e convoluções mantiveram Resource.
A solicitação também é herdada pelas ativações/pooling, sem otimizações manuais
por camada.

Por exemplo, em RF128 a conv1 usa RF150, embora seja pequena. Um RF alto pode
serializar desnecessariamente uma camada e criar um gargalo. A H2 poderá comparar
perfis por camada; eles não foram introduzidos nesta série.

FIFOs preservadas, em palavras do respectivo stream:

Stream                                               | Profundidade
-----------------------------------------------------+-------------
layer2_out / layer3_out                              | 576 cada    
layer4_out                                           | 144         
layer5_out / layer6_out                              | 64 cada     
layer7_out                                           | 16          
layer9_out / layer10_out / layer11_out / layer12_out | 1 cada      

O corpo C++ do top, incluindo as FIFOs, foi comparado com a referência anterior.

Dados e validação numérica
--------------------------

O arquivo data/mnist_test_uint8.npz foi copiado do fluxo anterior. Contém as
10.000 imagens de teste e rótulos. Pré-processamento: uint8 → float32, divisão por
255 e adição do eixo de canal, produzindo (10000,28,28,1). Não foi usado dado de
calibração nem treinamento nesta campanha.

Keras é avaliado com batch 256 no host, como na referência. Isso não configura um
batch de hardware; o acelerador processa transações de imagens individuais.

Critérios numéricos preservados: concordância Keras/HLS de pelo menos 99,9% e perda
de no máximo 10 acertos frente ao Keras nas 10.000 imagens. Cada configuração salva
logits, classes, rótulos, matriz de confusão e intervalo de Wilson de 95%.

RF solicitado | Acurácia HLS, 10.000 imagens | Concordância Keras/HLS | MAE dos logits | Erro máximo | RTL, 10 imagens
--------------+------------------------------+------------------------+----------------+-------------+----------------
32            | 98.99%                       | 99.99%                 | 0.015648982    | 0.098563671 | Pass           
64            | 98.99%                       | 99.99%                 | 0.015648982    | 0.098563671 | Pass           
128           | 98.99%                       | 99.99%                 | 0.015648982    | 0.098563671 | Pass           

Keras FP32: 98,98%. Os logits C++ são idênticos entre os três RFs:
True. Igualdade exata com os logits da referência
anterior, na ordem RF32/RF64/RF128: [True, True, True].
A concordância com a referência HLS é diferente da concordância com o Keras FP32.

Co-simulação RTL
----------------

Subconjunto determinístico: primeira imagem de cada classe de 0 a 9, índices
zero-based [3,2,1,18,4,8,11,0,61,7]. É o mesmo conjunto nos três RFs, sem escolha
orientada pelos resultados. C simulation e RTL recebem as mesmas dez entradas.

A auditoria compara CSimResults com CosimResults e confere sua correspondência
com os logits C++ das imagens selecionadas. A comparação entre logs C/RTL é exata;
contra o array C++ usa tolerância absoluta 1e-4, para a impressão decimal dos logs.
A co-simulação não cobre as 10.000 imagens: elas foram avaliadas em C++.

RF solicitado | Latência RTL (ciclos, min–max) | Intervalo RTL (ciclos, min–max)
--------------+--------------------------------+--------------------------------
32            | 18844–18844                    | 18690–18690                    
64            | 47806–47806                    | 47490–47490                    
128           | 91291–91291                    | 90690–90690                    

O Tcl efetivo segue os padrões gerados pelo hls4ml, incluindo a remoção de
log_wave -r / antes de executar XSIM. O Trace de camadas continua desabilitado.
O manifest do RF32 contém uma nota de correção descritiva: a compilação regenerou
o Tcl, portanto uma tentativa preliminar de alterar tracing não se aplicou.

Síntese HLS
-----------

RF solicitado | Latência HLS (ciclos, min–max) | II HLS (ciclos, min–max) | DSP  | LUT    | FF    | BRAM_18K | URAM | Período estimado (ns)
--------------+--------------------------------+--------------------------+------+--------+-------+----------+------+----------------------
32            | 24289–24317                    | 3138–24306               | 2819 | 486549 | 96728 | 947      | 0    | 10.818               
64            | 63438–63511                    | 3138–63506               | 1379 | 264142 | 75266 | 507      | 0    | 7.273                
128           | 122164–122312                  | 3138–122306              | 687  | 159871 | 72621 | 296      | 0    | 7.191                

Clock solicitado: 10 ns para todos. A latência em tempo de referência pode ser
calculada por ciclos × 10 ns; isso não é uma medição na placa. Em io_stream,
identifique o escopo do II: laços/pixels/camadas não devem ser confundidos com
intervalos entre imagens completas. Os valores acima vêm do relatório do top;
a tabela RTL mantém separadamente os intervalos observados no testbench.

Metodologia da coleta Vivado
----------------------------

Cada caso usa exatamente o RTL produzido pelo respectivo projeto Vitis HLS em
RF*/lenet_mnist_cap64_hls_prj/solution1/syn/verilog/. O script
run_vivado_ooc.tcl executa synth_design para o top lenet_mnist_cap64_hls e a
parte xczu7ev-ffvc1156-2-e, cria o clock ap_clk de 10 ns, executa opt_design e
gera relatórios de utilização, timing e metodologia, além do checkpoint DCP.

A síntese foi executada separadamente para RF32, RF64 e RF128. Os relatórios ficam
em RF*/vivado_ooc/. O RF128, sintetizado primeiro, teve o relatório de timing
regenerado a partir do checkpoint com a mesma restrição de 10 ns; isso não altera
a netlist nem a utilização de recursos.

O Vitis HLS estima recursos antes do mapeamento completo. O Vivado resolve as
operações e memórias em primitivas do dispositivo, elimina lógica redundante e
aplica otimizações. Por isso os valores Vivado são mais representativos que as
estimativas HLS, mas continuam anteriores ao posicionamento e roteamento.

As capacidades usadas são 1728 DSP, 230400 CLB LUTs, 460800 CLB Registers,
312 tiles de BRAM (624 blocos equivalentes de 18 Kb) e 96 URAM. Para comparar
memória com o HLS, cada RAMB36 foi contado como dois blocos de 18 Kb e somado aos
RAMB18. Essa conversão preserva a capacidade total, embora não descreva restrições
de empacotamento dentro de cada tile.

Tabela completa: HLS, RTL e Vivado
----------------------------------

RF global | RF efetivo C1/C2/D1/D2/Out | Acc. HLS | Lat. HLS      | II HLS      | Lat. RTL | Intervalo RTL | Lat. RTL µs | Imagens/s teóricas | DSP HLS | LUT HLS | FF HLS | BRAM18 HLS | URAM HLS | DSP Vivado    | LUT Vivado       | FF Vivado       | BRAM18 eq. Vivado | BRAM tiles | URAM Vivado | WNS ns | TNS ns | Cabe nos recursos?
----------+----------------------------+----------+---------------+-------------+----------+---------------+-------------+--------------------+---------+---------+--------+------------+----------+---------------+------------------+-----------------+-------------------+------------+-------------+--------+--------+-------------------
32        | 25/30/32/30/28             | 98.99%   | 24289–24317   | 3138–24306  | 18844    | 18690         | 188.44      | 5350.45            | 2819    | 486549  | 96728  | 947        | 0        | 1412 (81.71%) | 237576 (103.11%) | 101642 (22.06%) | 403 (64.58%)      | 201.5      | 0           | +4.606 | 0.000  | Não               
64        | 75/75/64/60/84             | 98.99%   | 63438–63511   | 3138–63506  | 47806    | 47490         | 478.06      | 2105.71            | 1379    | 264142  | 75266  | 507        | 0        | 692 (40.05%)  | 132299 (57.42%)  | 79750 (17.31%)  | 221 (35.42%)      | 110.5      | 0           | +4.847 | 0.000  | Sim               
128       | 150/150/128/120/168        | 98.99%   | 122164–122312 | 3138–122306 | 91291    | 90690         | 912.91      | 1102.66            | 687     | 159871  | 72621  | 296        | 0        | 346 (20.02%)  | 80890 (35.11%)   | 73523 (15.96%)  | 133 (21.31%)      | 66.5       | 0           | +4.990 | 0.000  | Sim               

Os valores Vivado são da netlist otimizada em síntese OOC para
xczu7ev-ffvc1156-2-e; entre parênteses aparece a ocupação do dispositivo. O clock
foi restringido a 10 ns/100 MHz. WNS positivo e TNS zero indicam que o timing de
síntese foi atendido. Essa etapa mapeia o RTL em primitivas reais do FPGA, mas ainda
não inclui efeitos de posicionamento, roteamento, block design, PS ou DMA.

RF  | DSP HLS → Vivado (% da estimativa) | LUT HLS → Vivado (% da estimativa) | FF HLS → Vivado (% da estimativa) | BRAM18 HLS → equivalente Vivado (% da estimativa) | URAM HLS → Vivado
----+------------------------------------+------------------------------------+-----------------------------------+---------------------------------------------------+------------------
32  | 2819 → 1412 (50.09%)               | 486549 → 237576 (48.83%)           | 96728 → 101642 (105.08%)          | 947 → 403 (42.56%)                                | 0 → 0            
64  | 1379 → 692 (50.18%)                | 264142 → 132299 (50.09%)           | 75266 → 79750 (105.96%)           | 507 → 221 (43.59%)                                | 0 → 0            
128 | 687 → 346 (50.36%)                 | 159871 → 80890 (50.6%)             | 72621 → 73523 (101.24%)           | 296 → 133 (44.93%)                                | 0 → 0            

O RF32 usa 237576 LUTs, excedendo a capacidade em 7176 LUTs. Sua síntese OOC
termina e apresenta WNS positivo, mas esse timing é abstrato: uma netlist que excede
o dispositivo não pode ser posicionada. RF64 deixa livres 98101 LUTs, 1036 DSPs e
201,5 tiles de BRAM. RF128 deixa livres 149510 LUTs, 1382 DSPs e 245,5 tiles de
BRAM. Essas folgas ainda precisam acomodar interconexões, DMA, controle e demais
blocos do sistema.

Do RF32 ao RF128, o uso Vivado cai 65,95% em LUTs e 75,50% em DSPs, enquanto o
intervalo RTL cresce de 18690 para 90690 ciclos, redução de aproximadamente 79,39%
na vazão teórica. Do RF64 ao RF128, LUTs caem 38,86% e DSPs 50,00%, mas a vazão cai
47,64%. Como a acurácia permaneceu idêntica, a escolha entre RF64 e RF128 depende
da folga exigida pela integração e da meta de desempenho.

O HLS superestimou LUTs e DSPs em aproximadamente 2× nos três casos. Para FFs, o
Vivado ficou entre 101,24% e 105,96% da estimativa HLS. Isso mostra por que as duas
fontes devem permanecer na tabela: a estimativa HLS ajuda durante a exploração, e
o Vivado mostra o mapeamento efetivo da netlist. O período estimado HLS do RF32
foi 10,818 ns, enquanto a síntese Vivado reportou WNS positivo a 10 ns; os valores
vêm de estágios e modelos distintos e não substituem o timing pós-route.

IPs e uso no Vivado
-------------------

- RF32 — IP ZIP (RF32/lenet_mnist_cap64_hls_prj/solution1/impl/ip/xilinx_com_hls_lenet_mnist_cap64_hls_1_0.zip)
- RF64 — IP ZIP (RF64/lenet_mnist_cap64_hls_prj/solution1/impl/ip/xilinx_com_hls_lenet_mnist_cap64_hls_1_0.zip)
- RF128 — IP ZIP (RF128/lenet_mnist_cap64_hls_prj/solution1/impl/ip/xilinx_com_hls_lenet_mnist_cap64_hls_1_0.zip)

Cada ZIP está em RF*/lenet_mnist_cap64_hls_prj/solution1/impl/ip/, junto com
component.xml e o HDL descompactado. O top/identidade foram preservados da
referência: lenet_mnist_cap64_hls, VLNV xilinx.com:hls:lenet_mnist_cap64_hls:1.0.
O termo cap64 no nome é legado e não descreve o RF desta série. Selecione o RF
pelo diretório, manifesto e checksum. Registre somente um desses repositórios por
projeto Vivado para evitar ambiguidade de identidade.

A interface mantém streams de entrada/saída e controle do IP original. A integração
com PS/DMA depende do wrapper e protocolo apropriados; não se deve assumir que um
bitstream do perfil anterior serve para qualquer um destes novos RFs.

Arquivos e reprodução
---------------------

- generate_lenet_h1.py: gera um RF, valida 10.000 imagens e executa build completo.
- run_h1.py: executa RF32/64/128 sequencialmente e preserva campanhas concluídas.
- audit_h1.py: audita parâmetros, resultados, tipos, FIFOs e pacotes; gera CSV/JSON.
- run_vivado_ooc.tcl: sintetiza e otimiza cada RTL no Vivado em modo OOC.
- report_timing_from_dcp.tcl: reaplica a restrição de 10 ns a um checkpoint.
- collect_vivado.py: extrai utilização/timing e acrescenta colunas ao CSV.
- write_report.py: gera README.md e README.txt a partir da auditoria e do CSV Vivado.
- reference/: configuração e design da referência anterior.
- data/: MNIST usado na validação.
- RF*/hls_config_requested.json: configuração global sem overrides de RF.
- RF*/configuration_manifest.json: hashes, versões, RFs efetivos e condições.
- RF*/reuse_resolved.json/csv, effective_layers.json, fifo_depths.json: auditoria da conversão.
- RF*/cpp_validation.json, cpp_predictions_10000.npz: resultados numéricos completos.
- RF*/rtl_test_subset.json, tb_input_features.dat, tb_output_predictions.dat: teste RTL.
- RF*/build_report.json, run.log, run_status.json: relatórios, logs e execução.
- RF*/export_audit.json: integridade e equivalência do Verilog exportado.
- RF*/firmware/: C++, tipos, pesos e bibliotecas geradas.
- RF*/lenet_mnist_cap64_hls_prj/solution1/syn/report/: relatórios HLS originais.
- RF*/lenet_mnist_cap64_hls_prj/solution1/sim/report/: relatórios RTL.
- resultados_hls.csv: tabelassa consolidada com HLS, RTL, Vivado e deltas.
- README.md e README.txt: documentação completa em Markdown e texto simples.
- vivado_ooc_summary.json: resumo estruturado dos três relatórios Vivado.
- RF*/vivado_ooc/: utilização, timing, metodologia e checkpoint pós-síntese.
- audit_summary.json, IP_SHA256SUMS.txt: auditoria e checksums dos IPs.

A partir desta pasta, a campanha HLS e a auditoria funcional podem ser refeitas com:

python3 run_h1.py
python3 audit_h1.py

A síntese Vivado OOC dos três RTLs é executada separadamente:

for rf in 32 64 128; do
  /opt/Xilinx/Vivado/2024.2/bin/vivado -mode batch -source run_vivado_ooc.tcl -tclargs "$PWD/RF${rf}" "RF${rf}"
done
python3 collect_vivado.py
python3 write_report.py
sha256sum -c IP_SHA256SUMS.txt

audit_h1.py recria o CSV somente com dados HLS/RTL; por isso collect_vivado.py
deve ser executado depois dele para recolocar as colunas Vivado. O coletor não roda
a síntese: ele apenas valida e extrai os relatórios existentes.

O executor cria staging exclusivo /tmp/h1_lenet_rfN_*, pois o Vitis HLS não aceita
espaços no caminho, e copia todo o projeto para a pasta permanente do RF. Pastas
concluídas são ignoradas; pastas parciais não são sobrescritas automaticamente.
Os diretórios temporários não são necessários para importar os ZIPs. Uma nova
conversão deve usar diretório de saída novo e sem espaços:

lenet_h1_out=$(mktemp -d /tmp/h1_lenet_rebuild.XXXXXX)
python3 generate_lenet_h1.py --reuse-factor 64 --output-dir "$lenet_h1_out"

--skip-build permite só conversão/validação. O gerador direto pode escrever em
saída existente: use uma pasta nova. A auditoria e o gerador usam a referência
original em ../../LeNet/hls4ml/builds/cap64_q22_12_rate_balanced/; preserve-a ao
transportar o workspace. Os IPs exportados podem ser usados independentemente dela.

Limites e continuidade
----------------------

Não foram realizadas otimizações de precisão, FIFO, estratégia ou RF por camada.
Houve síntese lógica OOC no Vivado, mas não place-and-route, bitstream/HWH,
acurácia na placa, FPS físico, potência, energia ou estatísticas de repetibilidade.
Essas etapas devem manter o mesmo clock, integração e protocolo de medição entre
as configurações escolhidas. Recursos/latência de outras campanhas não foram
atribuídos aos IPs desta H1.

Avisos do backend sobre RF inválido correspondem aos ajustes efetivos registrados.
Os logs também podem conter avisos de pragmas legados e caminhos críticos. A chamada
config_array_partition -maximum_size não reconhecida é tratada pelo catch do Tcl
padrão; não altera diferencialmente os RFs. Falhas funcionais ou de exportação são
bloqueadas pelas verificações do gerador e da auditoria.
