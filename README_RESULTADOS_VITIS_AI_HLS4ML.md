# Resultados dos fluxos Vitis AI e hls4ml

Este documento consolida os resultados obtidos com MLP/Iris, LeNet/MNIST e ResNet8/CIFAR-10 na plataforma AMD/Xilinx ZCU104. Foram considerados dois métodos de implementação: Vitis AI, com modelos INT8 executados na DPU `DPUCZDX8G_ISA1_B4096`, e hls4ml, com aceleradores dedicados em ponto fixo.

As métricas devem ser interpretadas conforme a fronteira de cada ensaio. `Inference-only` mede o caminho de execução com a entrada preparada; `ponta a ponta` acrescenta pré-processamento e pós-processamento; `saturado` caracteriza a maior carga sustentada do arranjo avaliado. Nenhum outlier foi removido das campanhas finais.

## Síntese geral

| Modelo | Fluxo | Acurácia na ZCU104 | Menor latência de inferência | Melhor vazão de inferência | Maior vazão saturada |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP | Vitis AI | 96,67% | 0,1767 ms, T1 | 4.894,83 inf/s, T2 | 10.470,98 inf/s, T3 |
| MLP | hls4ml | 96,67% | 0,0525 ms | 7.251,48 inf/s | 17.507,64 inf/s |
| LeNet | Vitis AI | 98,95% | 0,2256 ms, T1 | 5.291,79 inf/s, T2 | 9.292,01 inf/s, T3 |
| LeNet | hls4ml | 98,99% | 0,3700 ms | 2.791,62 inf/s | 2.784,80 inf/s |
| ResNet8 | Vitis AI | 73,96% | 0,2526 ms, T1 | 6.707,23 inf/s, T3 | 12.814,81 inf/s, T4 |
| ResNet8 | hls4ml | 74,92% | 0,8493 ms | 899,21 inf/s | 1.200,94 inf/s |

Nos resultados Vitis AI, `T1` a `T4` indicam a quantidade de threads e runners. Latência e vazão são métricas distintas; a configuração com menor latência não é necessariamente a de maior vazão.

## MLP — Vitis AI

O modelo 4–8–8–3 foi quantizado por PTQ INT8 sem perda de acurácia. O modelo float, a simulação quantizada e a DPU classificaram corretamente 29 das 30 amostras de teste. A concordância da DPU com a referência quantizada foi integral.

| Cenário | T | Latência média (ms) | Vazão (inf/s) | Potência ativa (W) | Potência dinâmica (W) | Energia total (mJ/inf) | Energia dinâmica (mJ/inf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inferência | 1 | 0,1767 | 3.462,66 | 14,6507 | 0,2147 | 4,2311 | 0,0620 |
| Inferência | 2 | 0,2717 | 4.894,83 | 14,8138 | 0,3769 | 3,0264 | 0,0770 |
| Inferência | 3 | 0,3772 | 4.231,66 | 14,7621 | 0,3215 | 3,4885 | 0,0760 |
| Inferência | 4 | 0,6005 | 4.060,09 | 14,7875 | 0,3505 | 3,6422 | 0,0863 |
| Ponta a ponta | 1 | 0,6935 | 1.412,79 | 14,6690 | 0,2326 | 10,3829 | 0,1646 |
| Ponta a ponta | 2 | 1,1979 | 1.597,40 | 14,6892 | 0,2530 | 9,1957 | 0,1584 |
| Ponta a ponta | 3 | 1,8070 | 1.511,09 | 14,7063 | 0,2719 | 9,7323 | 0,1799 |
| Ponta a ponta | 4 | 1,8479 | 1.570,83 | 14,6878 | 0,2563 | 9,3503 | 0,1632 |
| Saturado | 1 | 0,1694 | 5.484,88 | 14,6610 | 0,2287 | 2,6730 | 0,0417 |
| Saturado | 2 | 0,2078 | 8.917,20 | 14,8257 | 0,3896 | 1,6626 | 0,0437 |
| Saturado | 3 | 0,2641 | 10.470,98 | 14,9041 | 0,4691 | 1,4234 | 0,0448 |
| Saturado | 4 | 0,3654 | 9.639,87 | 14,8774 | 0,4415 | 1,5433 | 0,0458 |

Fonte: `MLP/Vitis AI/results/zcu104_final_20260909T204251Z/`.

## MLP — hls4ml

Foi empregada precisão `ap_fixed<16,6>`, estratégia `Latency`, `ReuseFactor=1`, interface paralela e Softmax em hardware. A síntese HLS estimou cinco ciclos e a medição física registrou seis ciclos, equivalentes a 60 ns no núcleo a 100 MHz.

| Cenário | Latência média (ms) | P95 (ms) | Vazão (inf/s) | Potência ativa (W) | Potência dinâmica (W) | Energia total (mJ/inf) | Energia dinâmica (mJ/inf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inferência | 0,0525 | 0,0529 | 7.251,48 | 10,1522 | 0,1974 | 1,4162 | 0,0275 |
| Ponta a ponta | 0,2247 | 0,2454 | 4.558,03 | 10,1530 | 0,2017 | 2,2463 | 0,0446 |
| Saturado | 0,0542 | 0,0545 | 17.507,64 | 10,1327 | 0,1784 | 0,5526 | 0,0097 |

| Item de implementação | Resultado |
| --- | ---: |
| Acurácia HLS/FPGA | 96,67% |
| Concordância FPGA × HLS | 100% |
| LUT pós-route | 2.633 (1,14%) |
| DSP48E2 pós-route | 108 (6,25%) |
| WNS pós-route | +3,098 ns |
| Frequência do sistema | 100 MHz |

Fontes: `MLP/hls4ml/RESULTADOS_IP.md` e `MLP/hls4ml/resultados_zcu104/`.

## LeNet — Vitis AI

O modelo foi convertido para Keras 2.12, quantizado em INT8 e compilado para a DPU. Nas 10.000 imagens de teste, a referência quantizada e a execução na DPU obtiveram 98,95% de acurácia, com 100% de concordância.

| Cenário | T | Latência média (ms) | Vazão (inf/s) | Potência ativa (W) | Potência dinâmica (W) | Energia total (mJ/inf) | Energia dinâmica (mJ/inf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inferência | 1 | 0,2256 | 3.066,89 | 14,7674 | 0,2803 | 4,8151 | 0,0914 |
| Inferência | 2 | 0,2532 | 5.291,79 | 14,9877 | 0,4986 | 2,8323 | 0,0942 |
| Inferência | 3 | 0,4117 | 4.487,86 | 14,9768 | 0,4856 | 3,3372 | 0,1082 |
| Inferência | 4 | 0,6122 | 4.351,15 | 14,9726 | 0,4801 | 3,4411 | 0,1103 |
| Ponta a ponta | 1 | 0,6506 | 1.436,19 | 14,7476 | 0,2531 | 10,2685 | 0,1762 |
| Ponta a ponta | 2 | 1,2062 | 1.565,82 | 14,8271 | 0,3312 | 9,4692 | 0,2115 |
| Ponta a ponta | 3 | 1,8207 | 1.570,13 | 14,8436 | 0,3460 | 9,4538 | 0,2204 |
| Ponta a ponta | 4 | 2,4726 | 1.495,12 | 14,8413 | 0,3447 | 9,9265 | 0,2305 |
| Saturado | 1 | 0,2217 | 4.167,43 | 14,7910 | 0,2969 | 3,5492 | 0,0712 |
| Saturado | 2 | 0,2353 | 7.750,04 | 15,0504 | 0,5533 | 1,9420 | 0,0714 |
| Saturado | 3 | 0,2818 | 9.292,01 | 15,1845 | 0,6940 | 1,6341 | 0,0747 |
| Saturado | 4 | 0,4199 | 8.281,59 | 15,1343 | 0,6434 | 1,8275 | 0,0777 |

Fonte: `LeNet/Vitis AI/results/zcu104_physical_20260908/`.

## LeNet — hls4ml

A implementação utiliza `ap_fixed<22,12>`, E/S em fluxo e frequência de 100 MHz. A FPGA reproduziu integralmente a referência fixa do hls4ml nas 10.000 imagens: 98,99% de acurácia, 100% de concordância e nenhum erro nos 100.000 logits comparados.

| Cenário | Latência média (ms) | Vazão (inf/s) | Potência ativa (W) | Potência dinâmica (W) | Energia total (mJ/inf) | Energia dinâmica (mJ/inf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DMA + FPGA | 0,3700 | 2.791,62 | 10,6032 | 0,2770 | 3,8062 | 0,0995 |
| Ponta a ponta | 1,0970 | 907,15 | 10,5442 | 0,2369 | 11,5453 | 0,2594 |
| Saturado serial | — | 2.784,80 | 10,6009 | 0,2867 | 3,8068 | aproximadamente 0,1030 |

| Recurso do sistema completo | Utilização |
| --- | ---: |
| CLB LUTs | 154.485 (67,05%) |
| Registradores | 96.655 (20,98%) |
| DSP48E2 | 746 (43,17%) |
| BRAM equivalentes | 125 (40,06%) |
| CLBs | 28.154 (97,76%) |

O projeto fechou timing em 100 MHz. A estimativa HLS do núcleo foi de 9.412 a 9.460 ciclos, equivalente a 94,12–94,60 µs; essa estimativa não possui a mesma fronteira da latência DMA + FPGA.

Fontes: `LeNet/hls4ml/README_FINAL_LENET_ZCU104_HLS4ML_COLETA_COMPLETA.md` e `LeNet/hls4ml/resultados_zcu104/`.

## ResNet8 — Vitis AI

A quantização PTQ INT8 reduziu a acurácia de 74,90% para 73,96%. A DPU reproduziu exatamente a referência quantizada nas 10.000 imagens, sem divergências.

| Cenário | Configuração | Latência média | Vazão | Potência ativa | Energia dinâmica |
| --- | --- | ---: | ---: | ---: | ---: |
| Inferência | 1 thread, 10.000 imagens | 0,2526 ms | 3.092,50 img/s | 15,2359 W | 0,2415 mJ/img |
| Inferência | 3 threads, 10.000 imagens | 0,3406 ms | 6.707,23 img/s | 16,2043 W | 0,2559 mJ/img |
| Ponta a ponta | 1 thread, 10.000 imagens | 0,6816 ms | 1.444,74 img/s | 14,9590 W | 0,3315 mJ/img |
| Ponta a ponta | 3 threads, 10.000 imagens | 1,4291 ms | 2.081,76 img/s | 15,2784 W | 0,3902 mJ/img |
| Saturado | 1 thread | 0,2467 ms | 3.991,11 img/s | 15,3896 W | 0,2287 mJ/img |
| Saturado | 4 threads | 0,3071 ms | 12.814,81 img/s | 17,2803 W | 0,2232 mJ/img |

O XModel possui 249.251 bytes e foi compilado para dois núcleos DPU a 300 MHz. A coleta completa está preservada em `resnet8/Vitis AI/resnet8_final_v1_RESULTS_2026-09-01.tar.gz`; os relatórios de conversão, quantização e compilação permanecem em `resnet8/Vitis AI/reports/`.

## ResNet8 — hls4ml

A implementação dedicada utiliza `ap_fixed<22,12>`, frequência de 100 MHz e saída sem Softmax. A validação nas 10.000 imagens do CIFAR-10 obteve 74,92% de acurácia.

| Cenário | Latência média (ms) | P95 (ms) | Vazão (img/s) | Potência ativa (W) | Potência dinâmica (W) | Energia total (mJ/img) | Energia dinâmica (mJ/img) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inferência | 0,8493 | 0,8621 | 899,21 | 12,2283 | 1,6189 | 13,4994 | 1,7871 |
| Ponta a ponta | 1,5977 | 1,6155 | 620,70 | 11,8088 | 1,1818 | 19,0992 | 1,9114 |
| Saturado | 0,8326 | 0,8452 | 1.200,94 | 12,6580 | 2,0410 | 10,7266 | 1,7295 |

| Recurso do sistema completo | Utilização |
| --- | ---: |
| CLB LUTs | 178.658 (77,54%) |
| BRAM | 85,5 blocos (27,40%) |
| DSP48E2 | 1.174 (67,94%) |
| WNS | +0,122 ns |

O roteamento foi concluído sem erros e as restrições de timing foram atendidas a 100 MHz. Os dados completos estão em `resnet8/hls4ml/Resultados_ZCU104/`.

## Observações para comparação

- As implementações Vitis AI utilizam dois núcleos DPU e podem executar vários runners; o hls4ml utiliza um acelerador dedicado operado de forma serial nos ensaios apresentados.
- A latência estimada pelo HLS não deve ser comparada diretamente à latência medida por software, pois esta inclui transferências e sincronização.
- As medições de potência correspondem ao rail disponível em cada imagem da ZCU104 e não representam exclusivamente a lógica do acelerador.
- Os cenários saturados não incluem o mesmo pré-processamento do cenário ponta a ponta e devem ser apresentados separadamente.
