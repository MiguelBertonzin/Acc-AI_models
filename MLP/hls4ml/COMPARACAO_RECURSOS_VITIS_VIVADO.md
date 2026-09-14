# Comparação de recursos — Vitis HLS × Vivado

## Configuração

MLP Iris 4–8–8–3 com softmax, `ap_fixed<16,6>`, `ReuseFactor=1`,
`Latency`, `io_parallel`, ZCU104 (`xczu7ev-ffvc1156-2-e`) e clock alvo de
100 MHz.

## Resultado

| Recurso | Vitis HLS | Vivado pós-síntese OOC | Vivado pós-rota OOC | Diferença HLS → pós-rota |
|---|---:|---:|---:|---:|
| LUT | 4.058 | 2.098 | 1.944 | −2.114 (−52,09%) |
| FF/registradores | 413 | 377 | 377 | −36 (−8,72%) |
| DSP | 108 | 108 | 108 | 0 |
| BRAM_18K | 3 | 3 | 3 | 0 |
| URAM | 0 | 0 | 0 | 0 |

A utilização física pós-rota do núcleo foi:

- 330 CLBs de 28.800: 1,15%;
- 1.944 LUTs de 230.400: 0,84%;
- 377 registradores de 460.800: 0,08%;
- 108 DSP48E2 de 1.728: 6,25%;
- 3 RAMB18E2 de 624: 0,48%;
- 151 CARRY8 de 28.800: 0,52%;
- nenhuma URAM.

## Timing pós-rota

| Métrica | Resultado |
|---|---:|
| Clock | 100 MHz / 10 ns |
| WNS de setup | +2,886 ns |
| TNS | 0 ns |
| WHS de hold | +0,064 ns |
| Caminho crítico aproximado | 7,114 ns |
| Redes não roteadas | 0 |
| Resultado | Timing atendido |

A síntese HLS estimou período de 6,586 ns e Fmax de aproximadamente 151,85 MHz.
O pós-rota OOC indica margem de 2,886 ns para 100 MHz; a frequência equivalente
do caminho crítico é aproximadamente 140,6 MHz. Esta frequência é apenas uma
referência OOC: a integração completa com PS, AXI, clocks e resets pode alterar
o timing.

## Interpretação

DSP e BRAM coincidiram exatamente entre HLS e Vivado. A principal diferença foi
em LUTs: o HLS fez uma estimativa conservadora de 4.058, enquanto o Vivado
mapeou 2.098 após síntese e 1.944 após otimizações físicas/place-and-route.
Assim, para relatar o custo isolado do núcleo, o valor pós-rota de 1.944 LUTs é
o mais fiel disponível neste estágio.

A execução foi out-of-context, portanto não contabiliza o wrapper AXI, o Zynq
MPSoC, interconexões, reset, clocking nem demais componentes do block design.
Esses elementos deverão ser medidos novamente depois da integração completa.
Também não foram definidos atrasos externos dos ports paralelos; o timing
interno está fechado, mas o timing final deve vir do projeto completo.

O DRC pós-rota não encontrou falhas de roteamento. Ele registrou avisos de
pipelining dos 108 DSPs, coerentes com a arquitetura de latência mínima, mas o
núcleo ainda atendeu 100 MHz com margem positiva.

## Evidências

- `reports/vivado/utilization_post_synth_ooc.rpt`;
- `reports/vivado/utilization_post_route.rpt`;
- `reports/vivado/timing_post_route.rpt`;
- `reports/vivado/drc_post_route.rpt`;
- `reports/vivado/mlp_iris_post_synth_ooc.dcp`;
- `reports/vivado/mlp_iris_post_route.dcp`;
- `reports/resource_comparison.json`;
- `mlp_iris_prj/solution1/syn/report/mlp_iris_csynth.rpt`.
