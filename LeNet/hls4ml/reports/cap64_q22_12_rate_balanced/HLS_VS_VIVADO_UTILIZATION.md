# LeNet cap64 Q22.12 — Vitis HLS versus Vivado

## Resultado executivo

O RTL gerado pelo Vitis HLS foi sintetizado novamente pelo Vivado 2024.2 para a ZCU104 (`xczu7ev-ffvc1156-2-e`). A estimativa do Vitis HLS dizia que o projeto usaria 121% das LUTs e, portanto, não caberia. Depois do mapeamento tecnológico e de `opt_design`, o Vivado mediu 61,38% das LUTs. Assim, o IP isolado cabe nos recursos lógicos da FPGA na etapa pós-síntese.

Este resultado é mais fiel que a estimativa de C synthesis do HLS, mas ainda não é um resultado pós-place-and-route. A conclusão definitiva de implementação exige `place_design`, `route_design`, análise de timing e a inclusão da infraestrutura AXI/DMA/PS do block design.

## Ambiente e método

| Item | Valor |
|---|---|
| Data | 2026-09-04 |
| hls4ml | 1.3.0 |
| Vitis HLS / Vivado | 2024.2 |
| Dispositivo | `xczu7ev-ffvc1156-2-e` |
| Modelo | LeNet/MNIST, saída em logits, sem Softmax |
| Precisão | `ap_fixed<22,12,AP_RND_CONV,AP_SAT>` |
| RF por camada | `conv1=5`, `conv2=50`, `dense1=64`, `dense2=60`, `output=42` |
| Estratégia / I/O | `Resource` / `io_stream` AXI4-Stream |
| Clock solicitado ao HLS | 10 ns, 100 MHz, incerteza de 27% |
| Vivado | `synth_design`, seguido de `opt_design` e `report_utilization` |
| Estado do Vivado | `Optimized`, 0 erros na síntese |

O próprio fluxo Tcl gerado pelo hls4ml foi usado. A opção `vsynth=1` do `build_prj.tcl` chama o Vivado sobre `${project_name}_prj/solution1/syn/verilog`. O Tcl executado pelo Vivado foi:

```tcl
add_files ${project_name}_prj/solution1/syn/verilog
synth_design -top ${project_name} -part $part
opt_design -retarget -propconst -sweep -bram_power_opt -shift_register_opt
report_utilization -file vivado_synth.rpt
```

## Comparação de utilização

| Recurso | Disponível | Vitis HLS | HLS (%) | Vivado pós-síntese | Vivado (%) | Diferença Vivado−HLS |
|---|---:|---:|---:|---:|---:|---:|
| LUT | 230.400 | 280.382 | 121,69% | 141.417 | 61,38% | −138.965 (−49,56%) |
| FF / registradores | 460.800 | 76.276 | 16,55% | 81.051 | 17,59% | +4.775 (+6,26%) |
| DSP | 1.728 | 1.487 | 86,05% | 746 | 43,17% | −741 (−49,83%) |
| BRAM18 equivalente | 624 | 539 | 86,38% | 228 | 36,54% | −311 (−57,70%) |
| URAM | 96 | 0 | 0,00% | 0 | 0,00% | 0 |

Para comparar memória na mesma unidade, o resultado do Vivado foi normalizado como `2 × RAMB36 + RAMB18 = 2 × 112 + 4 = 228` blocos equivalentes de 18 Kb. O Vivado também apresenta esse total como 114 Block RAM Tiles de 36 Kb, ou 36,54% dos 312 tiles.

Recursos adicionais informados somente pelo Vivado:

| Recurso | Usado | Disponível | Utilização |
|---|---:|---:|---:|
| CARRY8 | 15.933 | 28.800 | 55,32% |
| LUT como memória | 1.282 | 101.760 | 1,26% |
| LUT como RAM distribuída | 204 | — | — |
| LUT como shift register | 1.078 | — | — |
| F7 Mux | 3.570 | 115.200 | 3,10% |
| F8 Mux | 1.296 | 57.600 | 2,25% |

## Por que os números mudaram tanto

- O Vitis HLS faz uma estimativa antes do mapeamento tecnológico completo. Ela é deliberadamente conservadora em diversas estruturas.
- O Vivado conhece os primitivos reais do UltraScale+ e consegue combinar LUTs, remover lógica, compartilhar ou simplificar operadores e empacotar memórias com mais precisão.
- O limite estrutural calculado para esta configuração era de 746 multiplicadores paralelos. O Vivado usou exatamente 746 DSP48E2, enquanto o HLS havia estimado 1.487 DSPs.
- As FIFOs e memórias inferidas foram finalmente mapeadas em 112 RAMB36E2, 4 RAMB18E2 e RAM distribuída. O resultado final não utilizou URAM.

Portanto, para decidir se o RTL cabe, deve-se usar o relatório do Vivado. O relatório do HLS continua útil para comparar rapidamente candidatos e para latência/intervalo estimados, mas não deve ser usado sozinho como veto quando indica uma ultrapassagem moderada de recursos.

## Desempenho estimado pelo HLS

O `vsynth` do Vivado usado aqui mede utilização, não recalcula a latência funcional do pipeline. Os dados de desempenho permanecem os da C synthesis:

| Métrica | Resultado |
|---|---:|
| Clock alvo | 10,000 ns / 100 MHz |
| Período estimado pelo HLS | 7,204 ns |
| Fmax derivada da estimativa | 138,81 MHz |
| Latência | 9.412–9.460 ciclos |
| Latência a 100 MHz | 94,12–94,60 µs |
| Intervalo de iniciação | 3.138–9.410 ciclos |
| Pipeline | `dataflow` |

Esses valores não substituem um relatório de timing pós-route.

## Alertas e limites do ensaio

- O Tcl automático `vivado_synth.tcl` não contém `create_clock`. Por isso o Vivado avisou que não havia clock definido e que a estimativa de potência seria imprecisa. Nenhum número de potência desta execução deve ser usado no benchmark.
- O uso de 352 IOBs (97,78%) é um artefato da síntese isolada do top-level AXI4-Stream: as portas do módulo são tratadas como I/O externos. No block design, as interfaces AXIS serão conexões internas entre o IP e DMA/interconnect; esse número não representa 352 pinos físicos necessários na placa.
- `report_utilization` pós-síntese ainda não inclui a infraestrutura do sistema nem congestionamento de roteamento. A margem atual de LUT (38,62%) e BRAM (63,46%) precisa acomodar o restante do design.
- A advertência de corrente/potência desta execução também decorre da ausência de clock e de atividade realista. Potência confiável exige design implementado, clocks corretos e, idealmente, atividade de comutação proveniente de simulação.

## Critério para a próxima etapa

O candidato está aprovado para avançar ao Vivado OOC com restrição explícita de 10 ns. A próxima decisão deve usar, nesta ordem:

1. `report_utilization` pós-route;
2. `report_timing_summary` com WNS/TNS e clocks definidos;
3. `report_design_analysis` para congestionamento;
4. `report_power` somente depois de fornecer clock e atividade coerentes;
5. utilização e timing do block design completo com AXI DMA e Processing System.

## Arquivos de evidência

- `lenet_mnist_cap64_hls_csynth.rpt`: estimativa original do Vitis HLS;
- `lenet_mnist_cap64_hls_csynth.xml`: dados estruturados do HLS;
- `vivado_synth.rpt`: relatório de utilização pós-síntese e pós-`opt_design`;
- `vivado_vsynth.log`: log integral do Vivado;
- `vivado_vsynth.jou`: journal do Vivado;
- `utilization_comparison.csv`: tabela comparável e processável por scripts.
