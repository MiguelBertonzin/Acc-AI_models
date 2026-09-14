# Resultado do IP hls4ml — MLP Iris

## Configuração sintetizada

- Modelo: MLP 4–8–8–3, 139 parâmetros, softmax mantida.
- Backend: Vitis HLS 2024.2 / hls4ml 1.3.0.
- Dispositivo: ZCU104 (`xczu7ev-ffvc1156-2-e`).
- Clock alvo: 100 MHz (10 ns).
- Precisão padrão: `ap_fixed<16,6>`, sem QKeras.
- Tabelas auxiliares da softmax: `ap_fixed<18,8>`.
- `ReuseFactor = 1`, estratégia `Latency`, `IOType = io_parallel`.

## Validação funcional

- Holdout: 30 amostras.
- Keras: 29/30 = 96,67%.
- HLS C++: 29/30 = 96,67%.
- Concordância de classes Keras × HLS: 100%.
- Co-simulação C/RTL Verilog: Pass; resultados idênticos à C simulation.
- Erro absoluto médio das três saídas: 0,016745.
- Erro absoluto máximo: 0,177612.

A precisão padrão preservou todas as classes do modelo original. As probabilidades
não são numericamente idênticas, sobretudo pela aproximação em ponto fixo da
softmax; portanto, esta configuração está validada para classificação por
`argmax`, mas a fidelidade das probabilidades deve ser considerada separadamente.

## Síntese HLS

| Métrica | Resultado | Disponível na ZCU104 | Utilização |
|---|---:|---:|---:|
| Clock alvo | 10,000 ns | — | 100 MHz |
| Período estimado | 6,586 ns | — | Fmax estimada ≈ 151,84 MHz |
| Latência | 5 ciclos | — | 50 ns a 100 MHz |
| Intervalo de iniciação | 1 ciclo | — | até 1 entrada/ciclo no núcleo |
| BRAM_18K | 3 | 624 | 0,48% |
| DSP | 108 | 1.728 | 6,25% |
| FF | 413 | 460.800 | 0,09% |
| LUT | 4.058 | 230.400 | 1,76% |
| URAM | 0 | 96 | 0% |

Esses números são estimativas da síntese HLS, não resultados pós-place-and-route.

## Interface do IP

O IP usa `ap_ctrl_hs` com `ap_start`, `ap_done`, `ap_idle` e `ap_ready`.
A entrada paralela `features` tem 64 bits, contendo quatro valores de 16 bits.
As três saídas (`layer7_out_0..2`) têm 16 bits cada e sinais `ap_vld`.
Esta interface não é AXI; para conexão ao PS da ZCU104 será necessário um wrapper
AXI ou lógica de integração no block design do Vivado.

## Artefatos principais

- IP empacotado: `mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip`.
- Componente IP-XACT: `mlp_iris_prj/solution1/impl/ip/component.xml`.
- Exportação Vitis: `mlp_iris_prj/solution1/impl/export.zip`.
- Relatório principal: `mlp_iris_prj/solution1/syn/report/mlp_iris_csynth.rpt`.
- Configuração: `configuration_manifest.json`.
- Validação: `validation_summary.json`.
- Resumo estruturado: `build_report.json`.

SHA-256 do ZIP do IP:

```text
88a11380ea472636af4b4b72348aeda9e0b3b2421456b71b0b30222b2a774ba3
```
