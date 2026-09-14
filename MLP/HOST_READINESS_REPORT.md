# Relatório de prontidão para a ZCU104

**Host pronto para iniciar a placa:** SIM

| Verificação | Estado | Detalhe |
|---|---:|---|
| canonical_methodology | passed | MLP/Iris protocol present |
| xmodel_sha256 | passed | 47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce |
| hls_ip_sha256 | passed | 88a11380ea472636af4b4b72348aeda9e0b3b2421456b71b0b30222b2a774ba3 |
| vitis_deploy_hashes | passed | 12 files; all hashes valid |
| vitis_flow_hashes | passed | 52 files; all hashes valid |
| hls_golden_hashes | passed | 4 files; all hashes valid |
| hls_flow_hashes | passed | 747 files; all hashes valid |
| vitis_host_validation | passed | 29/30; agreement 30/30 |
| hls_rtl_golden | passed | 29/30; RTL golden |
| cpu_five_campaigns | passed | 30k=5, 100k=5 |
| gpu_five_campaigns | passed | 30k=5, 100k=5 |
| documentation_links | passed | 70 checked |

## Pendências que exigem a placa

- Vitis AI: compatibilidade da imagem/DPU, validação física e 5 campanhas de 30k + 5 de 100k.
- HLS4ML: wrapper AXI, block design, timing pós-route completo, bit/HWH, validação física e 5 + 5 campanhas.
- Energia: descoberta e calibração de sensores ou wattímetro externo com escopo documentado.

Este relatório não declara a implementação na placa concluída; apenas confirma que os insumos de host estão consistentes.
