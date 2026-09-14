# GPU — resumo de cinco campanhas independentes

A unidade experimental é a campanha (`n=5` por tamanho). Cada processo manteve batch 1, 200 warmups, baseline de 5 s, telemetria a 100 ms, semente 20260831 e nenhuma remoção de outliers. A primeira campanha é o diretório histórico e `rep02`–`rep05` foram executadas em 2026-09-03.

| Inferências | Métrica | Média entre campanhas | DP | IC95% t | CV |
|---:|---|---:|---:|---:|---:|
| 30,000 | inference_only_mean_ms | 0.176134 ms | 0.005978 | [0.168712, 0.183557] | 3.39% |
| 30,000 | application_e2e_effective_ms | 0.264565 ms | 0.010479 | [0.251554, 0.277577] | 3.96% |
| 30,000 | throughput_effective_inf_s | 3784.455187 inf/s | 147.503598 | [3601.305279, 3967.605094] | 3.90% |
| 30,000 | total_energy_mj_per_inference | 7.875803 mJ | 0.247599 | [7.568369, 8.183238] | 3.14% |
| 100,000 | inference_only_mean_ms | 0.168663 ms | 0.008801 | [0.157735, 0.179591] | 5.22% |
| 100,000 | application_e2e_effective_ms | 0.255899 ms | 0.009307 | [0.244343, 0.267455] | 3.64% |
| 100,000 | throughput_effective_inf_s | 3911.809485 inf/s | 137.969222 | [3740.498069, 4083.120900] | 3.53% |
| 100,000 | total_energy_mj_per_inference | 7.521092 mJ | 0.391685 | [7.034751, 8.007433] | 5.21% |

Todas as dez campanhas mantiveram 29/30 (96,67%) nas 30 amostras únicas. Os valores energéticos usam `nvidia-smi` e representam potência total da placa GPU, não são diretamente equivalentes a RAPL ou aos trilhos internos da ZCU104. Para a comparação energética principal, use o mesmo wattímetro externo.

Arquivos: `campanhas_gpu.csv` contém uma linha por campanha; `resumo_5_campanhas.json` preserva os cálculos completos.
