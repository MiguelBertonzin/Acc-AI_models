# MLP Iris em CPU — 30.000 inferências batch 1

## Resultados

| Janela / métrica | Resultado |
|---|---:|
| Inference-only média | 0.110375 ms |
| Inference-only IC95% | [0.109509, 0.111241] ms |
| Inference-only mediana | 0.103741 ms |
| Inference-only p95 / p99 | 0.151397 / 0.204160 ms |
| Inference-only mínimo / máximo | 0.093976 / 0.723585 ms |
| End-to-end média individual | 0.133952 ms |
| End-to-end IC95% | [0.132871, 0.135033] ms |
| End-to-end mediana | 0.125348 ms |
| End-to-end p95 / p99 | 0.189669 / 0.249419 ms |
| End-to-end média efetiva | 0.134728 ms |
| Vazão efetiva | 7422.37 inferências/s |

## Telemetria CPU

| Métrica | Resultado |
|---|---:|
| Amostras durante benchmark | 42 |
| Utilização do processo média / máxima | 110.90% / 125.60% |
| Núcleos lógicos equivalentes médios | 1.109 |
| Utilização global média / máxima | 4.99% / 5.80% |
| Frequência média / p95 / máxima | 1901.9 / 2235.5 / 2449.3 MHz |
| Temperatura do pacote média / máxima | 69.4 / 77.0 °C |
| RSS do processo média / máxima | 724.3 / 724.6 MiB |
| Fan média / máxima | 1021 / 1022 RPM |

## Energia RAPL da CPU

| Métrica | Resultado |
|---|---:|
| Fonte | Linux powercap intel-rapl / package-0 |
| Duração exata da janela RAPL | 4.229343268 s |
| Energia total do pacote | 225.628756 J |
| Potência média do pacote | 53.348414 W |
| Potência ociosa do pacote | 11.511580 W |
| Energia total por inferência | 7.520959 mJ |
| Potência dinâmica média | 41.836834 W |
| Energia dinâmica por inferência | 5.898078 mJ |
| Energia dos núcleos | 187.832039 J |
| Potência média dos núcleos | 44.411633 W |

A energia principal é a do pacote completo; core e uncore são subdomínios e não são somados novamente. A energia dinâmica subtrai a potência ociosa medida durante 5.0 s. RAPL estima energia no pacote e não equivale a consumo na tomada.

## Validação funcional

- Arquitetura: 4–8–8–3, ReLU/ReLU/softmax, 139 parâmetros, `float32`.
- Execução confirmada em `/job:localhost/replica:0/task:0/device:CPU:0`, com nenhuma GPU visível e soft placement desabilitado.
- Holdout: 120 treino / 30 teste, estratificado, `random_state=42`.
- Acurácia: 29/30 = 96.67%.
- Concordância com a GPU: 100.00%.
- Diferença máxima de probabilidade CPU×GPU: 1.192e-07.
- Todas as 30.000 repetições mantiveram as classes da validação inicial.

## Metodologia

1. Definiu `CUDA_VISIBLE_DEVICES` vazio antes de importar TensorFlow e removeu todas as GPUs visíveis.
2. Desabilitou soft placement e fixou modelo e grafo em `/CPU:0`; uma saída fora da CPU abortaria o teste.
3. Usou TensorFlow 2.21.0 com oneDNN, `tf.function`, batch 1, `float32`, `jit_compile=False`, sem XLA, TensorRT, mixed precision ou quantização.
4. Reutilizou o mesmo modelo, scaler, holdout, mapa de classes e seed das campanhas GPU.
5. Executou 200 inferências de aquecimento, excluídas da medição.
6. Mediu 5.0 s de baseline após aquecimento e amostrou telemetria, inclusive os acumuladores RAPL, a cada 100 ms.
7. Executou exatamente 30.000 inferências em 1000 passagens completas de 30 e uma passagem final de 0.
8. Permutou a ordem deterministicamente a cada passagem com seed 20260831.
9. Não removeu outliers.
10. `Inference-only`: cronômetro depois de `tf.convert_to_tensor` e até o retorno de `.numpy()`.
11. `End-to-end` individual: antes da criação do tensor até depois de `.numpy()` e `argmax`.
12. Vazão efetiva: total de inferências dividido pela soma dos tempos completos das passagens, incluindo laço e armazenamento.
13. IC95%: distribuição t aplicada às médias das passagens completas; o ciclo parcial não participa do IC.
14. Telemetria: thread auxiliar registrando CPU global/processo, memória, frequência, temperatura, carga, fan e acumuladores RAPL.
15. Energia: diferença dos contadores cumulativos energy_uj lidos nas fronteiras exatas da campanha, com correção de rollover pelo max_energy_range_uj; potência média é energia dividida pela duração da própria janela RAPL.
16. Energia dinâmica: energia total menos potência média ociosa multiplicada pela duração ativa; resultados negativos são truncados em zero.

O TensorFlow pode usar múltiplas threads oneDNN mesmo em batch 1. Portanto, batch 1 descreve uma amostra por chamada, não execução limitada a um único núcleo.

## Ambiente e arquivos

- CPU: 13th Gen Intel(R) Core(TM) i7-13700; 16 núcleos físicos / 24 CPUs lógicas.
- Governador: ['powersave'].
- Modelo SHA-256: `60d4e49c7082d97cac029ba919ffa8c88514ac9109d0bd69bafb23fa663fbb94`.
- Scaler SHA-256: `beef6de7823db389c06565f0eb199ba71a353d4c2050e425ef6efd425bb20e31`.
- `latencias_inference_only_cpu.npy` e `latencias_end_to_end_cpu.npy`: tempos individuais.
- `passagens_cpu.csv`: métricas por passagem.
- `telemetria_cpu.csv`: telemetria bruta.
- `metricas_resumo.json`, `resumo_telemetria.json`, `resumo_rapl.json`, `validacao_execucao.json` e `metadados.json`: resultados estruturados.
- `diagnostico_*.txt`: evidência da disponibilidade dos sensores.
