# Validação de energia RAPL — MLP Iris em CPU

## Conclusão

A energia RAPL da CPU foi medida e validada. A referência recomendada para comparação futura é a campanha de **100.000 inferências, batch 1**, resumida sobre cinco campanhas independentes:

- energia total do pacote: **7.927 mJ/inferência**, IC95% [6.899, 8.954], mediana 7.651, CV 10.44%;
- energia dinâmica do pacote: **6.121 mJ/inferência**, IC95% [5.424, 6.819];
- potência média do pacote: **55.245 W**, IC95% [50.288, 60.202].

Não houve diferença estatisticamente significativa entre 30.000 e 100.000 inferências. Para energia total, a diferença pareada 100k−30k foi +0.241 mJ/inferência, IC95% [-0.398, +0.881], p=0.354. Para energia dinâmica, p=0.351.

## Resultados agregados

| Campanha | Energia total package (mJ/inf) | Energia dinâmica (mJ/inf) | Potência package (W) | CV energia total |
|---|---:|---:|---:|---:|
| 30.000 × 5 | 7.685 [7.128, 8.243] | 5.974 [5.591, 6.356] | 53.444 [50.392, 56.497] | 5.85% |
| 100.000 × 5 | 7.927 [6.899, 8.954] | 6.121 [5.424, 6.819] | 55.245 [50.288, 60.202] | 10.44% |

Os colchetes apresentam IC95% da média entre campanhas. Nenhuma execução foi removida.

## Campanhas individuais

| Inferências | Réplica | Package W | Idle W | Total mJ/inf | Dinâmica mJ/inf | E2E ms | inf/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 1 | 57.411 | 13.487 | 8.442 | 6.459 | 0.140597 | 7112.53 |
| 30000 | 2 | 53.103 | 11.588 | 7.659 | 5.987 | 0.137990 | 7246.92 |
| 30000 | 3 | 53.348 | 11.512 | 7.521 | 5.898 | 0.134728 | 7422.37 |
| 30000 | 4 | 50.658 | 11.480 | 7.249 | 5.606 | 0.136864 | 7306.53 |
| 30000 | 5 | 52.701 | 11.426 | 7.557 | 5.918 | 0.137186 | 7289.40 |
| 100000 | 1 | 60.356 | 14.248 | 9.098 | 6.950 | 0.144457 | 6922.47 |
| 100000 | 2 | 54.140 | 11.740 | 7.651 | 5.992 | 0.135240 | 7394.24 |
| 100000 | 3 | 58.375 | 14.283 | 8.443 | 6.378 | 0.138320 | 7229.62 |
| 100000 | 4 | 50.915 | 11.476 | 7.110 | 5.508 | 0.133577 | 7486.30 |
| 100000 | 5 | 52.441 | 11.096 | 7.330 | 5.779 | 0.133748 | 7476.75 |

## Metodologia

1. Modelo 4–8–8–3, TensorFlow/oneDNN em CPU, `float32`, batch 1, chamadas seriais e sincronizadas.
2. Mesmos artefatos, holdout, seed 20260831 e fronteiras inference-only/end-to-end das campanhas anteriores.
3. Em cada campanha: 200 inferências de warm-up, 5 s de baseline ocioso e exatamente 30.000 ou 100.000 inferências.
4. Cinco campanhas independentes por tamanho; cada execução repetiu inicialização, warm-up e baseline.
5. Fonte: Linux powercap, domínio `intel-rapl:0` (`package-0`), contador cumulativo `energy_uj`.
6. Energia: diferença dos acumuladores imediatamente antes/depois da janela ativa, com correção por `max_energy_range_uj` em rollover.
7. Potência média: energia do pacote dividida pela duração entre as próprias leituras RAPL.
8. Energia total por inferência: energia package dividida pelo número exato de inferências.
9. Energia dinâmica: `max(E_ativa − P_ociosa × t_ativa, 0)`.
10. Telemetria auxiliar a cada 100 ms gravou package/core/uncore, uso, frequência e temperatura. As leituras de fronteira determinam a energia final.
11. `package` já contém os subdomínios; `core` e `uncore` não foram somados novamente.
12. IC95% t de Student sobre cinco campanhas; comparação 100k×30k por teste t pareado.
13. Nenhum outlier foi removido. Todas as dez campanhas tiveram 29/30 acertos e concordância CPU×GPU de 100%.

## Validações e limitações

- Dez campanhas e todos os artefatos foram lidos.
- Package foi positivo e package ≥ core em todas.
- Foram verificadas numericamente as identidades energia/duração = potência e energia/N = energia por inferência.
- RAPL mede uma estimativa do pacote do processador; não inclui fonte, tomada e periféricos.
- `turbostat` sem privilégios continua bloqueado de abrir MSR pelo kernel apesar do modo do dispositivo. Isso não impede a leitura validada via powercap.
- Nenhuma conclusão energética deve comparar diretamente package RAPL com o sensor da placa GPU, pois os escopos físicos são diferentes.

## Referência para CPU, GPU e FPGA

Use como baseline CPU principal **7.927 mJ/inferência package RAPL** e **6.121 mJ/inferência dinâmica**, ambos com seus IC95%.

Para FPGA, registre separadamente energia dos rails do dispositivo/placa e energia do host, mantendo a mesma janela batch 1 serial e sincronizada. O valor GPU de 8,193 mJ/inferência é energia da placa GPU e deve ser rotulado separadamente.

## Arquivos

- `validacao_rapl/campanhas_rapl.csv`: uma linha por campanha.
- `validacao_rapl/resumo_estatistico_rapl.json`: estatísticas, testes e integridade.
- `resultados_30000_rapl/` e `resultados_100000_rapl/`: réplicas 1.
- `validacao_rapl/30000_rep2..5` e `validacao_rapl/100000_rep2..5`: demais réplicas.
