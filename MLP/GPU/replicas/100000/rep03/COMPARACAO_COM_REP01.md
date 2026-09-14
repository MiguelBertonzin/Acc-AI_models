# Comparação da MLP Iris na GPU — 30.000 × 100.000 inferências

## Resultado principal

A campanha nova executou exatamente **100,000 inferências** na NVIDIA GeForce RTX 3050 OEM. Foram 3333 passagens completas de 30 amostras e uma passagem final de 10 amostras. Nenhum outlier foi removido.

| Métrica | 30.000 inferências | 100.000 inferências | Variação |
|---|---:|---:|---:|
| Inference-only — média | 0.185747 ms | 0.164835 ms | -11.26% |
| Inference-only — mediana | 0.160364 ms | 0.154058 ms | -3.93% |
| Inference-only — desvio-padrão | 0.065265 ms | 0.036779 ms | -43.65% |
| Inference-only — p95 | 0.318122 ms | 0.225169 ms | -29.22% |
| Inference-only — p99 | 0.448947 ms | 0.327987 ms | -26.94% |
| End-to-end — média efetiva | 0.280414 ms | 0.247784 ms | -11.64% |
| End-to-end — vazão efetiva | 3566.15 inf/s | 4035.77 inf/s | +13.17% |
| Potência média da placa | 28.520 W | 28.447 W | -0.25% |
| Energia total por inferência | 8.158457 mJ | 7.230771 mJ | -11.37% |
| Utilização média da GPU | 10.49 % | 9.46 % | -9.78% |
| Temperatura média | 45.42 °C | 47.98 °C | +5.64% |

Variação positiva significa que o valor da campanha de 100.000 foi maior. Para latência e energia, menor é melhor; para vazão, maior é melhor.

## Comparação estatística entre campanhas

O teste t de Welch foi aplicado às médias das passagens completas de 30 inferências, sem assumir variâncias iguais.

| Janela | Diferença 100k − 30k | IC95% da diferença | p bilateral | Diferença padronizada |
|---|---:|---:|---:|---:|
| Inference-only | -20.912 µs | [-24.279, -17.545] µs | 2.952e-32 | -0.501 |
| End-to-end efetiva | -32.632 µs | [-36.318, -28.945] µs | 9.540e-61 | -0.691 |

Na janela inference-only, um IC que inclua zero indica que a diferença não é distinguível da variação entre passagens. Como as campanhas ocorreram em sessões sequenciais e têm autocorrelação temporal, os testes são exploratórios e não demonstram causalidade.

## Campanha de 100.000 — inference-only

`Inference-only` começa depois de `tf.convert_to_tensor` e termina depois de `output.numpy()`. Inclui a execução do grafo TensorFlow na GPU, sincronização e materialização das três probabilidades no host. Como a cópia da entrada pode ser assíncrona, esta não é uma medição pura de kernel CUDA.

| Métrica | Resultado |
|---|---:|
| Média | 0.164835 ms |
| IC95% da média | [0.163893, 0.165777] ms |
| Mediana | 0.154058 ms |
| Desvio-padrão | 0.036779 ms |
| CV | 22.31% |
| p90 / p95 / p99 | 0.197558 / 0.225169 / 0.327987 ms |
| Mínimo / máximo | 0.134714 / 1.376023 ms |

## Campanha de 100.000 — end-to-end

A latência individual `end-to-end` começa antes da criação do tensor e termina depois da sincronização, transferência/materialização da saída e `argmax`. A vazão efetiva usa a soma dos tempos das passagens e também inclui laço Python e armazenamento dos resultados.

| Métrica | Resultado |
|---|---:|
| Média individual | 0.246803 ms |
| IC95% da média | [0.245567, 0.248034] ms |
| Mediana | 0.246956 ms |
| Desvio-padrão | 0.048501 ms |
| p90 / p95 / p99 | 0.282677 / 0.310653 / 0.440643 ms |
| Mínimo / máximo | 0.172404 / 2.034348 ms |
| Média efetiva pelas passagens | 0.247784 ms |
| Vazão efetiva | 4035.77 inferências/s |

A média individual exclui somente o custo de registrar a própria duração. A média efetiva pelas passagens é mais abrangente e, por isso, é a recomendada para comparação end-to-end.

## Telemetria NVIDIA

O `nvidia-smi` foi executado como processo persistente, com intervalo nominal de 100 ms. A potência foi integrada no tempo com regra trapezoidal. A potência de baseline foi medida durante 5.0 s depois do aquecimento e com o contexto TensorFlow ainda carregado.

| Métrica | Resultado |
|---|---:|
| Amostras durante benchmark | 251 |
| Janela energética | 25.418 s |
| Potência média / p95 / máxima | 28.447 / 28.730 / 28.760 W |
| Potência ociosa | 28.254 W |
| Energia total | 723.077 J |
| Energia total por inferência | 7.230771 mJ |
| Energia dinâmica por inferência | 0.049108 mJ |
| Utilização GPU média / p95 / máxima | 9.46% / 12.00% / 12.00% |
| Memória usada média / máxima | 700.0 / 700.0 MiB |
| Clock SM médio / máximo | 1755.0 / 1755.0 MHz |
| Temperatura média / máxima | 48.0 / 49.0 °C |
| Ventoinha média / máxima | 31.0% / 31.0% |
| P-state | P2: 251 |

A energia é referente ao sensor da placa GPU e não inclui CPU, RAM, fonte ou perdas na tomada. A energia dinâmica é a diferença entre duas leituras próximas e deve ser interpretada com cautela frente à incerteza do sensor e à granularidade de 100 ms.

## Validação funcional e estatística

- Arquitetura conferida: 4–8–8–3, ReLU/ReLU/softmax, 139 parâmetros, `float32`.
- Divisão reconstruída: 120 treino / 30 teste, `test_size=0.2`, `random_state=42`, estratificada.
- Holdout balanceado: 10 amostras por classe.
- Acurácia: 29/30 = 96.67%.
- Predições iguais às da campanha de 30.000: True.
- Concordância NumPy/CPU × TensorFlow/GPU: 100.00%.
- Diferença máxima de probabilidade NumPy/CPU × TensorFlow/GPU: 1.192e-07.
- Posicionamento confirmado: `/job:localhost/replica:0/task:0/device:GPU:0`; soft placement desabilitado.
- Todas as 100,000 classes repetidas coincidiram com a predição funcional inicial.

As 100.000 repetições caracterizam desempenho e determinismo, mas não aumentam o tamanho amostral da acurácia: a unidade estatística continua sendo o holdout com 30 flores únicas. O resultado estatístico completo permanece em `GPU/resultados/validacao_estatistica.json`.

## Metodologia completa

1. Expôs somente a GPU de índice 0 com `CUDA_VISIBLE_DEVICES`.
2. Desabilitou soft placement e fixou modelo e grafo em `/GPU:0`; uma saída fora desse dispositivo abortaria o teste.
3. Habilitou determinismo do TensorFlow e manteve `jit_compile=False`, batch 1, `float32`, sem XLA, TensorRT, mixed precision ou quantização.
4. Reconstruiu o holdout original a partir do Iris e confirmou que média e escala coincidem com o `StandardScaler` salvo.
5. Executou 200 inferências de aquecimento, fora das estatísticas.
6. Confirmou probabilidades, classes, paridade NumPy/CPU e igualdade com a campanha de referência.
7. Iniciou telemetria persistente e mediu 5.0 s de baseline ocioso.
8. Processou uma amostra por vez, em ordem permutada deterministicamente pela seed 20260831.
9. Materializou cada saída com `.numpy()`, impedindo medir apenas o enfileiramento assíncrono.
10. Executou exatamente 100,000 inferências: 3333 × 30 + 10.
11. Não removeu outliers. Calculou média, mediana, desvio-padrão amostral, CV, p90, p95, p99, mínimo e máximo.
12. Calculou IC95% com distribuição t sobre as médias das 3333 passagens completas; o ciclo parcial não entrou no IC, mas todas as amostras entraram nas demais estatísticas.
13. Calculou throughput como inferências totais divididas pela soma dos tempos end-to-end das passagens.
14. Integrou `power.draw.instant` na janela completa e dividiu a energia pelo número exato de inferências.
15. Preservou CSVs, arrays NumPy, snapshots `nvidia-smi -q`, hashes, versões e índices do dataset.

## Ambiente

- GPU: NVIDIA GeForce RTX 3050 OEM, compute capability 8.6, 8192 MiB.
- Driver: 580.173.02; limite de potência: 120.00 W.
- TensorFlow 2.21.0, Keras 3.13.2.
- TensorFlow compilado com CUDA 12.5.1 e cuDNN 9.
- Python 3.12.7; NumPy 1.26.4; scikit-learn 1.5.1.
- Modelo SHA-256: `60d4e49c7082d97cac029ba919ffa8c88514ac9109d0bd69bafb23fa663fbb94`.
- Scaler SHA-256: `beef6de7823db389c06565f0eb199ba71a353d4c2050e425ef6efd425bb20e31`.

O “CUDA Version” mostrado no cabeçalho do `nvidia-smi` representa a versão máxima suportada pelo driver; o runtime efetivamente usado pelo TensorFlow é o registrado acima.

## Arquivos produzidos

- `latencias_inference_only_gpu.npy`: 100.000 tempos individuais.
- `latencias_end_to_end_gpu.npy`: 100.000 tempos individuais.
- `passagens_gpu.csv`: uma linha por passagem.
- `metricas_resumo.json`: estatísticas agregadas.
- `comparacao_30k_100k.json`: valores e variações da comparação.
- `comparacao_estatistica_passagens.json`: teste de Welch e IC95% da diferença.
- `telemetria_nvidia_smi.csv` e `resumo_telemetria.json`: telemetria bruta e resumo.
- `validacao_execucao.json`: qualidade, paridade e posicionamento.
- `metadados.json`: ambiente, configuração, hashes e divisão.
- `nvidia_smi_*.txt`: snapshots antes e depois.

## Limitações de comparação

- As campanhas foram sequenciais, não alternadas nem pareadas por reinicialização; diferenças pequenas podem refletir temperatura, DVFS, desktop, driver e escalonamento.
- A campanha antiga não registrou a distribuição individual end-to-end. Para ela, a média end-to-end foi reconstruída como `1000 / throughput_efetivo`; percentis end-to-end não podem ser recuperados.
- A RTX 3050 também atendia ao ambiente gráfico. A memória e potência do `nvidia-smi` representam a placa, não somente o processo do benchmark.
- Este é throughput sustentado batch 1 e síncrono, não throughput máximo com batching ou requisições concorrentes.
