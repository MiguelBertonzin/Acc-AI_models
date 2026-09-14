# ResNet8 no Vitis AI — fluxo, validação e resultados na ZCU104

Este diretório reúne o fluxo completo usado para converter, quantizar, compilar,
validar e medir uma ResNet8 para CIFAR-10 no DPU de uma ZCU104. O modelo produz
10 logits sem `softmax`; a classe prevista é obtida por `argmax`.

O fluxo foi concluído. O XModel foi executado na placa, validado nas 10.000
imagens oficiais de teste e submetido a benchmarks de latência, vazão, potência,
energia e impacto da telemetria. A fonte canônica dos resultados da placa é:

```text
resnet8_vitis_ai_COMPLETE_2026-09-02.tar.gz
SHA-256: 604a8f397bd451172ce6b3aedd12586a1660a08f077de1c6271f4db0ef71ceee
```

## 1. Resumo executivo

| Item | Resultado |
|---|---:|
| Acurácia float no host | 74,90% (7.490/10.000) |
| Acurácia quantizada no host | 73,96% (7.396/10.000) |
| Acurácia no DPU | 73,96% (7.396/10.000) |
| Concordância DPU × referência quantizada | 100,00% |
| Erro máximo nos logits DPU × host quantizado | 0 |
| Menor latência de inferência, 1 thread e N=10.000 | 0,2526 ms |
| Vazão de inferência, 3 threads e N=10.000 | 6.707,23 imagens/s |
| Vazão ponta a ponta, 3 threads e N=10.000 | 2.081,76 imagens/s |
| Maior vazão saturada, 4 threads | 12.814,81 imagens/s |
| Potência ociosa média do rail monitorado | 14,4789 W |
| Menor energia dinâmica saturada | 0,2232 mJ/imagem, 4 threads |

Conclusões:

- a conversão Keras 3 → Keras 2.12 preservou todas as decisões na prova;
- a quantização PTQ INT8 reduziu a acurácia em 0,94 ponto percentual;
- o DPU reproduziu exatamente a referência quantizada do host;
- uma thread fornece a menor latência individual;
- três threads fornecem a maior vazão nos cenários de aplicação;
- quatro threads maximizam apenas o ensaio saturado;
- a potência pertence ao sensor disponível no Linux, não exclusivamente ao DPU.

## 2. O que foi feito

| Etapa | Procedimento | Evidência |
|---|---|---|
| Referência | Exportação de entradas/saídas do Keras 3 | `reports/keras3_reference.json` |
| Compatibilização | Reconstrução no Keras 2.12 no container Vitis AI | `reports/keras2_conversion.json` |
| Dados | Preparação do CIFAR-10 e da calibração | `reports/cifar10_preparation.json` |
| Float | Avaliação das 10.000 imagens no host | `reports/float_validation.json` |
| Quantização | PTQ INT8 `pof2s`, 1.000 imagens estratificadas | `reports/quantization.json` |
| Validação INT8 | Comparação com o modelo float | `reports/quantized_validation.json` |
| Compilação | Compilação para `DPUCZDX8G_ISA1_B4096` | `reports/compilation.json` |
| Inspeção | Subgrafos e interfaces do XModel | `reports/xmodel_inspection.json` |
| Implantação | Pacote autocontido para a ZCU104 | `scripts/08_package_deploy.py` |
| Placa | DPU, smoke test e dataset completo | `results/validation/` no pacote |
| Desempenho | Inferência, ponta a ponta e saturação | `performance/` no pacote |
| Energia | Ocioso, carga ativa e overhead | `power/` e `telemetry_overhead/` |

## 3. Organização

```text
Vitis AI/
├── artifacts/                 # modelo quantizado, XModel e deploy
├── backups/                   # cópia anterior do pacote da placa
├── board/                     # scripts básicos enviados à placa
├── config/                    # experimento e arquitetura
├── data/prepared/             # calibração, teste e prova de conversão
├── docker/                    # inicialização do container
├── logs/                      # logs das ferramentas
├── manifests/                 # inventários e hashes do host
├── models/float/              # modelos Keras 3 e Keras 2
├── reports/                   # relatórios do fluxo do host
├── results/                   # saídas float e INT8 do host
├── scripts/                   # scripts 00 a 09
├── resnet8_final_v1_RESULTS_2026-09-01.tar.gz
└── resnet8_vitis_ai_COMPLETE_2026-09-02.tar.gz
```

Os milhares de arquivos da placa não foram duplicados soltos. O pacote
`COMPLETE` contém scripts, modelo, dados, validação e a árvore
`results/benchmark_vart/resnet8_final_v1/`.

## 4. Modelo e dados

| Propriedade | Valor |
|---|---|
| Rede | ResNet8 |
| Dataset | CIFAR-10 |
| Entrada | `float32`, NHWC, `32 × 32 × 3` |
| Saída | 10 logits, sem `softmax` |
| Decisão | `argmax(logits)` |
| Parâmetros | 78.714 |
| Camadas Keras | 31 |
| Teste | 10.000 imagens, 1.000 por classe |
| Pré-processamento | `float32` e divisão por 255 |
| Calibração | 1.000 imagens, 100 por classe |
| Amostragem | estratificada, sem reposição |
| Semente | 20260825 |

| Modelo | SHA-256 |
|---|---|
| Keras 3 original | `047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e` |
| Keras 2.12 reconstruído | `6632bdf0dd8becd7afcb9dbb623be2c090ef500e57343cbee2c751729cdbce3f` |

### 4.1 Conversão Keras

Nas 32 entradas de prova, as 32 decisões por `argmax` foram iguais. O erro
absoluto médio foi `3,524986 × 10⁻⁶`; o máximo foi `1,287460 × 10⁻⁵`,
abaixo da tolerância de `2 × 10⁻⁵`.

### 4.2 Acurácia no host

| Modelo | Corretas | Total | Acurácia | IC 95% de Wilson |
|---|---:|---:|---:|---:|
| Float Keras 2.12 | 7.490 | 10.000 | 74,90% | 74,04%–75,74% |
| Quantizado INT8 | 7.396 | 10.000 | 73,96% | 73,09%–74,81% |

A concordância float × INT8 foi 93,75%. A queda foi 0,94 ponto percentual.

## 5. Quantização, compilação e XModel

| Propriedade | Valor |
|---|---|
| Ferramenta | Vitis AI 3.5 / TensorFlow 2.12 |
| Método | PTQ INT8 `pof2s` |
| Alvo | `DPUCZDX8G_ISA1_B4096` |
| Retorno do compilador | 0, sucesso |
| XModel | 249.251 bytes |
| SHA-256 do XModel | `bfacc85783958ff0d734e5c7e44f126a6fd4af9b6b1f20e4d9b2f7ae96fa7802` |
| Subgrafos | entrada USER, computação DPU e dequantização CPU |
| Operações no subgrafo DPU | 36 |

| Tensor DPU | Forma | Tipo | `fix_point` | Tratamento |
|---|---|---|---:|---|
| Entrada `quant_input_layer` | `[1,32,32,3]` | INT8 | 6 | `round(x × 64)` e saturação |
| Saída `quant_dense_fix` | `[1,1,1,10]` | INT8 | 2 | dequantização por 4 |

## 6. Ambiente da ZCU104

| Componente | Valor observado |
|---|---|
| CPU | AArch64, 4 CPUs |
| Kernel | Linux 5.15.36, Xilinx 2022.2 |
| Python / NumPy | 3.9.9 / 1.21.2 |
| VART | 3.0 |
| XRT | 2.14 / 2022.2 |
| DPU | 2 núcleos a 300 MHz |
| Frequência XRT | 300 MHz |
| Fingerprint | `0x101000056010407` |

O host usou Vitis AI 3.5 e a placa, VART 3.0. A combinação foi aceita e validada
empiricamente: as 10.000 saídas do DPU foram idênticas à referência INT8.

## 7. Validação na placa

### 7.1 Smoke test

| Item | Resultado |
|---|---:|
| Índice / rótulo CIFAR-10 | 0 / 3 |
| Classe DPU / host INT8 | 3 / 3 |
| Aquecimentos | 20 |
| Latência observada | 0,23909 ms |
| Estado | aprovado |

A latência acima é apenas um teste rápido; as medições oficiais estão na seção 9.

### 7.2 Dataset completo

| Verificação | Resultado |
|---|---:|
| Imagens / acertos | 10.000 / 7.396 |
| Acurácia DPU | 73,96% |
| Concordância DPU × host INT8 | 100,00% |
| Divergências | 0 |
| MAE / RMSE / erro máximo nos logits | 0 / 0 / 0 |
| Intervalo de saída INT8 | −109 a 84 |
| Saturação em −128 ou +127 | nenhuma |

### 7.3 Resultado por classe

| ID | Classe | Acertos/1.000 | Acurácia |
|---:|---|---:|---:|
| 0 | avião | 875 | 87,50% |
| 1 | automóvel | 847 | 84,70% |
| 2 | pássaro | 554 | 55,40% |
| 3 | gato | 624 | 62,40% |
| 4 | cervo | 610 | 61,00% |
| 5 | cachorro | 722 | 72,20% |
| 6 | sapo | 829 | 82,90% |
| 7 | cavalo | 792 | 79,20% |
| 8 | navio | 776 | 77,60% |
| 9 | caminhão | 767 | 76,70% |

## 8. Protocolo de benchmark

| Parâmetro | Configuração |
|---|---|
| Batch | 1 |
| Threads | 1, 2, 3 e 4 |
| Recursos | um runner VART e buffers privados por worker |
| Afinidade | threads fixadas; zero erros |
| Aquecimento | 100 inferências por runner |
| Medições normais | N=100, 1.000 e 10.000; 100 ciclos |
| Latência saturada | 30 blocos × 2.000 requisições |
| Vazão saturada | 20 janelas × 10 s |
| Outliers | nenhum removido |
| Relógio | `time.perf_counter_ns()` |
| Semente / bootstrap final | 20260825 / 2.000 reamostragens |

Fronteiras:

- **latência de inferência:** apenas `execute_async()` + `wait()`; INT8 copiado;
- **vazão de inferência:** seleção INT8, cópia, execução, espera, `argmax` e Python;
- **ponta a ponta:** `uint8` em RAM → normalização → INT8 → DPU → `argmax`;
- **saturado:** mesmo INT8 já carregado, sem preprocessamento, cópia nova ou `argmax`.

Latência individual e vazão agregada são métricas distintas. Com concorrência, a
latência pode piorar enquanto a vazão aumenta.

## 9. Resultados de desempenho

### 9.1 Inferência

| Threads | N | Média (ms) | Mediana | P95 | P99 | Imagens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0,253053 | 0,252320 | 0,261123 | 0,275160 | 3.076,16 |
| 1 | 1.000 | 0,252772 | 0,252260 | 0,254630 | 0,265880 | 3.088,37 |
| 1 | 10.000 | 0,252552 | 0,252100 | 0,254570 | 0,264000 | 3.092,50 |
| 2 | 100 | 0,293600 | 0,281820 | 0,321640 | 0,376532 | 5.136,10 |
| 2 | 1.000 | 0,265992 | 0,261150 | 0,280270 | 0,291000 | 5.830,91 |
| 2 | 10.000 | 0,264278 | 0,260120 | 0,279330 | 0,290420 | 5.889,36 |
| 3 | 100 | 0,365383 | 0,365205 | 0,410700 | 0,489211 | 6.288,77 |
| 3 | 1.000 | 0,352149 | 0,351990 | 0,398140 | 0,412220 | 6.652,32 |
| 3 | 10.000 | 0,340574 | 0,351310 | 0,397550 | 0,406090 | **6.707,23** |
| 4 | 100 | 0,511541 | 0,513910 | 0,674356 | 0,809684 | 6.157,41 |
| 4 | 1.000 | 0,516953 | 0,514660 | 0,578660 | 0,761012 | 6.465,61 |
| 4 | 10.000 | 0,519735 | 0,525890 | 0,558600 | 0,699660 | 6.545,64 |

Uma thread minimiza a latência. Três threads maximizam a vazão normal.

### 9.2 Ponta a ponta

| Threads | N | Média (ms) | Mediana | P95 | P99 | Imagens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0,682474 | 0,679615 | 0,693571 | 0,729761 | 1.439,45 |
| 1 | 1.000 | 0,682308 | 0,680120 | 0,693310 | 0,701790 | 1.443,16 |
| 1 | 10.000 | 0,681568 | 0,679280 | 0,693230 | 0,709620 | 1.444,74 |
| 2 | 100 | 1,047969 | 1,047180 | 1,113362 | 1,190655 | 1.874,68 |
| 2 | 1.000 | 0,998279 | 1,040360 | 1,064950 | 1,117783 | 1.982,49 |
| 2 | 10.000 | 0,954863 | 0,934430 | 1,050910 | 1,071540 | 2.070,73 |
| 3 | 100 | 1,436915 | 1,424990 | 1,629381 | 1,817143 | 2.039,18 |
| 3 | 1.000 | 1,420869 | 1,416170 | 1,583540 | 1,622200 | **2.090,53** |
| 3 | 10.000 | 1,429096 | 1,423760 | 1,593330 | 1,628230 | 2.081,76 |
| 4 | 100 | 1,917751 | 1,902495 | 2,744354 | 3,202636 | 1.994,41 |
| 4 | 1.000 | 1,928452 | 1,905340 | 2,724241 | 3,178991 | 2.040,68 |
| 4 | 10.000 | 1,923526 | 1,900700 | 2,696090 | 3,156770 | 2.054,89 |

O custo de pré/pós-processamento reduz a vazão. Três threads continuam sendo o
melhor ponto prático.

### 9.3 Saturado

| Threads | Média (ms) | Mediana | P95 | P99 | Imagens/s | IC 95% |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,246675 | 0,246260 | 0,248450 | 0,257490 | 3.991,11 | 3.988,56–3.993,70 |
| 2 | 0,261533 | 0,262200 | 0,272530 | 0,281540 | 7.517,01 | 7.509,11–7.524,95 |
| 3 | 0,273826 | 0,274150 | 0,287460 | 0,301660 | 10.725,94 | 10.694,83–10.753,21 |
| 4 | 0,307111 | 0,301690 | 0,340680 | 0,369240 | **12.814,81** | 12.734,74–12.893,41 |

Esse teto explora os dois núcleos DPU, mas não representa uma aplicação completa.

## 10. Potência e energia

O sensor foi o INA226 em `/sys/class/hwmon/hwmon0/power1_input`, sem rótulo.
Logo, os valores são do **rail monitorado**, não potência isolada do DPU.
Ocioso: 120 s, média **14,4788595 W**. Carga: 20 janelas de 10 s por combinação,
telemetria a cada 1 s. Energia total = potência ativa/vazão; energia dinâmica =
(ativa − ociosa)/vazão.

| Cenário | T | Ativa (W) | Incremento (W) | Img/s | Total (mJ/img) | Dinâmica (mJ/img) |
|---|---:|---:|---:|---:|---:|---:|
| Inferência | 1 | 15,235936 | 0,757077 | 3.134,285 | 4,861062 | 0,241548 |
| Inferência | 2 | 15,911795 | 1,432936 | 5.880,780 | 2,706365 | 0,243666 |
| Inferência | 3 | 16,204295 | 1,725436 | 6.743,447 | 2,403952 | 0,255883 |
| Inferência | 4 | 16,158591 | 1,679731 | 6.566,472 | 2,461005 | 0,255793 |
| Ponta a ponta | 1 | 14,959018 | 0,480159 | 1.448,529 | 10,327046 | 0,331481 |
| Ponta a ponta | 2 | 15,243927 | 0,765068 | 2.069,512 | 7,366680 | 0,369679 |
| Ponta a ponta | 3 | 15,278391 | 0,799531 | 2.049,414 | 7,455756 | 0,390169 |
| Ponta a ponta | 4 | 15,304650 | 0,825790 | 2.036,524 | 7,515353 | 0,405510 |
| Saturado | 1 | 15,389582 | 0,910722 | 3.981,747 | 3,865034 | 0,228724 |
| Saturado | 2 | 16,195805 | 1,716945 | 7.483,457 | 2,164410 | 0,229455 |
| Saturado | 3 | 16,903268 | 2,424409 | 10.692,851 | 1,581055 | 0,226760 |
| Saturado | 4 | 17,280268 | 2,801409 | 12.554,478 | **1,376718** | **0,223152** |

## 11. Impacto da telemetria

Foram feitos seis pares ON/OFF por cenário/thread, 4 s por estado e ordem
alternada. Valor positivo indica execução aparentemente mais lenta com telemetria.

| Cenário | T | Overhead médio | IC 95% |
|---|---:|---:|---:|
| Inferência | 1 | −0,0547% | −0,1509% a 0,0309% |
| Inferência | 2 | 0,0569% | −0,0722% a 0,2247% |
| Inferência | 3 | −0,0243% | −0,9559% a 0,7550% |
| Inferência | 4 | −0,3254% | −1,0519% a 0,3820% |
| Ponta a ponta | 1 | 0,0783% | 0,0222% a 0,1519% |
| Ponta a ponta | 2 | −0,3710% | −0,8288% a 0,1092% |
| Ponta a ponta | 3 | −0,3550% | −1,5242% a 0,6079% |
| Ponta a ponta | 4 | 0,0932% | −0,3170% a 0,7387% |
| Saturado | 1 | 0,0867% | −0,0458% a 0,2440% |
| Saturado | 2 | −1,3238% | −2,8682% a −0,0279% |
| Saturado | 3 | 2,2983% | 0,5225% a 4,3102% |
| Saturado | 4 | 1,3265% | −0,5811% a 3,0454% |

A maioria dos ICs inclui zero. Saturado/3 threads mostrou ~2,3% detectável; por
isso desempenho e potência foram coletados separadamente.

## 12. Comparação com hls4ml

A comparação direta mais defensável usa uma thread no Vitis AI, pois o hls4ml
foi medido de forma serial/síncrona. Inferência hls4ml usa N=10.000/100 ciclos;
ponta a ponta e saturado disponíveis usam N=5.000/20 ciclos.

| Métrica | hls4ml | Vitis AI, 1 thread | Relação |
|---|---:|---:|---:|
| Acurácia | 74,92% | 73,96% | hls4ml +0,96 p.p. |
| Latência de inferência | 0,8493 ms | 0,2526 ms | Vitis 3,36× menor |
| Vazão de inferência | 899,21 img/s | 3.092,50 img/s | Vitis 3,44× maior |
| Latência ponta a ponta | 1,5977 ms | 0,6816 ms | Vitis 2,34× menor |
| Vazão ponta a ponta | 620,70 img/s | 1.444,74 img/s | Vitis 2,33× maior |
| Vazão saturada serial | 1.200,94 img/s | 3.991,11 img/s | Vitis 3,32× maior |

| Comparação pareada das 10.000 imagens | Quantidade |
|---|---:|
| Ambos corretos | 7.203 |
| Ambos errados | 2.315 |
| Apenas hls4ml correto | 289 |
| Apenas Vitis AI correto | 193 |
| Predições idênticas | 9.348 (93,48%) |
| McNemar exato | `p ≈ 1,421065 × 10⁻⁵` |

Vitis AI com 2–4 threads explora dois DPUs a 300 MHz e não é comparável
diretamente a um acelerador hls4ml a 100 MHz. Potência também não é estritamente
comparável: hls4ml usou `12V_power`; no Vitis AI o rail não tem rótulo.

## 13. Artefatos coletados

| Local | Conteúdo |
|---|---|
| `models/float/` | Keras 3 original e Keras 2.12 |
| `artifacts/quantized/` | H5 quantizado |
| `data/prepared/` | prova, calibração e teste |
| `results/` | logits/predições float e INT8 do host |
| `reports/` | relatórios de cada etapa do host |
| `manifests/` | inventário e hashes |
| `results/smoke_inference.json` no pacote | smoke test |
| `results/validation/` no pacote | previsões, logits e matriz de confusão |
| `.../performance/` | NPYs de latência, ciclos CSV e configurações |
| `.../power/` | ocioso, 240 janelas ativas e telemetria bruta |
| `.../telemetry_overhead/` | pares ON/OFF e resumos |
| `.../FINAL_SUMMARY.{json,csv}` | consolidação final |
| `.../METADATA*.json` | ambiente, protocolo e proveniência |
| `.../SHA256SUMS*.txt` | integridade dos subconjuntos |

## 14. Auditoria de integridade

| Verificação | Resultado |
|---|---:|
| Manifesto do host | 45/45 hashes válidos |
| Pacote de deploy | 7/7 hashes válidos |
| Conjunto de validação | 8/8 hashes válidos |
| Manifesto visível de desempenho | 102/102 hashes válidos |
| JSON / NPY / NPZ analisados | 48 / 35 / 2 válidos |
| NaN ou infinito | nenhum |
| Latências normais / saturadas | 8.880.000 / 240.000 |
| Inferências em janelas saturadas | 7.010.073 |
| Erros ao recalcular estatísticas | 0 |
| Erros de afinidade | 0 |
| Scripts Python com sintaxe válida | 20 |
| Script shell válido | sim |

Ressalvas da auditoria:

- `SHA256SUMS_COMPLETE.txt` cita o transitório
  `power_run_2026-09-02.pid`, não empacotado; os demais itens foram validados;
- `power_run_2026-09-02.log` está truncado textualmente, mas CSVs e resumos estão completos;
- `postrun/thermal_after.txt` não fornece temperatura numérica;
- o backup antigo não contém dois NPZ do manifesto; ambos existem no `COMPLETE`;
- o resumo final usa 2.000 bootstraps; o intermediário usou 10.000, sem alteração
  das estimativas pontuais.

## 15. Limitações

- O benchmark foi feito em Python/VART; não há fonte C/C++ coletada.
- O batch é 1; não há caracterização para lotes maiores.
- Vazão saturada é limite superior, não substitui ponta a ponta.
- O rail INA226 não identificado não pode ser atribuído apenas ao DPU.
- Energia dinâmica herda ruído da subtração da média ociosa.
- Sem temperatura final numérica, não se correlaciona formalmente vazão e calor.
- Comparações exigem declarar clock, aceleradores, fronteira, N e ciclos.
- Vitis AI 3.5/VART 3.0 foi validado para este XModel, não como garantia geral.

## 16. Reprodução no host

```bash
python3 scripts/00_export_keras3_reference.py
chmod +x docker/run_vai.sh
./docker/run_vai.sh python scripts/01_rebuild_keras2_model.py
./docker/run_vai.sh python scripts/02_prepare_cifar10.py
./docker/run_vai.sh python scripts/03_validate_float_model.py
./docker/run_vai.sh python scripts/04_quantize_ptq.py
./docker/run_vai.sh python scripts/05_validate_quantized_model.py
./docker/run_vai.sh python scripts/06_compile_xmodel.py
./docker/run_vai.sh python scripts/07_inspect_xmodel.py
python3 scripts/08_package_deploy.py
python3 scripts/09_generate_manifest.py
```

O primeiro comando usa Keras 3 no host. Reconstrução, quantização e compilação
usam o container fixado do Vitis AI.

## 17. Execução na placa

```bash
python3 00_inspect_dpu.py
python3 00_power_discovery.py
python3 01_smoke_inference.py
python3 02_validate_full_dataset.py
python3 03_benchmark_vart_robusto.py
python3 04_analyze_benchmark.py
```

Conferir hashes antes da execução. A inspeção precede a inferência para registrar
DPU/XIR/VART; o smoke test valida tensor, escala e decisão antes do dataset todo.

```bash
sha256sum -c resnet8_vitis_ai_COMPLETE_2026-09-02.tar.gz.sha256
tar -tzf resnet8_vitis_ai_COMPLETE_2026-09-02.tar.gz
```

## 18. Estado final

- [x] modelo original preservado;
- [x] conversão Keras 3 → 2.12 validada;
- [x] CIFAR-10 e calibração preparados;
- [x] modelos float e INT8 validados;
- [x] XModel compilado e inspecionado;
- [x] DPU identificado e smoke test aprovado;
- [x] 10.000 imagens validadas no DPU;
- [x] latência, vazão, potência, energia e overhead coletados;
- [x] resultados consolidados e empacotados com hashes.

O fluxo está funcional e reproduz exatamente o modelo quantizado no host. Use
uma thread para priorizar latência e três para vazão de aplicação. Quatro threads
só foram superiores no cenário saturado.
