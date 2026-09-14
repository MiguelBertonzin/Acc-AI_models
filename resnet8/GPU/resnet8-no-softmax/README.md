# ResNet-8 sem softmax — benchmark de inferência na GPU

## Tabela completa do resultado principal

O resultado principal usa as 10.000 imagens do conjunto oficial de teste do CIFAR-10 e 100 ciclos completos. Ele é o ponto mais significativo para acurácia e o mais consistente temporalmente: cobre todas as imagens de teste e reúne 1.000.000 de latências batch 1. Os pontos de 10, 20 e 50 ciclos são prefixos da mesma série de 100 ciclos.

| Métrica coletada | Resultado principal |
|---|---:|
| Variante do modelo | ResNet-8 sem softmax |
| Ativação de saída | linear (logits) |
| Tamanho do arquivo do modelo | 416.384 bytes |
| Dispositivo de inferência | NVIDIA GeForce RTX 3050 OEM |
| Precisão numérica | float32 |
| Batch | 1 |
| Imagens únicas | 10.000 |
| Ciclos completos | 100 |
| Inferências cronometradas | 1.000.000 |
| Predições corretas | 7.489 |
| Acurácia | 74,8900% |
| IC 95% da acurácia, Wilson | [74,0306%; 75,7303%] |
| Latência média | 0,6912 ms |
| IC 95% da média das latências por ciclo | [0,6615; 0,7209] ms |
| Latência mediana | 0,5122 ms |
| Desvio padrão da latência | 0,2609 ms |
| Latência p95 | 1,1219 ms |
| Latência mínima | 0,4721 ms |
| Latência máxima | 2,9269 ms |
| Vazão média | 1.268,4927 inferências/s |
| Desvio padrão da vazão entre ciclos | 233,5978 inferências/s |
| IC 95% da vazão média | [1.222,7084; 1.314,2771] inferências/s |
| Vazão efetiva global | 1.223,8367 inferências/s |
| Tempo total dos ciclos medidos | 817,1025 s |
| Potência total média da placa | 35,1824 W |
| Potência média pelo sensor de janela de 1 s | 35,1736 W |
| Potência dinâmica média | 6,1194 W |
| Energia total por inferência | 28,4617 mJ |
| Energia dinâmica por inferência | 4,7143 mJ |
| Utilização média da GPU | 42,7366% |
| Clock SM médio | 1.844,6568 MHz |
| Temperatura média da GPU | 53,2559 °C |
| Amostras de potência | 8.155 |
| Potência ociosa usada como linha de base | 29,0630 W |
| Resolução da telemetria | 100 ms |

## Convergência no conjunto completo de 10.000 imagens

Esta tabela mostra como os valores evoluem quando a mesma série cresce de 10 para 100 ciclos. A linha de 100 ciclos é a recomendada para comparações com as futuras implementações FPGA.

| Ciclos | Inferências | Acurácia | Latência média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | Vazão média (FPS) | Potência total (W) | Potência dinâmica (W) | Energia total/inf. (mJ) | Energia dinâmica/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100.000 | 74,89% | 0,6505 | 0,5084 | 0,2336 | 1,0771 | 1.326,1883 | 36,3124 | 7,2494 | 27,9193 | 5,4407 |
| 20 | 200.000 | 74,89% | 0,6690 | 0,5085 | 0,2497 | 1,1026 | 1.298,8403 | 35,8367 | 6,7737 | 28,2102 | 5,1489 |
| 50 | 500.000 | 74,89% | 0,6829 | 0,5107 | 0,2582 | 1,1180 | 1.277,1279 | 35,2875 | 6,2245 | 28,2658 | 4,7672 |
| **100** | **1.000.000** | **74,89%** | **0,6912** | **0,5122** | **0,2609** | **1,1219** | **1.268,4927** | **35,1824** | **6,1194** | **28,4617** | **4,7143** |

Os intervalos de confiança, mínimos, máximos, vazão efetiva, utilização, clock, temperatura e contagem de amostras de potência de cada prefixo estão preservados integralmente em [resultados/resultados.csv](resultados/resultados.csv).

## Por que 10.000 imagens e 100 ciclos são o resultado principal

Para acurácia, as 10.000 imagens correspondem a todo o conjunto de teste oficial do CIFAR-10. Não há erro de seleção de subconjunto e o intervalo de Wilson é muito mais estreito que nos conjuntos de 100 e 1.000 imagens:

| Imagens únicas | Corretas | Acurácia | IC 95% de Wilson |
|---:|---:|---:|---:|
| 100 | 74 | 74,00% | [64,629%; 81,595%] |
| 1.000 | 740 | 74,00% | [71,193%; 76,623%] |
| **10.000** | **7.489** | **74,89%** | **[74,031%; 75,730%]** |

Para desempenho, 100 ciclos capturam variações de clock, temperatura, escalonamento do sistema e DVFS durante um período maior. A mediana próxima de 0,51 ms representa o regime típico, enquanto o p95 próximo de 1,12 ms mostra a cauda observada. Reportar somente a média esconderia essa diferença.

Repetir as imagens não aumenta artificialmente a evidência de acurácia. A acurácia e o intervalo de Wilson usam apenas as imagens únicas; as repetições servem para caracterizar desempenho e estabilidade temporal.

## Objetivo e escopo

O ensaio mede o modelo `resnet8_cifar10_keras3_no_softmax.h5`, cuja última camada Dense possui ativação linear. A saída contém 10 logits, e a classe prevista é o índice do maior logit. O modelo tem entrada `(1, 32, 32, 3)`, saída `(1, 10)` e 78.714 parâmetros. A inferência numérica foi executada em float32.

O objetivo é produzir uma referência GPU batch 1 comparável às implementações futuras em hls4ml e Vitis. Batch 1 foi mantido em todas as medições; não foi usado batching para aumentar artificialmente a vazão.

## Validação funcional da remoção da softmax

A equivalência foi validada separadamente nas 10.000 imagens, também com batch 1 e execução em GPU. Os 47 arrays de pesos são exatamente iguais entre os modelos (erro máximo absoluto zero). Aplicar softmax aos logits reconstruiu as probabilidades originais com erro absoluto máximo de `2,3841858 × 10⁻⁷` e erro médio de `2,3514084 × 10⁻⁹`.

| Verificação | Resultado |
|---|---:|
| Ativação do modelo original | softmax |
| Ativação do modelo avaliado | linear |
| Índices de classe iguais | 10.000 de 10.000 |
| Índices divergentes | 0 |
| Acurácia do modelo com softmax, batch 1 | 74,89% |
| Acurácia do modelo sem softmax, batch 1 | 74,89% |
| Erro máximo entre pesos | 0 |
| Erro máximo na reconstrução das probabilidades | 2,3841858 × 10⁻⁷ |

O resultado completo está em [`resultados/validacao_equivalencia.json`](resultados/validacao_equivalencia.json), e o procedimento reproduzível está em [`validar_equivalencia.py`](validar_equivalencia.py). A validação usa batch 1 porque um teste auxiliar com batch 256 alterou uma decisão de classe por arredondamento numérico em ambos os modelos; o protocolo de referência para FPGA permanece estritamente batch 1.

Métricas FPGA presentes no `info.txt` — LUT, FF, DSP, BRAM, URAM e frequência atingida da lógica — não se aplicam à execução GPU. O clock SM médio da GPU foi registrado como telemetria, mas não deve ser confundido com a frequência atingida por uma implementação FPGA.

## Ambiente utilizado

| Componente | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 OEM |
| UUID da GPU | GPU-234713d7-e691-1f3e-f20d-851ede38469f |
| Memória da GPU | 8.192 MiB |
| Limite de potência | 120 W |
| Driver NVIDIA | 580.173.02 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| NumPy | 1.26.4 |
| CUDA do build TensorFlow | 12.5.1 |
| cuDNN do build TensorFlow | 9 |
| Python | 3.12.7 |
| Sistema | Linux 7.0.0-30-generic x86_64, glibc 2.39 |
| Modelo | `resnet8_cifar10_keras3_no_softmax.h5` |
| SHA-256 do modelo | `047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e` |
| Seed | 20260825 |
| Início da coleta | 2026-08-25 18:35:39 UTC |
| Fim da coleta | 2026-08-25 18:50:44 UTC |

## Preparação do dataset

O conjunto foi carregado por `tf.keras.datasets.cifar10.load_data()`. Foi utilizada a divisão oficial de teste, com 10.000 imagens RGB de 32 × 32 pixels e 1.000 imagens por classe.

O pré-processamento foi:

```python
test_images = test_images.astype(np.float32) / 255.0
```

A normalização foi identificada e validada antes do benchmark. Em um teste de 1.000 imagens, a escala `[0, 1]` produziu aproximadamente 75,8% de acurácia, enquanto pixels brutos produziram aproximadamente 17,5%, confirmando a escala esperada pelo modelo.

Para os experimentos de 100, 1.000 e 10.000 imagens foi criada uma ordem estratificada e aninhada:

- 100 imagens: 10 por classe;
- 1.000 imagens: 100 por classe;
- 10.000 imagens: 1.000 por classe, isto é, todo o teste;
- as 100 imagens pertencem ao conjunto de 1.000, que pertence ao conjunto de 10.000;
- a seed fixa 20260825 torna a seleção reproduzível;
- dentro de cada ciclo, a ordem é permutada deterministicamente para evitar dependência de uma ordem fixa.

## Garantia de execução na GPU

O script não aceita fallback silencioso para CPU. Foram aplicadas as seguintes verificações:

1. `CUDA_VISIBLE_DEVICES=0` tornou visível somente a GPU selecionada.
2. `tf.config.list_physical_devices("GPU")` confirmou a existência de uma GPU TensorFlow.
3. `tf.config.set_visible_devices(..., "GPU")` manteve uma única GPU lógica.
4. `tf.config.set_soft_device_placement(False)` desativou realocação automática de operações para CPU.
5. Modelo e inferência foram criados sob `tf.device("/GPU:0")`.
6. A saída de cada inferência foi verificada; uma saída fora de `GPU:0` interromperia o ensaio.
7. `tf.config.experimental.enable_op_determinism()` foi habilitado.
8. As 100 repetições produziram exatamente a mesma contagem de classes corretas.

A CPU continua responsável pela orquestração Python, leitura dos dados, cronômetro, chamada do driver e gravação dos resultados. “Somente na GPU” refere-se ao cálculo do grafo da rede, não à inexistência de participação do host.

## Execução do modelo

O modelo Keras foi traçado uma única vez com `tf.function` e assinatura fixa `(1, 32, 32, 3)`. XLA foi explicitamente desativado com `jit_compile=False`. Isso remove a sobrecarga de interpretação eager a cada imagem sem alterar a arquitetura ou aplicar fusões XLA que dificultariam a comparação.

Antes da coleta foram realizadas 100 inferências de aquecimento. Elas inicializam o runtime, o grafo e os kernels CUDA e não entram nas métricas.

Para cada tamanho foram executados 100 ciclos completos. Os resultados de 10, 20 e 50 ciclos são prefixos cumulativos da mesma série de 100 ciclos. Portanto:

- 10.000 imagens × 10 ciclos = 100.000 latências;
- 10.000 imagens × 20 ciclos = 200.000 latências;
- 10.000 imagens × 50 ciclos = 500.000 latências;
- 10.000 imagens × 100 ciclos = 1.000.000 de latências.

O protocolo cumulativo evita executar conjuntos diferentes para cada quantidade de ciclos e permite observar a convergência conforme novas passagens são acrescentadas.

## Medição e cálculo da latência

Cada imagem foi processada com batch 1. A normalização e a seleção da imagem ocorreram fora do intervalo cronometrado. O intervalo medido foi:

```python
batch = tf.convert_to_tensor(image)
start = time.perf_counter_ns()
output = infer(batch)
host_output = output.numpy()
end = time.perf_counter_ns()
latency_ms = (end - start) / 1_000_000
```

`time.perf_counter_ns()` fornece um relógio monotônico de alta resolução. A chamada `.numpy()` força a sincronização da saída, impedindo que apenas o enfileiramento assíncrono do kernel seja medido.

A transferência do tensor é enfileirada antes do início do cronômetro, mas qualquer trabalho de transferência ainda pendente é necessariamente concluído antes da saída sincronizada. O valor representa a chamada batch 1 do grafo até a saída estar disponível no host. Carregamento do CIFAR-10, normalização global e escrita dos arquivos não entram na latência individual.

As estatísticas foram calculadas sobre todas as latências individuais do prefixo:

- média: soma das latências dividida pelo número de inferências;
- mediana: percentil 50;
- desvio padrão amostral: `numpy.std(..., ddof=1)`;
- p95: `numpy.percentile(..., 95)`;
- mínimo e máximo: extremos observados, sem remoção de outliers.

Nenhuma latência foi descartada ou aparada. Isso preserva pausas e oscilações reais observadas durante a execução.

O IC 95% da latência média foi calculado sobre as médias de cada ciclo:

```text
IC95 = média_dos_ciclos ± 1,9599639845 × desvio_dos_ciclos / sqrt(K)
```

onde `K` é 10, 20, 50 ou 100. Trata-se de uma aproximação normal aplicada aos valores independentes por ciclo.

## Medição e cálculo da vazão

O cronômetro de cada ciclo começa antes da primeira imagem e termina depois da última predição. Esse tempo inclui o laço host, criação dos tensores, transferências necessárias, chamada GPU, sincronização, `argmax` e armazenamento da predição em memória. Escrita dos checkpoints ocorre depois do fim do ciclo e não entra na vazão.

Para cada ciclo:

```text
vazão_do_ciclo = número_de_imagens / duração_do_ciclo
```

A vazão média exibida é a média aritmética das vazões dos ciclos. Também foram calculados:

- desvio padrão entre ciclos;
- IC 95% da vazão média;
- vazão efetiva global = total de inferências dividido pela soma dos tempos de todos os ciclos.

Por incluir mais atividades do caminho host–GPU, a vazão efetiva não é simplesmente `1000 / latência_média_ms`. As duas métricas descrevem escopos diferentes e ambas foram preservadas.

## Medição da acurácia

A classe prevista foi obtida com `numpy.argmax(output[0])`. A acurácia é:

```text
acurácia = predições_corretas / imagens_únicas
```

As mesmas imagens são repetidas para medir desempenho, mas cada imagem conta uma única vez para acurácia. Assim, 100 ciclos não transformam 10.000 imagens em 1.000.000 de amostras estatisticamente independentes de acurácia.

O intervalo de confiança de 95% foi calculado pelo método de Wilson, mais apropriado para uma proporção binomial que a aproximação simétrica simples. Para 7.489 acertos em 10.000 imagens, o resultado foi [74,0306%; 75,7303%].

## Telemetria de potência, energia e estado da GPU

A telemetria foi coletada por um processo persistente do `nvidia-smi`, executado a cada 100 ms com estes campos:

```text
timestamp
power.draw.instant
power.draw.average
utilization.gpu
clocks.current.sm
temperature.gpu
```

Uma thread Python leu continuamente a saída do processo. Cada amostra recebeu também um timestamp monotônico de `time.perf_counter()`, permitindo associá-la ao início e ao fim de cada ciclo.

Depois do aquecimento, a GPU permaneceu 5 segundos sem novas inferências medidas. Foram coletadas 50 amostras e calculada a potência ociosa média de 29,0630 W.

As métricas foram calculadas assim:

```text
potência_total_média = média de power.draw.instant durante o ciclo
potência_dinâmica = max(potência_total_média - potência_ociosa, 0)
energia_total = potência_total_média × duração_do_ciclo
energia_dinâmica = potência_dinâmica × duração_do_ciclo
energia_total_por_inferência_mJ = energia_total_J × 1000 / número_de_imagens
energia_dinâmica_por_inferência_mJ = energia_dinâmica_J × 1000 / número_de_imagens
```

Para cada prefixo, os valores apresentados são médias aritméticas dos valores coletados por ciclo. Utilização, clock SM e temperatura seguem a mesma agregação. `power_samples_total` informa quantas leituras contribuíram.

A documentação local do driver informa precisão aproximada de ±5 W para a leitura de potência. A potência também possui resolução temporal muito menor que a latência. Por isso:

- os valores de 10.000 imagens, com centenas ou milhares de amostras, são os mais defensáveis;
- ciclos de apenas 100 imagens são curtos demais para uma caracterização de potência igualmente robusta;
- energia é uma estimativa derivada da potência da placa e do tempo, não uma leitura direta de um medidor externo;
- a potência dinâmica depende da linha de base ociosa observada nesta sessão.

## Ferramentas e bibliotecas utilizadas

| Ferramenta ou biblioteca | Uso |
|---|---|
| TensorFlow 2.21.0 | Carregamento do modelo, runtime CUDA, grafo de inferência e carregamento do CIFAR-10 |
| Keras 3.13.2 | Desserialização do modelo HDF5 e execução das camadas |
| NumPy 1.26.4 | Normalização, seleção estratificada, `argmax`, arrays de latência e estatísticas |
| CUDA 12.5.1 | Backend de execução GPU do build TensorFlow utilizado |
| cuDNN 9 | Primitivas aceleradas de redes neurais utilizadas pelo TensorFlow |
| Python 3.12.7 | Orquestração do benchmark, checkpoints e agregação |
| `time.perf_counter_ns` | Cronometragem monotônica de cada inferência |
| `time.perf_counter` | Cronometragem dos ciclos e associação da telemetria |
| `nvidia-smi` | Potência, utilização, clock SM, temperatura e metadados da GPU |
| `subprocess` e `threading` | Processo persistente e leitura assíncrona da telemetria |
| `csv` e `json` | Persistência dos resultados e metadados |
| NPY/OpenMemmap do NumPy | Checkpoint das latências individuais sem perda de precisão float32 |
| SHA-256 | Verificação de integridade do modelo e arrays brutos |

Não foram usados TensorRT, XLA, quantização, mixed precision ou batching maior que 1.

## Estrutura dos arquivos

```text
GPU/resnet8-no-softmax/
├── README.md
├── benchmark_no_softmax_gpu.py
├── validar_equivalencia.py
└── resultados/
    ├── RESULTADOS.md
    ├── resultados.csv
    ├── metadados.json
    ├── validacao_equivalencia.json
    ├── passagens_n100.csv
    ├── passagens_n1000.csv
    ├── passagens_n10000.csv
    ├── latencias_n100.npy
    ├── latencias_n1000.npy
    └── latencias_n10000.npy
```

Descrição:

- `README.md`: documentação completa e resultado recomendado;
- `benchmark_no_softmax_gpu.py`: benchmark reproduzível e retomável;
- `validar_equivalencia.py`: validação funcional batch 1 contra o modelo com softmax;
- `resultados/RESULTADOS.md`: tabela compacta das 12 combinações;
- `resultados/resultados.csv`: todas as 33 colunas agregadas;
- `resultados/metadados.json`: hardware, software, seed e parâmetros;
- `resultados/validacao_equivalencia.json`: equivalência de pesos, classes e probabilidades;
- `resultados/passagens_n*.csv`: uma linha por ciclo, 100 linhas por tamanho;
- `resultados/latencias_n*.npy`: latências individuais em milissegundos.

## Reprodução

A partir da raiz do projeto:

```bash
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=2 \
python3 GPU/resnet8-no-softmax/benchmark_no_softmax_gpu.py
```

Validação funcional reproduzível:

```bash
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=2 \
python3 GPU/resnet8-no-softmax/validar_equivalencia.py
```

O script usa por padrão:

- modelo: `resnet8_cifar10_keras3_no_softmax.h5`, na raiz do projeto;
- saída: `GPU/resnet8-no-softmax/resultados/`;
- tamanhos: 100, 1.000 e 10.000;
- prefixos: 10, 20, 50 e 100;
- batch: 1;
- aquecimento: 100 inferências;
- telemetria: 100 ms;
- linha de base: 5 segundos;
- seed: 20260825.

O script salva checkpoint ao final de cada ciclo. Se for interrompido, o mesmo comando continua a partir do próximo ciclo incompleto. A opção `--restart` descarta os checkpoints dos tamanhos solicitados e deve ser usada somente quando uma coleta totalmente nova for desejada.

## Integridade e validações realizadas

Após a execução foram verificados:

- exatamente 12 resultados agregados: 3 tamanhos × 4 prefixos;
- exatamente 100 registros por tamanho nos CSVs de ciclos;
- formatos dos arrays: `(100, 100)`, `(100, 1000)` e `(100, 10000)`;
- total de 1.110.000 latências individuais;
- todos os valores de latência finitos e positivos;
- nenhuma divergência de predição entre ciclos;
- equivalência de classe em 10.000 de 10.000 imagens entre logits e softmax;
- igualdade exata dos 47 arrays de pesos;
- saída de inferência em `GPU:0`;
- ausência de erros da thread de telemetria;
- igualdade entre médias, medianas e p95 recomputados dos NPYs e os valores do CSV.

Hashes dos arrays brutos:

| Arquivo | SHA-256 |
|---|---|
| `latencias_n100.npy` | `f2e1228cd8332d6d5d94606a85c0e8ce98985935219d3e3f5cf528155d64fa71` |
| `latencias_n1000.npy` | `d34ff8c13e88412b4fed02f2b58b074bcb89e3c3de1c109d8eccf329c98c256b` |
| `latencias_n10000.npy` | `b3c3fb943589802cc67f9bb212006c428b4ae8a6c64c5645c4c038388d223450` |

## Limitações e cuidados de interpretação

- O ensaio representa esta RTX 3050, este driver, esta versão do TensorFlow e as condições térmicas e de carga da sessão.
- Clock e limite de potência não foram travados. Variações de DVFS fazem parte dos resultados observados.
- Não foi usado um wattímetro externo; potência e energia dependem do sensor NVIDIA.
- O sistema operacional e o host podem introduzir caudas de latência. Elas não foram removidas.
- A aproximação normal dos intervalos das médias usa ciclos como unidades; os ciclos são executados sequencialmente na mesma sessão.
- Os prefixos de 10, 20, 50 e 100 não são quatro experimentos independentes: são séries cumulativas intencionalmente pareadas.
- As coletas com e sem softmax foram realizadas em sessões sequenciais, não intercaladas; diferenças de desempenho observadas não devem ser atribuídas exclusivamente à remoção da softmax sem um ensaio comparativo pareado.
- Acurácia e energia da GPU não devem ser comparadas diretamente a uma implementação FPGA sem manter exatamente dataset, normalização, batch, definição do intervalo cronometrado e escopo de potência.
- Para a comparação principal com FPGA, recomenda-se usar a linha de 10.000 imagens e 100 ciclos, mantendo mediana e p95 além da média.
