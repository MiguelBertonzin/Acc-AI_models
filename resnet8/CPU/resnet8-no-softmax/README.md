# ResNet-8 sem softmax — benchmark de inferência na CPU

## Tabela completa do resultado principal

O resultado recomendado usa todas as 10.000 imagens do teste oficial do CIFAR-10 e 100 ciclos completos. São 1.000.000 de inferências cronometradas com batch 1. Os resultados de 10, 20 e 50 ciclos são prefixos da mesma série de 100 ciclos.

| Métrica coletada | Resultado principal |
|---|---:|
| Variante | ResNet-8 sem softmax |
| Ativação de saída | linear, dez logits |
| Dispositivo real da saída | CPU:0 |
| CPU | 13th Gen Intel Core i7-13700 |
| Núcleos físicos / CPUs lógicas | 16 / 24 |
| Precisão numérica | float32 |
| Batch | 1 |
| Imagens únicas | 10.000 |
| Ciclos completos | 100 |
| Inferências cronometradas | 1.000.000 |
| Predições corretas | 7.490 |
| Acurácia | 74,9000% |
| IC 95% da acurácia, Wilson | [74,0407%; 75,7401%] |
| Latência média | 0,9760 ms |
| IC 95% da média entre ciclos | [0,9750; 0,9769] ms |
| Latência mediana | 0,9028 ms |
| Desvio-padrão amostral da latência | 0,2447 ms |
| Latência p95 | 1,5704 ms |
| Latência mínima | 0,7341 ms |
| Latência máxima | 6,1157 ms |
| Vazão média entre ciclos | 975,0379 inferências/s |
| Desvio-padrão da vazão entre ciclos | 4,9508 inferências/s |
| IC 95% da vazão média | [974,0675; 976,0082] inferências/s |
| Vazão efetiva global | 975,0129 inferências/s |
| Tempo total dos ciclos medidos | 1.025,6274 s |
| Tempo de CPU acumulado | 7.320,7580 s |
| Tempo de CPU acumulado por inferência | 7,3208 ms |
| Utilização média do sistema, psutil | 29,7089% |
| Utilização média do processo, psutil | 713,2470% |
| Utilização do processo normalizada por 24 CPUs | 29,7186% |
| Busy médio da CPU, turbostat | 34,8733% |
| Frequência média durante ciclos ocupados, Bzy_MHz | 3.364,4939 MHz |
| Frequência média do psutil | N/A — unidade inválida no kernel local |
| Temperatura média do pacote, turbostat | 71,8508 °C |
| Temperatura média do pacote, psutil/coretemp | 71,0124 °C |
| Memória residente média do processo | 992,0268 MiB |
| Potência média do pacote, RAPL | 57,4144 W |
| Potência média dos núcleos, RAPL | 47,4848 W |
| Potência dinâmica média do pacote | 44,5866 W |
| Potência DRAM | N/A — domínio não exposto |
| Energia total do pacote por inferência | 58,3599 mJ |
| Energia dinâmica por inferência | 45,2034 mJ |
| Potência ociosa usada como linha de base | 12,8278 W |
| Amostras RAPL referenciadas pelos 100 ciclos | 10.265 |
| Amostras internas de psutil nos 100 ciclos | 6.424 |
| Resolução solicitada da telemetria | 100 ms |
| LUT / FF / DSP / BRAM / URAM | N/A para execução em CPU |

Esta linha é a referência mais consistente: cobre o conjunto de teste inteiro, um regime térmico sustentado e mais de dez mil referências de amostras RAPL. Os subconjuntos curtos permanecem nos arquivos brutos, mas não têm a mesma resolução energética.

## Convergência em 10.000 imagens

| Ciclos | Inferências | Latência média (ms) | IC 95% da média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | FPS médio | Pacote (W) | Dinâmica (W) | Energia/inf. (mJ) | Energia dinâmica/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100.000 | 0,9796 | [0,9761; 0,9831] | 0,9059 | 0,2455 | 1,5698 | 971,4519 | 57,8863 | 45,0585 | 59,0303 | 45,8252 |
| 20 | 200.000 | 0,9788 | [0,9767; 0,9809] | 0,9048 | 0,2464 | 1,5733 | 971,6647 | 57,7950 | 44,9672 | 58,9255 | 45,7233 |
| 50 | 500.000 | 0,9765 | [0,9752; 0,9779] | 0,9032 | 0,2446 | 1,5695 | 974,4025 | 57,5718 | 44,7441 | 58,5493 | 45,3842 |
| **100** | **1.000.000** | **0,9760** | **[0,9750; 0,9769]** | **0,9028** | **0,2447** | **1,5704** | **975,0379** | **57,4144** | **44,5866** | **58,3599** | **45,2034** |

Os mínimos, máximos, intervalos de confiança da vazão, tempos de CPU, ocupação, frequência, temperaturas, memória e amostras de todas as 12 combinações estão em [resultados/resultados.csv](resultados/resultados.csv).

## Tamanho do conjunto e acurácia

| Imagens únicas | Corretas | Acurácia | IC 95% de Wilson |
|---:|---:|---:|---:|
| 100 | 74 | 74,00% | [64,6290%; 81,5953%] |
| 1.000 | 741 | 74,10% | [71,2962%; 76,7194%] |
| **10.000** | **7.490** | **74,90%** | **[74,0407%; 75,7401%]** |

Os conjuntos são estratificados, balanceados e aninhados. Repetições medem desempenho, não criam novas observações de acurácia; cada imagem única é contada uma vez no intervalo de Wilson.

## Modelo e validação funcional da remoção da softmax

O artefato avaliado é `resnet8_cifar10_keras3_no_softmax.h5`, com entrada `(1, 32, 32, 3)`, saída `(1, 10)`, 78.714 parâmetros e ativação final linear. O arquivo possui 416.384 bytes e SHA-256:

```text
047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e
```

A equivalência com `resnet8_cifar10_keras3.h5` foi validada separadamente na CPU, nas 10.000 imagens, batch 1:

| Verificação | Resultado |
|---|---:|
| Ativação original | softmax |
| Ativação avaliada | linear |
| Arrays de pesos comparados | 47 |
| Erro absoluto máximo entre pesos | 0 |
| Classes iguais | 10.000 de 10.000 |
| Classes divergentes | 0 |
| Acurácia com softmax | 74,90% |
| Acurácia com logits | 74,90% |
| Erro máximo ao reconstruir softmax(logits) | 2,3841858 × 10⁻⁷ |
| Erro médio absoluto da reconstrução | 5,5136735 × 10⁻⁹ |
| Dispositivo da validação | CPU:0 |
| GPU visível na validação | nenhuma |
| Resultado | aprovado |

A classe prevista é `argmax(logits)`. Como softmax é monotônica em relação aos logits, a remoção não deveria alterar o índice máximo; o teste empírico confirma isso em todo o conjunto. Evidência completa: [resultados/validacao_equivalencia.json](resultados/validacao_equivalencia.json). Código reproduzível: [validar_equivalencia.py](validar_equivalencia.py).

## Objetivo e relação com FPGA

O objetivo é estabelecer uma referência CPU float32 e batch 1 comparável ao futuro caminho em hls4ml/Vitis. Não foram usados quantização, mixed precision, XLA ou batching.

LUT, FF, DSP, BRAM e URAM, solicitados no `info.txt`, não se aplicam a um modelo executado pelo TensorFlow em CPU. `Bzy_MHz` é a frequência ocupada observada da CPU; ela não deve ser comparada diretamente com a frequência de síntese de uma implementação FPGA.

## Ambiente

| Componente | Valor |
|---|---|
| CPU | 13th Gen Intel Core i7-13700 |
| Topologia | 16 núcleos físicos, 24 CPUs lógicas |
| Afinidade permitida | CPUs 0–23 |
| Política de threads TensorFlow | automática; intra-op=0 e inter-op=0 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| NumPy | 1.26.4 |
| psutil | 5.9.0 |
| Python | 3.12.7 |
| Sistema | Linux 7.0.0-30-generic x86_64, glibc 2.39 |
| Backend CPU | TensorFlow com oneDNN habilitado |
| Seed | 20260825 |
| Início | 2026-08-25 20:02:49 UTC |
| Fim | 2026-08-25 20:21:53 UTC |
| Log turbostat | SHA-256 `9ff82abd8d258b6ad427788a0046e9d03453d8b1cab73e2383132560e69cf480` |

## Dataset e amostragem

O teste oficial do CIFAR-10 foi carregado por `tf.keras.datasets.cifar10.load_data()` e normalizado para float32:

```python
test_images = test_images.astype(np.float32) / 255.0
```

Uma ordem estratificada com seed 20260825 produziu:

- 100 imagens, 10 por classe;
- 1.000 imagens, 100 por classe;
- 10.000 imagens, 1.000 por classe;
- inclusão aninhada entre os três conjuntos;
- permutação determinística diferente dentro de cada ciclo.

## Garantia de execução exclusiva na CPU

1. `CUDA_VISIBLE_DEVICES` foi definido como vazio antes do TensorFlow.
2. `tf.config.set_visible_devices([], "GPU")` ocultou GPUs.
3. O soft placement foi desativado.
4. Modelo, grafo e chamadas ficaram sob `tf.device("/CPU:0")`.
5. Cada saída foi verificada; uma saída fora de `CPU:0` abortaria.
6. Os metadados registraram `visible_tensorflow_gpus: []`.
7. O dispositivo real foi `/device:CPU:0`.
8. O determinismo do TensorFlow foi habilitado.

O aviso de inicialização `CUDA_ERROR_NO_DEVICE` confirma ausência de CUDA utilizável; todas as saídas foram verificadas na CPU.

## Protocolo de execução

O modelo foi traçado uma vez com `tf.function`, assinatura `(1, 32, 32, 3)` e `jit_compile=False`. Foram realizadas 100 inferências de aquecimento, excluídas das métricas.

Cada tamanho recebeu 100 ciclos completos. Os resultados publicados de 10/20/50 são prefixos cumulativos da mesma matriz de 100 ciclos:

```text
10.000 × 10  =   100.000 latências
10.000 × 20  =   200.000 latências
10.000 × 50  =   500.000 latências
10.000 × 100 = 1.000.000 latências
```

Não houve remoção de outliers. Cada ciclo foi persistido em CSV, e cada latência individual foi gravada em uma matriz NPY retomável.

## Latência e estatística

```python
batch = tf.convert_to_tensor(image)
t0 = time.perf_counter_ns()
output = infer(batch)
host_output = output.numpy()
t1 = time.perf_counter_ns()
latency_ms = (t1 - t0) / 1_000_000
```

A imagem já normalizada e sua seleção ficam fora do cronômetro. O intervalo cobre a chamada batch 1 do grafo e a materialização da saída no host. `perf_counter_ns()` é monotônico; `.numpy()` impede medir somente o despacho.

Média, mediana, desvio-padrão amostral, p95, mínimo e máximo são calculados sobre todas as latências individuais do prefixo. O IC 95% usa as médias por ciclo:

```text
IC95 = média_das_médias ± 1,9599639845 × desvio_das_médias / sqrt(K)
```

`K` vale 10, 20, 50 ou 100, tornando o ciclo completo a unidade estatística para a estabilidade da média.

## Vazão, utilização e tempo de CPU

```text
FPS_ciclo = imagens / duração_de_parede_do_ciclo
FPS_efetivo = total_de_inferências / soma_das_durações
```

A duração do ciclo inclui laço Python, tensor, inferência, sincronização, `argmax` e armazenamento da predição, mas exclui escrita posterior do checkpoint.

`process_time()` acumula o consumo de todas as threads: 7,3208 ms de CPU por inferência coexistem com 0,9760 ms de parede porque múltiplos núcleos trabalham em paralelo. `process.cpu_percent()` também soma núcleos; 713,2470% corresponde a 29,7186% dos 24 processadores lógicos disponíveis.

## Potência, energia, frequência e temperatura

Um processo root independente coletou, a cada 100 ms:

```text
Time_Of_Day_Seconds  Busy%  Bzy_MHz  PkgTmp  PkgWatt  CorWatt
```

Comando usado:

```bash
sudo turbostat --quiet --no-perf --Summary --debug \
  --interval 0.1 \
  --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt \
  --out /tmp/resnet8_cpu_turbostat.log
```

O benchmark não recebeu ou armazenou senha. Uma cópia imutável está em [resultados/turbostat_bruto.log](resultados/turbostat_bruto.log). Foram parseadas 24.743 amostras no log compartilhado pelas duas sessões; timestamps delimitam exatamente quais pertencem a cada modelo.

Cada leitura em `t_i` representa `[t_i − 0,1; t_i]`. Para a passagem `[a,b]`:

```text
w_i = duração([t_i − 0,1; t_i] ∩ [a; b])
potência_média = Σ(P_i × w_i) / Σ(w_i)
energia_J = Σ(P_i × w_i)
```

Se a cobertura fosse menor que 98%, a integral seria ajustada por `duração/cobertura`; cobertura inferior a metade de uma amostra causaria erro. A linha de base usa 50 leituras sobre os 5 s ociosos após o aquecimento:

```text
potência_ociosa = 12,8278 W
potência_dinâmica = max(potência_pacote − potência_ociosa, 0)
energia_total_por_inferência_mJ = energia_pacote_J × 1000 / N
energia_dinâmica_por_inferência_mJ =
    max(energia_pacote_J − potência_ociosa × duração, 0) × 1000 / N
```

`PkgWatt` já contém `CorWatt`; os campos não devem ser somados. DRAM RAPL não estava disponível. Potência e energia dos prefixos são médias aritméticas dos valores calculados por ciclo.

O campo `cpu_freq()` do psutil foi marcado como N/A por unidade inconsistente no kernel local. A frequência válida publicada é `Bzy_MHz`, medida pelo `turbostat`. Temperatura também foi coletada por `PkgTmp` e, como verificação secundária, por `coretemp`.

## Comparação responsável com a variante softmax

Na mesma máquina, o resultado softmax foi 0,9551 ms e o resultado sem softmax foi 0,9760 ms. A diferença observada é aproximadamente +2,19% para logits nesta coleta, apesar de a remoção da operação parecer intuitivamente mais barata.

Isso não demonstra que softmax acelera causalmente a rede. As sessões foram sequenciais, não pareadas nem intercaladas, e podem diferir em escalonamento, DVFS, estado de cache, temperatura e escolhas internas do oneDNN. A conclusão defensável é:

- cada resultado é consistente dentro de sua própria série de 100 ciclos;
- as classes e pesos são funcionalmente equivalentes;
- uma atribuição causal à camada final exigiria rodadas alternadas/pareadas adicionais.

Para comparação FPGA, cada linha deve ser tratada como referência empírica da variante completa correspondente.

## Ferramentas e bibliotecas

| Ferramenta | Uso |
|---|---|
| TensorFlow 2.21.0 | Grafo CPU, execução e CIFAR-10 |
| Keras 3.13.2 | Modelos HDF5 e camadas |
| oneDNN | Kernels CPU |
| NumPy 1.26.4 | Normalização, logits, probabilidades e estatística |
| psutil 5.9.0 | Ocupação, RSS e coretemp |
| turbostat | Busy, Bzy_MHz, PkgTmp e RAPL |
| `time.perf_counter_ns` | Latência individual |
| `time.process_time` | Tempo de CPU acumulado |
| `threading` | Telemetria psutil concorrente |
| CSV / JSON | Passagens, agregados, metadados e validação |
| NPY/OpenMemmap | Matrizes de latência |
| SHA-256 | Integridade dos modelos e log |

## Limitações

- RAPL mede energia estimada do pacote, não consumo na tomada.
- Trabalho do sistema concorrente dentro da janela também afeta `PkgWatt`.
- Energia dinâmica depende da baseline desta sessão.
- A resolução de 100 ms é muito maior que uma inferência; ciclos longos tornam a integração útil.
- DRAM não foi exposta, logo permanece N/A.
- Threads automáticas do TensorFlow devem ser mantidas em reproduções.
- As sessões softmax/logits não foram intercaladas, portanto a comparação não isola apenas a última ativação.

## Arquivos

```text
CPU/resnet8-no-softmax/
├── README.md
├── benchmark_no_softmax_cpu.py
├── enriquecer_rapl.py
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
    ├── latencias_n10000.npy
    ├── turbostat_bruto.log
    └── turbostat_amostras.csv
```

- `resultados.csv`: 12 combinações agregadas;
- `passagens_n*.csv`: 100 ciclos por tamanho e seus timestamps;
- `latencias_n*.npy`: 100 × N latências individuais;
- `validacao_equivalencia.json`: prova de pesos, classes e probabilidades;
- `turbostat_bruto.log`: telemetria original congelada;
- `turbostat_amostras.csv`: amostras normalizadas;
- `metadados.json`: ambiente, baseline, hashes e protocolo.

## Reprodução

Iniciar o `turbostat` em um terminal:

```bash
sudo turbostat --quiet --no-perf --Summary --debug \
  --interval 0.1 \
  --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt \
  --out /tmp/resnet8_cpu_turbostat.log
```

Executar benchmark e validação em outro terminal:

```bash
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \
python3 CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py --restart

CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \
python3 CPU/resnet8-no-softmax/validar_equivalencia.py --batch-size 1
```

Após `Ctrl+C` no `turbostat`:

```bash
python3 CPU/resnet8-no-softmax/enriquecer_rapl.py \
  --log /tmp/resnet8_cpu_turbostat.log
```

Sem `--restart`, o benchmark pode retomar checkpoints existentes; com a opção, inicia uma sessão limpa apenas nesta pasta.
