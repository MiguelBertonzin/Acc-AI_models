# ResNet-8 com softmax — benchmark de inferência na CPU

## Tabela completa do resultado principal

O resultado recomendado usa todas as 10.000 imagens do teste oficial do CIFAR-10 e 100 ciclos completos. São 1.000.000 de inferências cronometradas com batch 1. Os resultados de 10, 20 e 50 ciclos são prefixos da mesma série, portanto não misturam sessões independentes.

| Métrica coletada | Resultado principal |
|---|---:|
| Variante | ResNet-8 com softmax |
| Ativação de saída | softmax |
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
| Latência média | 0,9551 ms |
| IC 95% da média entre ciclos | [0,9532; 0,9570] ms |
| Latência mediana | 0,8843 ms |
| Desvio-padrão amostral da latência | 0,2371 ms |
| Latência p95 | 1,5290 ms |
| Latência mínima | 0,7218 ms |
| Latência máxima | 6,2877 ms |
| Vazão média entre ciclos | 997,5456 inferências/s |
| Desvio-padrão da vazão entre ciclos | 10,1833 inferências/s |
| IC 95% da vazão média | [995,5497; 999,5415] inferências/s |
| Vazão efetiva global | 997,4433 inferências/s |
| Tempo total dos ciclos medidos | 1.002,5632 s |
| Tempo de CPU acumulado | 7.000,2484 s |
| Tempo de CPU acumulado por inferência | 7,0002 ms |
| Utilização média do sistema, psutil | 29,1217% |
| Utilização média do processo, psutil | 697,5423% |
| Utilização do processo normalizada por 24 CPUs | 29,0643% |
| Busy médio da CPU, turbostat | 34,1735% |
| Frequência média durante ciclos ocupados, Bzy_MHz | 3.367,6017 MHz |
| Frequência média do psutil | N/A — unidade inválida no kernel local |
| Temperatura média do pacote, turbostat | 71,8976 °C |
| Temperatura média do pacote, psutil/coretemp | 71,1078 °C |
| Memória residente média do processo | 993,4258 MiB |
| Potência média do pacote, RAPL | 57,3368 W |
| Potência média dos núcleos, RAPL | 47,5384 W |
| Potência dinâmica média do pacote | 44,5380 W |
| Potência DRAM | N/A — domínio não exposto |
| Energia total do pacote por inferência | 56,9807 mJ |
| Energia dinâmica por inferência | 44,1491 mJ |
| Potência ociosa usada como linha de base | 12,7988 W |
| Amostras RAPL referenciadas pelos 100 ciclos | 10.035 |
| Amostras internas de psutil nos 100 ciclos | 6.294 |
| Resolução solicitada da telemetria | 100 ms |
| LUT / FF / DSP / BRAM / URAM | N/A para execução em CPU |

Esta é a linha mais defensável para comparação com FPGA: usa o conjunto de teste inteiro, o maior tempo de observação e a maior quantidade de amostras de potência. Resultados de subconjuntos curtos continuam disponíveis, mas são mais sensíveis à granularidade de 100 ms da telemetria.

## Convergência em 10.000 imagens

| Ciclos | Inferências | Latência média (ms) | IC 95% da média (ms) | Mediana (ms) | Desvio (ms) | p95 (ms) | FPS médio | Pacote (W) | Dinâmica (W) | Energia/inf. (mJ) | Energia dinâmica/inf. (mJ) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100.000 | 0,9613 | [0,9587; 0,9639] | 0,8893 | 0,2371 | 1,5375 | 990,6445 | 57,8163 | 45,0176 | 57,8400 | 44,9201 |
| 20 | 200.000 | 0,9597 | [0,9567; 0,9628] | 0,8883 | 0,2375 | 1,5357 | 992,2750 | 57,8680 | 45,0692 | 57,8061 | 44,9070 |
| 50 | 500.000 | 0,9573 | [0,9552; 0,9593] | 0,8862 | 0,2370 | 1,5329 | 995,0256 | 57,6252 | 44,8264 | 57,4047 | 44,5411 |
| **100** | **1.000.000** | **0,9551** | **[0,9532; 0,9570]** | **0,8843** | **0,2371** | **1,5290** | **997,5456** | **57,3368** | **44,5380** | **56,9807** | **44,1491** |

Os mínimos, máximos, intervalos de confiança da vazão, ocupação, frequência, temperaturas, tempos de CPU, memória e contagens de amostras de todas as 12 combinações estão em [resultados/resultados.csv](resultados/resultados.csv).

## Tamanho do conjunto e acurácia

Os conjuntos são estratificados e aninhados: as 100 imagens estão contidas nas 1.000, que estão contidas nas 10.000. A acurácia é calculada apenas nas imagens únicas; repetições não são tratadas como novas observações de acurácia.

| Imagens únicas | Corretas | Acurácia | IC 95% de Wilson |
|---:|---:|---:|---:|
| 100 | 74 | 74,00% | [64,6290%; 81,5953%] |
| 1.000 | 741 | 74,10% | [71,2962%; 76,7194%] |
| **10.000** | **7.490** | **74,90%** | **[74,0407%; 75,7401%]** |

O conjunto completo elimina erro de seleção de subconjunto e produz o intervalo de Wilson mais estreito. As 100 repetições caracterizam estabilidade temporal, DVFS, escalonamento e temperatura; elas não estreitam artificialmente o intervalo da acurácia.

## Modelo e objetivo

O artefato avaliado é `resnet8_cifar10_keras3.h5`, com entrada `(1, 32, 32, 3)`, dez probabilidades de saída, 78.714 parâmetros e ativação final softmax. O arquivo possui 1.115.216 bytes e SHA-256:

```text
9cc7cd3ea1c9603501f0c77dc0848b41d0bd136f80cc1d328881ed89cbb9cb1b
```

O objetivo é estabelecer uma referência CPU float32, batch 1, comparável ao caminho futuro em hls4ml/Vitis. Não foram usados quantização, mixed precision, XLA ou batch maior que 1.

As métricas de recursos listadas no `info.txt` — LUT, FF, DSP, BRAM e URAM — pertencem à implementação FPGA e não existem neste ensaio. A frequência `Bzy_MHz` é a frequência ocupada observada da CPU e não equivale à frequência de clock atingida por uma síntese FPGA.

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
| Início | 2026-08-25 19:41:17 UTC |
| Fim | 2026-08-25 19:59:56 UTC |
| Log turbostat | SHA-256 `9ff82abd8d258b6ad427788a0046e9d03453d8b1cab73e2383132560e69cf480` |

## Preparação do CIFAR-10

O dataset foi carregado por `tf.keras.datasets.cifar10.load_data()`. Foi usada a divisão oficial de teste: 10.000 imagens RGB de 32 × 32 pixels, com 1.000 elementos por classe.

```python
test_images = test_images.astype(np.float32) / 255.0
```

A ordem estratificada foi construída com seed 20260825. Cada ciclo recebe uma permutação determinística própria, evitando dependência de uma ordem fixa. Os três tamanhos são balanceados:

- 100 imagens: 10 por classe;
- 1.000 imagens: 100 por classe;
- 10.000 imagens: 1.000 por classe.

## Garantia de execução exclusiva na CPU

O benchmark aplica verificações explícitas:

1. `CUDA_VISIBLE_DEVICES` é vazio antes da importação do TensorFlow.
2. `tf.config.set_visible_devices([], "GPU")` oculta GPUs.
3. O soft placement é desativado.
4. Modelo, grafo e chamadas são executados sob `tf.device("/CPU:0")`.
5. A saída de cada inferência é verificada; uma saída fora de `CPU:0` aborta o ensaio.
6. Os metadados finais registraram `visible_tensorflow_gpus: []`.
7. O dispositivo real registrado foi `/device:CPU:0`.
8. Determinismo do TensorFlow foi habilitado.

O aviso `CUDA_ERROR_NO_DEVICE` emitido durante a inicialização informa justamente que não havia dispositivo CUDA utilizável; não ocorreu fallback de uma inferência GPU.

## Execução e protocolo estatístico

O modelo foi traçado uma vez com `tf.function`, assinatura fixa `(1, 32, 32, 3)` e `jit_compile=False`. Cem inferências de aquecimento inicializaram runtime, grafo e kernels oneDNN e não entraram nas estatísticas.

Para cada tamanho foram executados 100 ciclos. Os pontos de 10, 20 e 50 são prefixos cumulativos:

```text
10.000 × 10  =   100.000 latências
10.000 × 20  =   200.000 latências
10.000 × 50  =   500.000 latências
10.000 × 100 = 1.000.000 latências
```

Nenhuma latência foi removida, aparada ou classificada como outlier. Checkpoints CSV e NPY são gravados após cada ciclo, permitindo auditoria e retomada.

## Latência e suas estatísticas

A seleção e normalização da imagem ocorrem antes do trecho cronometrado:

```python
batch = tf.convert_to_tensor(image)
t0 = time.perf_counter_ns()
output = infer(batch)
host_output = output.numpy()
t1 = time.perf_counter_ns()
latency_ms = (t1 - t0) / 1_000_000
```

`perf_counter_ns()` é monotônico e de alta resolução. `.numpy()` materializa a saída antes do fim, de modo que o tempo não representa apenas despacho assíncrono. O intervalo cobre a chamada do grafo batch 1 e a disponibilização da saída no host; carregamento global do dataset, normalização e escrita dos checkpoints ficam fora.

Sobre todas as latências individuais de cada prefixo são calculados média, mediana, desvio-padrão amostral (`ddof=1`), p95, mínimo e máximo.

O IC 95% da média usa as médias dos ciclos:

```text
IC95 = média_das_médias ± 1,9599639845 × desvio_das_médias / sqrt(K)
```

`K` vale 10, 20, 50 ou 100. A unidade de repetição para esse intervalo é o ciclo completo, e não cada imagem repetida.

## Vazão e tempo de CPU

Para cada ciclo:

```text
FPS_ciclo = imagens / duração_de_parede_do_ciclo
```

A duração inclui o laço Python, conversão individual em tensor, chamada, sincronização, `argmax` e armazenamento da classe. A vazão média é a média dos FPS dos ciclos; a vazão efetiva é o total de inferências dividido pela soma das durações.

`time.process_time()` acumula tempo consumido por todas as threads. Por isso os 7,0002 ms de CPU por inferência podem ser maiores que os 0,9551 ms de parede: aproximadamente sete núcleos trabalharam simultaneamente. De forma análoga, `process.cpu_percent()` pode exceder 100%; o valor normalizado divide 697,5423% pelas 24 CPUs lógicas.

## Potência, energia, frequência e temperatura

A coleta privilegiada foi executada fora do processo Python:

```bash
sudo turbostat --quiet --no-perf --Summary --debug \
  --interval 0.1 \
  --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt \
  --out /tmp/resnet8_cpu_turbostat.log
```

O `turbostat` leu os contadores MSR/RAPL como root. O benchmark não recebeu nem armazenou a senha. O log foi congelado em [resultados/turbostat_bruto.log](resultados/turbostat_bruto.log); 24.743 amostras válidas foram analisadas.

Cada amostra terminada em `t_i` representa aproximadamente o intervalo `[t_i − 0,1 s; t_i]`. Para uma passagem `[a,b]`, o peso é a duração da interseção:

```text
w_i = duração([t_i − 0,1; t_i] ∩ [a; b])
potência_média = Σ(P_i × w_i) / Σ(w_i)
energia_J = Σ(P_i × w_i)
```

Quando a cobertura ficou abaixo de 98% da duração, a integral foi corrigida proporcionalmente por `duração/cobertura`. Todas as passagens tiveram cobertura suficiente; uma cobertura menor que metade de uma amostra abortaria a consolidação.

Após o aquecimento houve uma janela ociosa de 5 s. As 50 amostras sobrepostas produziram a linha de base de 12,7988 W:

```text
potência_dinâmica = max(potência_pacote − potência_ociosa, 0)
energia_total_por_inferência_mJ = energia_pacote_J × 1000 / N
energia_dinâmica_por_inferência_mJ =
    max(energia_pacote_J − potência_ociosa_W × duração_s, 0) × 1000 / N
```

`PkgWatt` é potência do pacote completo; `CorWatt` é o subconjunto atribuído aos núcleos. Não se deve somar os dois. O domínio de DRAM não foi exposto pelo equipamento e foi registrado como N/A.

`Busy%`, `Bzy_MHz` e `PkgTmp` vêm do mesmo fluxo temporal. O campo de frequência do psutil foi descartado porque este kernel devolveu valores `current` em escala incompatível (por exemplo, 3,5 com limites 800–4.783 MHz). A frequência publicada é exclusivamente `Bzy_MHz`, cuja unidade foi confirmada pelo `turbostat`.

## Ferramentas e bibliotecas

| Ferramenta | Uso |
|---|---|
| TensorFlow 2.21.0 | Modelo, grafo CPU, execução e CIFAR-10 |
| Keras 3.13.2 | Desserialização HDF5 e camadas |
| oneDNN | Kernels CPU usados pelo TensorFlow |
| NumPy 1.26.4 | Normalização, seleção, latências e estatística |
| psutil 5.9.0 | Ocupação, RSS e temperatura via coretemp |
| turbostat | Busy, Bzy_MHz, PkgTmp e leitura MSR/RAPL |
| `time.perf_counter_ns` | Latência individual monotônica |
| `time.process_time` | Tempo de CPU acumulado por processo |
| `threading` | Amostragem psutil paralela ao benchmark |
| CSV / JSON | Passagens, agregados e metadados |
| NPY/OpenMemmap | Latências individuais sem arredondamento textual |
| SHA-256 | Integridade de modelo e log bruto |

## Limitações e interpretação

- RAPL estima energia interna do pacote; não mede consumo na tomada, VRM, memória externa ou perdas da fonte.
- A energia por inferência inclui o pacote inteiro e atividades concorrentes do sistema durante a janela.
- Potência dinâmica depende da linha de base ociosa desta sessão.
- A resolução energética de 100 ms é muito mais grossa que uma inferência; a robustez vem da integração de ciclos longos.
- Escalonador, DVFS, temperatura e processos concorrentes continuam sendo fontes reais de variabilidade.
- A política automática de threads favorece o desempenho padrão do TensorFlow, mas deve ser mantida igual em uma reprodução.
- Resultados CPU e GPU podem diferir por ordem de operações float32/oneDNN; a acurácia CPU observada foi 7.490/10.000.

## Arquivos

```text
CPU/resnet8-softmax/
├── README.md
├── benchmark_softmax_cpu.py
├── enriquecer_rapl.py
└── resultados/
    ├── RESULTADOS.md
    ├── resultados.csv
    ├── metadados.json
    ├── passagens_n100.csv
    ├── passagens_n1000.csv
    ├── passagens_n10000.csv
    ├── latencias_n100.npy
    ├── latencias_n1000.npy
    ├── latencias_n10000.npy
    ├── turbostat_bruto.log
    └── turbostat_amostras.csv
```

- `resultados.csv`: 12 linhas agregadas (3 tamanhos × 4 prefixos);
- `passagens_n*.csv`: 100 linhas por tamanho, incluindo timestamps e métricas por ciclo;
- `latencias_n*.npy`: matrizes 100 × N de latências em ms;
- `turbostat_bruto.log`: evidência original congelada;
- `turbostat_amostras.csv`: 24.743 amostras parseadas;
- `metadados.json`: ambiente, protocolo, baseline, hashes e fontes;
- `RESULTADOS.md`: visão compacta gerada automaticamente.

## Reprodução

Na raiz do projeto, iniciar a telemetria em um terminal:

```bash
sudo turbostat --quiet --no-perf --Summary --debug \
  --interval 0.1 \
  --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt \
  --out /tmp/resnet8_cpu_turbostat.log
```

Em outro terminal:

```bash
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \
python3 CPU/resnet8-softmax/benchmark_softmax_cpu.py --restart
```

Depois de encerrar o `turbostat` com `Ctrl+C`:

```bash
python3 CPU/resnet8-softmax/enriquecer_rapl.py \
  --log /tmp/resnet8_cpu_turbostat.log
```

O uso de `--restart` remove somente os checkpoints pertencentes a esta pasta e inicia uma nova sessão. Sem essa opção, o benchmark pode retomar passagens já persistidas.
