# Metodologia de benchmark — CPU, GPU, ZCU104, hls4ml e Vitis AI

**Data de consolidação:** 31/08/2026  
**Projeto de referência:** ResNet8 para CIFAR-10, com foco no modelo sem softmax  
**Objetivo:** documentar como os benchmarks deste repositório foram construídos, o que cada métrica realmente mede e como repetir a metodologia em outros modelos, datasets, placas e aplicações.

> Este arquivo é um manual de execução e adaptação. O inventário histórico, os resultados obtidos e o estado dos artefatos estão em [ESTADO-31-08.md](ESTADO-31-08.md).

---

## 1. Escopo e grau de validação

Há duas partes diferentes neste documento:

1. **Fluxo reproduzido e validado neste repositório**
   - TensorFlow/Keras em CPU;
   - TensorFlow/Keras em GPU NVIDIA;
   - conversão e otimização da ResNet8 com hls4ml;
   - execução do acelerador hls4ml na ZCU104 por AXI DMA;
   - cenários inference-only, end-to-end e “saturado serial”;
   - medição de potência e energia;
   - validação da retirada da softmax.

2. **Roteiro de adaptação para Vitis AI**
   - descreve como aplicar o mesmo contrato experimental a um DPU/VART ou a um Execution Provider compatível;
   - não representa um benchmark Vitis AI já executado neste repositório;
   - a pasta **Vitis AI/** está vazia na data desta consolidação;
   - nomes de APIs, ferramentas e compatibilidade devem ser confirmados para a versão do Vitis AI, XRT, imagem da placa e DPU utilizados.

O princípio central é simples:

> Só compare números quando os dois lados medirem a mesma fronteira operacional, com o mesmo batch, o mesmo conjunto de amostras e a mesma política de sincronização.

---

## 2. As perguntas que um benchmark deve responder

Um único número de FPS não descreve todo o sistema. Neste trabalho, as medições foram separadas conforme a pergunta:

| Cenário | Pergunta respondida | Começa em | Termina em |
|---|---|---|---|
| Validação funcional | O modelo/acelerador continua correto? | amostra do dataset | predição comparada ao rótulo/referência |
| Latência inference-only | Quanto custa uma inferência síncrona com dados prontos? | entrada já preparada para o backend | saída do backend disponível no host |
| Throughput efetivo batch 1 | Quantas imagens novas o caminho de inferência processa por segundo? | início do laço serial da passagem | fim do laço após todas as predições |
| End-to-end | Quanto custa transformar uma amostra crua em uma decisão? | amostra crua em RAM | classe/resultado final |
| Saturado serial | Qual o limite do caminho acelerado com trabalho mínimo do host? | entrada já carregada e reutilizável | conclusão síncrona de cada execução |
| Saturado concorrente | Qual o máximo de vazão com múltiplos trabalhos em voo? | janela de carga contínua | fim da janela, após drenar os trabalhos |

### 2.1 Cobertura realmente implementada

| Backend | Validação | Latência inference-only | Throughput efetivo batch 1 | End-to-end dedicado | Saturado serial | Saturado concorrente |
|---|---:|---:|---:|---:|---:|---:|
| TensorFlow CPU | sim | sim | sim | não | não | não |
| TensorFlow GPU | sim | sim | sim | não | não | não |
| ZCU104 + hls4ml | sim | sim | sim | sim | sim, no notebook intermediário | não |
| ZCU104 + Vitis AI | roteiro | roteiro | roteiro | roteiro | roteiro | roteiro |

Nos scripts de CPU e GPU, a “latência” é inference-only e o “FPS efetivo” mede o laço batch 1. Eles não publicam uma campanha end-to-end separada nem um modo saturado. Na ZCU104, inference-only e end-to-end estão no script robusto; o saturado serial está documentado e executado no notebook intermediário. Esta distinção impede atribuir aos arquivos capacidades que eles não implementam.

### 2.2 O que não deve ser misturado

- acurácia não é latência;
- tempo total de uma validação de 10.000 imagens não substitui um benchmark controlado;
- FPS global não é a média aritmética dos FPS de cada passagem;
- potência do pacote da CPU, potência da GPU e potência de um rail da placa têm escopos físicos diferentes;
- batch 1 serial não é equivalente a batching ou múltiplos runners;
- o “saturado” usado no ensaio hls4ml foi serial e síncrono; não significa utilização máxima possível de todos os recursos por concorrência;
- remover a softmax não muda o argmax, mas muda o tipo de saída: logits não são probabilidades.

---

## 3. Contrato experimental comum

Antes de escrever código específico para CPU, GPU ou FPGA, congele um contrato experimental.

### 3.1 Dataset

No experimento de referência:

- dataset: conjunto de teste CIFAR-10;
- total: 10.000 imagens;
- entrada original: uint8, formato NHWC, 32 × 32 × 3;
- normalização de software: conversão para float32 e divisão por 255;
- tamanhos testados: 100, 1.000 e 10.000 amostras;
- seed: 20260825;
- seleção estratificada por classe;
- subconjuntos aninhados: as primeiras 100 pertencem às 1.000, que pertencem às 10.000.

Subconjuntos aninhados reduzem uma fonte de variação: ao aumentar N, o conjunto anterior continua contido no próximo. A estratificação evita que um subconjunto pequeno fique desbalanceado.

Ao portar:

1. carregue uma vez a lista canônica de amostras;
2. gere os índices com uma seed fixa;
3. salve os índices ou um hash deles;
4. use exatamente os mesmos índices em todos os backends;
5. registre transformação, layout, dtype, escala, zero point e ordem dos canais.

### 3.2 Batch, concorrência e sincronização

O benchmark comparável deste projeto usa:

- batch 1;
- uma imagem por chamada;
- execução serial;
- uma chamada em voo;
- sincronização explícita antes de parar o cronômetro;
- 100 inferências de aquecimento;
- nenhuma remoção de outliers.

Qualquer mudança deve aparecer no nome do cenário e nos metadados:

- batch maior que 1;
- múltiplos streams;
- múltiplas threads/runners;
- fila assíncrona;
- double buffering;
- transferência sobreposta à computação.

### 3.3 Repetições

Cada N foi executado em até 100 passagens completas. Os resultados de 10, 20, 50 e 100 ciclos são **prefixos cumulativos da mesma campanha**:

- resultado de 10 ciclos: ciclos 1–10;
- resultado de 20 ciclos: ciclos 1–20;
- resultado de 50 ciclos: ciclos 1–50;
- resultado de 100 ciclos: ciclos 1–100.

Eles não são quatro campanhas independentes. Isso deve ser mantido ao interpretar intervalos de confiança ou comparar tabelas.

### 3.4 Aquecimento

O aquecimento ocorre fora da medição e serve para absorver:

- tracing e compilação de grafo;
- inicialização de kernels;
- alocação tardia;
- carregamento de caches;
- inicialização de DMA/runner;
- estabilização inicial de frequência.

Neste projeto foram usadas 100 inferências de warm-up. Em outro modelo, verifique se a latência estabiliza. Registre o número escolhido; não ajuste silenciosamente depois de ver os resultados.

### 3.5 Relógio

Para intervalos curtos no host, use um relógio monotônico de alta resolução:

~~~python
from time import perf_counter_ns

t0 = perf_counter_ns()
saida = executar_uma_inferencia()
sincronizar_e_materializar(saida)
t1 = perf_counter_ns()
latencia_ms = (t1 - t0) / 1e6
~~~

Nunca encerre a medição apenas quando uma API assíncrona retorna. O fim do intervalo deve representar a conclusão real do trabalho.

---

## 4. Validação funcional antes da velocidade

O benchmark só é válido depois de provar que o caminho medido executa o modelo pretendido.

### 4.1 Modelo com e sem softmax

O repositório contém:

- **resnet8_cifar10_keras3.h5**: saída com softmax;
- **resnet8_cifar10_keras3_no_softmax.h5**: saída linear, em logits;
- [scripts modelo/remove_softmax.py](scripts%20modelo/remove_softmax.py): remove a softmax preservando os pesos.

Os validadores:

- [CPU/resnet8-no-softmax/validar_equivalencia.py](CPU/resnet8-no-softmax/validar_equivalencia.py);
- [GPU/resnet8-no-softmax/validar_equivalencia.py](GPU/resnet8-no-softmax/validar_equivalencia.py).

Eles verificam:

1. estrutura e pesos dos modelos;
2. igualdade do argmax;
3. reconstrução da softmax a partir dos logits;
4. comportamento em todo o conjunto de teste.

Exemplo:

~~~bash
CUDA_VISIBLE_DEVICES="" python3 CPU/resnet8-no-softmax/validar_equivalencia.py \
  --softmax-model resnet8_cifar10_keras3.h5 \
  --logits-model resnet8_cifar10_keras3_no_softmax.h5 \
  --output CPU/resnet8-no-softmax/resultados/equivalencia_softmax_logits.json \
  --batch-size 1
~~~

Na GPU:

~~~bash
CUDA_VISIBLE_DEVICES=0 python3 GPU/resnet8-no-softmax/validar_equivalencia.py \
  --softmax-model resnet8_cifar10_keras3.h5 \
  --logits-model resnet8_cifar10_keras3_no_softmax.h5 \
  --output GPU/resnet8-no-softmax/resultados/equivalencia_softmax_logits.json \
  --batch-size 1
~~~

### 4.2 Validação de um backend quantizado

Para hls4ml ou Vitis AI, compare contra uma referência explícita:

- referência preferencial: logits float do modelo sem softmax;
- compare argmax em todo o conjunto;
- salve matriz de confusão;
- meça erro numérico por saída: máximo absoluto, médio absoluto e, quando útil, erro relativo;
- conte amostras cujo argmax divergiu;
- salve exemplos divergentes;
- confirme saturação e overflow;
- não use apenas algumas imagens “que funcionaram”.

No caso de classificação sem softmax:

~~~python
classe = int(logits.argmax())
~~~

Não aplique softmax se a tarefa só precisa da classe. Se probabilidades calibradas forem necessárias, aplique a softmax no pós-processamento e declare que esse custo pertence ao end-to-end.

### 4.3 Smoke test de hardware

Antes de uma campanha longa na placa:

1. carregue BIT e HWH;
2. localize e reinicialize o DMA;
3. execute uma amostra conhecida;
4. confira TLAST e estados de erro;
5. valide padding e empacotamento;
6. compare os logits decodificados;
7. só então inicie warm-up e ciclos.

---

## 5. Fronteiras de medição

### 5.1 Inference-only

Definição usada:

> A entrada já está transformada para o formato exigido pelo backend. O cronômetro cobre a submissão, a execução e a sincronização necessária para a saída estar disponível.

CPU/GPU TensorFlow:

- tensor de batch é criado antes ou imediatamente fora da região específica de inferência, conforme o script;
- a chamada do grafo está dentro;
- a materialização com .numpy() está dentro para forçar sincronização;
- argmax e preparação do próximo item ficam fora da latência unitária quando não forem parte da inferência.

ZCU104 hls4ml:

- a imagem já está quantizada, empacotada e copiada para o PynqBuffer;
- dentro do tempo: DMA MM2S, acelerador, DMA S2MM e waits;
- fora: normalização, quantização, packing, cópia da nova imagem, decode e argmax.

Vitis AI:

- a entrada já deve estar no dtype/layout/escala do tensor do runner;
- declare se sync_for_write está dentro ou fora;
- dentro: submissão ao runner, execução, wait e sincronização de leitura necessária;
- dequantização, decode e argmax ficam fora.

### 5.2 Throughput efetivo batch 1

Definição usada:

> Tempo de parede de uma passagem serial por N imagens novas, incluindo o trabalho necessário para alimentar o backend e obter a classe, mas sem incluir preparação global do dataset, carregamento do modelo ou checkpoint em disco.

No TensorFlow deste repositório, o tempo efetivo inclui:

- laço Python;
- criação/cópia do tensor de cada amostra;
- inferência;
- sincronização pela leitura no host;
- argmax;
- armazenamento da predição.

Na ZCU104 hls4ml, inclui:

- cópia de uma entrada pré-empacotada nova para o PynqBuffer;
- DMA → FPGA → DMA;
- espera síncrona;
- decode dos logits;
- argmax;
- laço serial.

Fórmula por passagem:

~~~text
FPS_passagem = N / tempo_de_parede_da_passagem
~~~

FPS global:

~~~text
FPS_global = soma_de_todas_as_imagens / soma_dos_tempos_de_todas_as_passagens
~~~

Não calcule FPS global como média simples de FPS quando os tempos ou N diferirem.

### 5.3 End-to-end

Definição usada na ZCU104:

> A amostra começa em uint8 na RAM e termina como classe predita.

Dentro da região:

- conversão para float;
- normalização por 255;
- quantização para o fixed-point de entrada;
- empacotamento;
- cópia para PynqBuffer;
- DMA → FPGA → DMA;
- decode;
- argmax.

Fora:

- leitura do arquivo do dataset;
- carga do overlay;
- alocação permanente dos buffers;
- warm-up;
- escrita de checkpoints.

Em outra aplicação, “end-to-end” precisa ser definido pelo produto:

- câmera: pode começar no frame capturado;
- áudio: pode começar no bloco PCM;
- detecção: termina depois de decode e NMS;
- segmentação: termina depois do mapa final redimensionado;
- NLP: pode incluir tokenização e decode.

Se I/O de disco ou rede for relevante ao produto, crie um cenário adicional. Não altere silenciosamente a fronteira.

### 5.4 Saturado serial

O ensaio salvo neste repositório reutiliza uma entrada já preparada e reduz o trabalho do host. Na ZCU104:

- reutiliza buffer de entrada;
- repete DMA → FPGA → DMA;
- espera cada execução antes da próxima;
- não inclui pré-processamento de uma nova imagem;
- não inclui decode nem argmax;
- permanece batch 1, serial e síncrono.

O nome tecnicamente mais preciso é:

> caminho acelerado com carga mínima do host, serial.

Ele é útil para estimar o limite do caminho de hardware/DMA, mas não prova throughput máximo com múltiplas requisições em voo.

Os ensaios salvos estão em:

- [README-SATURADO.md](hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/README-SATURADO.md);
- [benchmark_resnet8_hls_ip11.ipynb](hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/benchmark_resnet8_hls_ip11.ipynb).

A campanha utilizou:

- janelas de 10 segundos para throughput;
- 8 janelas;
- 50.000 execuções no estudo de latência saturada.

Pseudocódigo:

~~~python
for janela in range(numero_de_janelas):
    concluidas = 0
    t0 = perf_counter_ns()
    while tempo_decorrido(t0) < duracao_da_janela:
        dma_recv.transfer(saida)
        dma_send.transfer(entrada_ja_carregada)
        dma_send.wait()
        dma_recv.wait()
        concluidas += 1
    t1 = perf_counter_ns()
    salvar(concluidas / segundos(t1 - t0))
~~~

### 5.5 Saturado concorrente

Para medir vazão máxima de um runtime que suporta concorrência:

1. crie runners/streams suficientes;
2. mantenha uma fila limitada de trabalhos em voo;
3. faça warm-up com a mesma concorrência;
4. meça em janelas de duração fixa;
5. pare de submeter ao fim da janela;
6. drene os trabalhos já submetidos;
7. conte somente execuções concluídas segundo a regra documentada;
8. registre runners, threads, batch, profundidade da fila e afinidade.

Esse cenário deve ser publicado separadamente do batch 1 serial.

---

## 6. Estatística

Para cada matriz de latências com forma ciclos × N, calcule:

- média;
- mediana;
- desvio-padrão amostral, com ddof = 1;
- percentil 95;
- mínimo;
- máximo;
- número total de inferências.

Para as médias por passagem:

- média das médias;
- intervalo de confiança de 95%;
- FPS por passagem;
- FPS global ponderado pelo tempo.

Para acurácia:

- acertos / total;
- intervalo de confiança de Wilson de 95%;
- matriz de confusão;
- divergência contra o backend de referência.

Exemplo conceitual:

~~~python
import numpy as np

flat = latencias.reshape(-1)
media = np.mean(flat)
mediana = np.median(flat)
desvio = np.std(flat, ddof=1)
p95 = np.percentile(flat, 95)
~~~

### 6.1 Prefixos de ciclos

Se a matriz contém 100 ciclos, o resumo de 20 ciclos usa:

~~~python
prefixo = latencias[:20]
~~~

Não execute novamente as 20 passagens e chame o resultado de prefixo da campanha de 100.

### 6.2 Outliers

Neste trabalho nenhum outlier foi removido. Se outro projeto optar por remoção:

- preserve os dados brutos;
- defina o critério antes da campanha;
- publique resultado com e sem filtro;
- conte quantos pontos foram removidos;
- não filtre apenas uma plataforma.

---

## 7. Potência e energia

### 7.1 Regra geral

Potência precisa de:

- rail/domínio medido;
- unidade;
- intervalo de amostragem;
- timestamps;
- baseline idle;
- fronteira temporal ativa;
- política de integração;
- dados crus preservados.

Fórmulas:

~~~text
potência_dinâmica = max(potência_ativa - potência_idle, 0)

energia_por_imagem = potência_média_ativa / FPS_global

energia_dinâmica_por_imagem = potência_dinâmica / FPS_global
~~~

Equivalente por passagem:

~~~text
energia = integral_de_potência_no_intervalo
energia_por_imagem = energia / N
~~~

### 7.2 CPU

Foi usado turbostat a cada 100 ms, com:

- Busy%;
- Bzy_MHz;
- PkgTmp;
- PkgWatt;
- CorWatt;
- timestamp.

Fluxo:

1. iniciar turbostat externamente;
2. manter o processo durante baseline e benchmark;
3. executar o benchmark CPU;
4. enriquecer os CSVs com integração ponderada pela sobreposição temporal;
5. preservar o log bruto.

Exemplo:

~~~bash
sudo turbostat \
  --quiet \
  --no-perf \
  --Summary \
  --debug \
  --interval 0.1 \
  --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt \
  --out /tmp/resnet8_cpu_turbostat.log
~~~

Em outro terminal:

~~~bash
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \
python3 CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py --restart
~~~

Depois:

~~~bash
python3 CPU/resnet8-no-softmax/enriquecer_rapl.py \
  --log /tmp/resnet8_cpu_turbostat.log \
  --results-dir CPU/resnet8-no-softmax/resultados \
  --interval-seconds 0.1
~~~

O enriquecedor associa amostras de potência aos timestamps das passagens. Uma amostra parcial é ponderada pela fração de sobreposição, em vez de ser incluída integralmente.

### 7.3 GPU NVIDIA

O benchmark inicia um sampler persistente de nvidia-smi. Foram coletados, aproximadamente a cada 100 ms:

- power.draw.instant ou power.draw.average, conforme disponibilidade;
- utilização;
- clock;
- temperatura;
- timestamp.

O benchmark mede um baseline idle antes das passagens e integra as amostras no intervalo ativo.

Cuidados:

- confirme qual GPU foi selecionada;
- salve driver, CUDA, cuDNN e TensorFlow;
- desative outros processos que usem a GPU;
- declare limites de potência e política de clocks;
- confirme que a saída realmente foi materializada antes de parar o relógio.

### 7.4 ZCU104

Na placa foi usado o rail PMBus **12V_power**, escolhido depois de inspecionar os sensores disponíveis. O estudo de overhead levou ao uso preferencial de amostragem de 1 segundo na campanha de potência.

Procedimento:

1. listar rails;
2. escolher e registrar o rail;
3. coletar idle por pelo menos 5 segundos;
4. coletar amostras durante cada passagem ou janela;
5. salvar os dados crus;
6. calcular potência ativa e dinâmica;
7. relatar que a potência representa a placa/rail, não apenas o IP.

Intervalos muito curtos de telemetria podem perturbar o benchmark, principalmente em Python no ARM. Faça um estudo de overhead sem telemetria e com diferentes intervalos antes da campanha definitiva.

### 7.5 Comparação honesta

Não conclua que uma plataforma é mais eficiente sem declarar o escopo:

- PkgWatt representa pacote da CPU;
- nvidia-smi representa a placa/domínio reportado pelo driver;
- 12V_power representa um rail da ZCU104;
- nenhuma dessas medidas é automaticamente “potência exclusiva do modelo”.

Use energia por imagem dentro de cada escopo e exponha a diferença física.

---

## 8. Benchmark em CPU

### 8.1 Script principal

[CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py](CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py)

Responsabilidades:

- força execução em CPU;
- carrega modelo e CIFAR-10;
- cria subconjuntos estratificados e aninhados;
- compila uma tf.function com entrada fixa de batch 1;
- executa warm-up;
- mede latência unitária;
- mede tempo efetivo da passagem;
- calcula acurácia;
- cria checkpoints;
- salva metadados e resumos.

Execução padrão:

~~~bash
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \
python3 CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py \
  --model resnet8_cifar10_keras3_no_softmax.h5 \
  --output-dir CPU/resnet8-no-softmax/resultados \
  --sample-sizes 100 1000 10000 \
  --repetitions 10 20 50 100 \
  --seed 20260825 \
  --warmup 100 \
  --baseline-seconds 5 \
  --telemetry-ms 100 \
  --intra-op-threads 0 \
  --inter-op-threads 0 \
  --restart
~~~

Use --restart somente quando quiser substituir checkpoints dos Ns solicitados. Sem ele, o script pode continuar uma campanha interrompida.

### 8.2 Fronteira de latência no TensorFlow

O padrão usado é:

~~~python
@tf.function(
    input_signature=[tf.TensorSpec((1, 32, 32, 3), tf.float32)],
    jit_compile=False,
)
def infer(batch):
    return model(batch, training=False)

t0 = perf_counter_ns()
output = infer(batch)
host_output = output.numpy()
t1 = perf_counter_ns()
~~~

A chamada .numpy() é essencial na GPU e mantém uma fronteira equivalente na CPU: ela força a saída a estar disponível no host.

### 8.3 Threads

As opções --intra-op-threads e --inter-op-threads devem ser registradas. O valor 0 deixa o TensorFlow escolher. Para um estudo de escalabilidade:

- crie campanhas separadas por número de threads;
- fixe afinidade e NUMA quando relevante;
- registre governor e turbo;
- não misture configurações na mesma tabela.

### 8.4 Versão com softmax

[CPU/resnet8-softmax/benchmark_softmax_cpu.py](CPU/resnet8-softmax/benchmark_softmax_cpu.py) repete o contrato com o modelo original.  
[CPU/resnet8-softmax/enriquecer_rapl.py](CPU/resnet8-softmax/enriquecer_rapl.py) aplica o mesmo tratamento de potência.

Esses resultados permitem medir o custo observado da softmax, desde que ambiente, amostras e parâmetros sejam iguais.

---

## 9. Benchmark em GPU

### 9.1 Script principal

[GPU/resnet8-no-softmax/benchmark_no_softmax_gpu.py](GPU/resnet8-no-softmax/benchmark_no_softmax_gpu.py)

Responsabilidades adicionais:

- exige uma GPU visível;
- seleciona o dispositivo explicitamente;
- verifica o placement da saída;
- inicia e encerra o sampler nvidia-smi;
- força sincronização pela materialização da saída.

Execução:

~~~bash
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=2 \
python3 GPU/resnet8-no-softmax/benchmark_no_softmax_gpu.py \
  --model resnet8_cifar10_keras3_no_softmax.h5 \
  --output-dir GPU/resnet8-no-softmax/resultados \
  --sample-sizes 100 1000 10000 \
  --repetitions 10 20 50 100 \
  --seed 20260825 \
  --warmup 100 \
  --baseline-seconds 5 \
  --telemetry-ms 100 \
  --restart
~~~

### 9.2 Por que sincronizar

Kernels de GPU são normalmente enfileirados de forma assíncrona. Medir apenas:

~~~python
saida = infer(batch)
~~~

pode capturar submissão, não conclusão. Neste projeto, output.numpy() fica antes de t1.

### 9.3 Cuidados de portabilidade

Em outro framework:

- PyTorch: sincronize CUDA antes de t0 e antes de t1, ou use CUDA Events;
- TensorRT: aguarde o stream ou use eventos;
- OpenCL: finalize/aguarde o evento;
- Vitis AI: aguarde o job e sincronize o buffer conforme a API.

Não acrescente sincronizações diferentes entre plataformas sem declarar a mudança.

### 9.4 Versão com softmax

[GPU/resnet8-softmax/benchmark_softmax_gpu.py](GPU/resnet8-softmax/benchmark_softmax_gpu.py) usa o mesmo desenho experimental para o modelo com softmax.

---

## 10. Arquivos produzidos por CPU e GPU

Cada pasta de resultados segue este contrato:

| Arquivo | Conteúdo |
|---|---|
| latencias_n100.npy | matriz ciclos × 100, em milissegundos |
| latencias_n1000.npy | matriz ciclos × 1.000 |
| latencias_n10000.npy | matriz ciclos × 10.000 |
| passagens_nN.csv | uma linha por ciclo, timestamps, tempo total, FPS e acurácia |
| resultados.csv | resumos para prefixos de 10/20/50/100 ciclos |
| metadados.json | ambiente, modelo, seed, parâmetros e fronteiras |
| RESULTADOS.md | relatório humano derivado |
| telemetria bruta | amostras com timestamps |

Boas práticas:

- o NPY é fonte primária para latência;
- o CSV de passagens é fonte primária para throughput;
- RESULTADOS.md deve ser regenerável;
- salve hashes dos artefatos;
- confira se o número de linhas do CSV é igual ao número de linhas do NPY;
- grave checkpoint ao fim de cada ciclo;
- escreva primeiro em arquivo temporário e depois faça substituição atômica quando possível.

---

## 11. Construção do acelerador hls4ml

O benchmark na placa depende de um IP correto. A metodologia de geração também precisa ser reproduzível.

### 11.1 Pipeline do repositório

Os scripts estão em [hls4ml/scripts](hls4ml/scripts):

| Script | Função |
|---|---|
| 00_check_environment.py | verifica Python, TensorFlow, hls4ml e ferramentas AMD/Xilinx |
| 01_inspect_models.py | inspeciona modelos com/sem softmax e gera inventário |
| 02_prepare_cifar10.py | prepara CIFAR-10 e artefatos de entrada |
| 03_build_baseline.py | gera, compila, valida e opcionalmente sintetiza o baseline |
| 04_optimize_fifo.py | faz profiling RTL e otimiza profundidades FIFO |
| 05_make_correct_ip.py | reconstrói o pacote IP com RTL sintetizado correto |
| hls_common.py | parâmetros e funções compartilhadas |

Ordem:

~~~bash
cd hls4ml

python3 scripts/00_check_environment.py
python3 scripts/01_inspect_models.py
python3 scripts/02_prepare_cifar10.py
python3 scripts/03_build_baseline.py --validate-samples 100
python3 scripts/04_optimize_fifo.py --tb-samples 2
~~~

Para sintetizar/exportar:

~~~bash
python3 scripts/03_build_baseline.py --validate-samples 100 --synth
python3 scripts/04_optimize_fifo.py --tb-samples 2 --vsynth
~~~

Não execute --force em um build que contenha evidência que ainda não foi copiada. Builds HLS são demorados e parte dos relatórios é necessária para auditoria.

### 11.2 Configuração usada

- backend: Vitis;
- IOType: io_stream;
- Strategy: Resource;
- convolução: LineBuffer;
- precisão principal: ap_fixed<22,12,AP_RND_CONV,AP_SAT>;
- período alvo: 10 ns, equivalente a 100 MHz;
- reuse factor máximo planejado: 288;
- modelo sem softmax.

Esses valores não são universais. Em outro projeto:

1. obtenha ranges de pesos e ativações;
2. escolha precisão inicial conservadora;
3. valide C Simulation;
4. reduza precisão progressivamente;
5. confira saturações e divergências;
6. só depois avalie recursos e timing.

### 11.3 Otimização FIFO

O fluxo utilizado:

1. gerar um projeto io_stream;
2. colocar FIFOs em uma profundidade de profiling alta;
3. executar testbench RTL com pelo menos duas chamadas ao top;
4. registrar ocupação máxima;
5. gerar profundidades otimizadas com margem;
6. falhar se algum FIFO alcançar a profundidade de profiling;
7. sintetizar novamente;
8. validar C/RTL e conferir o Tcl final.

Neste projeto foram 42 FIFOs, profundidade de profiling 4096 e máximo otimizado observado de 2100, sem atingir o teto.

Uma única amostra não é suficiente para o flow de profiling usado. O script exige pelo menos duas chamadas.

### 11.4 Integridade do IP

Foi detectado que a exportação podia reintroduzir profundidades 4096 no IP, apesar de o projeto otimizado conter os valores corretos. Por isso:

- não confie apenas no diretório intermediário;
- inspecione o RTL e o component.xml do IP exportado;
- confira os parâmetros do Tcl final;
- gere manifest e hashes;
- valide o IP em projeto limpo;
- preserve relatórios de síntese e implementação.

O script [hls4ml/scripts/05_make_correct_ip.py](hls4ml/scripts/05_make_correct_ip.py) recebe:

~~~bash
python3 hls4ml/scripts/05_make_correct_ip.py \
  --scaffold CAMINHO_DO_IP_SCAFFOLD \
  --syn-rtl CAMINHO_DO_RTL_SINTETIZADO \
  --out CAMINHO_DO_IP_CORRETO \
  --version 1.1 \
  --archive CAMINHO_DO_ZIP
~~~

No estado atual, o hardware validado é o IP 1.1. Não misture guias, bitstreams ou manifests de versões antigas.

---

## 12. Integração hls4ml na ZCU104

### 12.1 Caminho físico

~~~text
RAM DDR
  → PynqBuffer de entrada
  → AXI DMA MM2S, 128 bits
  → wrapper de largura
  → entrada do acelerador, 96 bits
  → ResNet8 hls4ml
  → saída do acelerador, 320 bits
  → wrapper de largura
  → AXI DMA S2MM, 512 bits
  → PynqBuffer de saída
  → decode dos 10 logits
~~~

O wrapper está em [hardware/src/Accel_dma_wrapper.vhd](hardware/src/Accel_dma_wrapper.vhd).

Características:

- ap_start mantido ativo;
- adaptação de largura na entrada e na saída;
- propagação de TLAST;
- polling do DMA;
- sem interrupções no benchmark;
- uma transação por vez.

### 12.2 Buffers

Entrada:

- forma física: (1024, 4);
- dtype: uint32;
- tamanho: 16 KiB;
- contém os pixels empacotados e padding.

Saída:

- forma física: (16,);
- dtype: uint32;
- tamanho: 64 bytes;
- contém 10 logits e padding.

O tamanho físico do DMA não é igual ao número lógico de elementos. Ao adaptar, derive o tamanho a partir:

- largura AXI;
- largura do tipo;
- número de elementos;
- alinhamento;
- packing escolhido pelo wrapper.

### 12.3 Quantização

Para ap_fixed<22,12>, há 10 bits fracionários:

~~~text
escala = 2^10 = 1024
~~~

Codificação conceitual:

~~~python
q = round(valor_float * 1024)
q = clip(q, minimo_representavel, maximo_representavel)
~~~

Decodificação:

~~~python
valor_float = inteiro_com_sinal / 1024.0
~~~

Não suponha que outro projeto terá a mesma escala. Derive-a do tipo fixo e valide valores negativos, arredondamento, saturação e extensão de sinal.

### 12.4 Ordem segura de transferência

O caminho validado arma primeiro a recepção:

~~~python
dma.recvchannel.transfer(out_buffer)
dma.sendchannel.transfer(in_buffer)
dma.sendchannel.wait()
dma.recvchannel.wait()
~~~

Depois:

1. confira status;
2. invalide/sincronize cache se a plataforma exigir;
3. decodifique apenas as palavras válidas;
4. aplique argmax.

---

## 13. Script robusto da ZCU104

O script principal é:

[benchmark_resnet8_zcu104_ip11_robusto.py](hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/benchmark_resnet8_zcu104_ip11_robusto.py)

Ele:

- valida BIT, HWH e dataset;
- carrega o overlay;
- encontra o DMA;
- lista/seleciona rail de potência;
- executa smoke test;
- verifica padding e erros do DMA;
- gera os mesmos subconjuntos estratificados;
- aloca buffers uma vez;
- mede acurácia separadamente;
- pré-empacota entradas para inference-only;
- executa inference_only, end_to_end ou ambos;
- salva NPY, CSV, telemetria e resumos;
- retoma checkpoints.

### 13.1 Preparar a pasta na placa

Copie para a mesma pasta:

- resnet8_hls_ip11.bit;
- resnet8_hls_ip11.hwh;
- cifar10_test_uint8.npz;
- benchmark_resnet8_zcu104_ip11_robusto.py.

Exemplo de destino:

~~~text
/home/xilinx/jupyter_notebooks/resnet8_hls_ip11/
~~~

### 13.2 Conferir rails

~~~bash
python3 benchmark_resnet8_zcu104_ip11_robusto.py \
  --base /home/xilinx/jupyter_notebooks/resnet8_hls_ip11 \
  --list-rails
~~~

### 13.3 Executar inference-only e end-to-end

~~~bash
python3 benchmark_resnet8_zcu104_ip11_robusto.py \
  --base /home/xilinx/jupyter_notebooks/resnet8_hls_ip11 \
  --bit resnet8_hls_ip11.bit \
  --dataset cifar10_test_uint8.npz \
  --cycles 100 \
  --warmup 100 \
  --idle-seconds 5 \
  --telemetry-interval 1 \
  --seed 20260825 \
  --power-rail 12V_power \
  --scenario both \
  --restart
~~~

Opções de --scenario:

- inference_only;
- end_to_end;
- both.

O valor padrão de --telemetry-interval no script é 0,1 s. A campanha definitiva de potência preferiu 1 s para reduzir interferência. Faça a mesma avaliação no ambiente novo.

### 13.4 Campanha final

O notebook final está em:

[benchmark_resnet8_hls_ip11.ipynb](hls4ml/Resultados_ZCU104/resnet8_ip11_final_10k100_2026-08-28/benchmark_resnet8_hls_ip11.ipynb)

Ele executa a campanha principal de 10.000 imagens × 100 ciclos e preserva predições, latências, CSVs, JSONs e hashes. O notebook não substitui o script robusto como especificação metodológica; ambos devem concordar nas fronteiras.

---

## 14. Como portar o benchmark hls4ml

Crie um adaptador com cinco operações:

~~~python
class Backend:
    def preparar(self, amostra_crua):
        # preprocessamento, quantização e packing
        ...

    def carregar_entrada(self, entrada_preparada):
        # cópia/sync para buffer do acelerador
        ...

    def executar_sincrono(self):
        # submissão, execução e wait
        ...

    def ler_saida(self):
        # sync, unpacking e dequantização
        ...

    def posprocessar(self, saida):
        # argmax, NMS, decode etc.
        ...
~~~

Então componha:

- inference-only = executar_sincrono;
- efetivo batch 1 = carregar + executar + ler + pós-processar;
- end-to-end = preparar + carregar + executar + ler + pós-processar;
- saturado serial = repetir executar_sincrono com entrada já carregada;
- validação = caminho completo mais comparação à referência.

Itens que precisam mudar:

- shape e layout;
- packing;
- tipos fixed-point;
- largura AXI;
- tamanho dos buffers;
- número e significado das saídas;
- função de pós-processamento;
- clock;
- IP/overlay;
- rail de potência.

Itens que devem permanecer:

- seed e índices;
- política de warm-up;
- número de ciclos;
- relógio;
- regra de sincronização;
- arquivos produzidos;
- fórmulas estatísticas;
- documentação das fronteiras.

---

## 15. Adaptação para Vitis AI

Esta seção traduz o contrato acima para um DPU. Ela é um roteiro, não evidência de execução local.

### 15.1 Fixar a pilha compatível

Antes do benchmark, registre:

- versão do Vitis AI;
- versão do XRT;
- versão da imagem/PetaLinux da placa;
- arquitetura e fingerprint do DPU;
- versão do compilador/quantizador;
- modelo compilado;
- runtime usado: VART, ONNX Runtime Execution Provider ou outro;
- operadores executados no DPU e operadores que caíram para CPU.

Não escolha uma versão apenas por ser a mais nova. Use a matriz compatível com a imagem da placa e com o DPU implementado.

### 15.2 Preparar e validar o modelo

Fluxo genérico:

1. exportar o modelo float no formato aceito;
2. quantizar com conjunto de calibração representativo;
3. compilar para a arquitetura do DPU da placa;
4. inspecionar os subgrafos;
5. identificar fallback em CPU;
6. executar validação funcional completa;
7. salvar hash do modelo compilado.

Confirme no tensor de entrada:

- shape;
- batch;
- NHWC/NCHW;
- int8/uint8/float;
- escala e zero point;
- alinhamento;
- necessidade de sincronização explícita.

Confirme o mesmo para a saída.

### 15.3 Adaptador VART conceitual

~~~python
class VitisAIBackend:
    def preparar(self, amostra):
        x = preprocessar(amostra)
        return quantizar_e_reorganizar(x, input_tensor)

    def carregar_entrada(self, preparada):
        copiar_para_tensor_buffer(preparada)
        sincronizar_para_escrita()

    def executar_sincrono(self):
        job_id = runner.execute_async(inputs, outputs)
        runner.wait(job_id)

    def ler_saida(self):
        sincronizar_para_leitura()
        return dequantizar_saida(outputs)

    def posprocessar(self, saida):
        return decode(saida)
~~~

Os nomes exatos variam por versão e linguagem. O importante é manter a fronteira.

### 15.4 Cenários no Vitis AI

**Inference-only**

- entrada já quantizada e copiada;
- declare se a sincronização para escrita ocorreu antes de t0;
- t0;
- execute_async;
- wait;
- sincronização de leitura necessária;
- t1;
- dequantização e pós-processamento fora.

**Efetivo batch 1**

- entrada pré-processada;
- cópia/sync;
- runner;
- leitura/sync;
- dequantização;
- argmax/decode;
- repetir serialmente para N imagens.

**End-to-end**

- começa na amostra crua em RAM;
- inclui resize/crop/normalização;
- inclui quantização/layout;
- inclui runner;
- inclui pós-processamento final.

**Saturado serial**

- reutiliza entrada pronta;
- uma chamada assíncrona seguida imediatamente de wait;
- permite comparação conceitual com o saturado serial hls4ml.

**Saturado concorrente**

- múltiplos runners, threads ou jobs em voo;
- janelas fixas;
- concorrência registrada;
- resultado publicado separadamente.

### 15.5 Particularidades do DPU

- Em muitos deployments Zynq, o batch efetivo por core é 1; confirme no tensor/compilação.
- Para throughput, múltiplos cores ou runners podem ser necessários.
- Um grafo pode conter subgrafos DPU e CPU; cronometre o grafo completo ou apenas o subgrafo, mas diga qual.
- Pré e pós-processamento podem dominar modelos pequenos.
- Use o profiler do Vitis AI para localizar gargalos, mas não substitua o benchmark de aplicação pelo perfil interno.

### 15.6 Potência

Se Vitis AI for executado na mesma ZCU104:

- use o mesmo rail e intervalo do hls4ml quando possível;
- use a mesma definição de idle;
- meça os mesmos Ns e ciclos;
- não deixe o sampler mudar entre as duas campanhas;
- registre número de cores DPU e frequência.

Assim a comparação energética fica mais defensável, embora os aceleradores e caminhos de software continuem diferentes.

---

## 16. Portabilidade para outras aplicações

### 16.1 Classificação

- saída: logits;
- pós-processamento: argmax;
- métrica funcional: acurácia e matriz de confusão;
- softmax opcional e normalmente fora de inference-only.

### 16.2 Detecção de objetos

Separe:

- inferência do backbone/head;
- decode de caixas;
- threshold;
- NMS;
- transformação de coordenadas.

Inference-only pode terminar nos tensores crus. End-to-end termina nas caixas finais. Registre mAP, não apenas acurácia.

### 16.3 Segmentação

Defina se argmax por pixel e resize final estão dentro. Use mIoU/Dice e preserve mapas de referência.

### 16.4 Áudio

End-to-end pode incluir janela, FFT, mel-spectrogram e normalização. Inference-only começa com o tensor de features pronto.

### 16.5 Séries temporais e dados tabulares

Declare construção de janelas, imputação e normalização. Evite incluir leitura de arquivo em uma plataforma e excluí-la em outra.

### 16.6 Modelos com múltiplas entradas/saídas

- sincronize todos os buffers;
- registre a ordem;
- salve shape e dtype de cada tensor;
- só pare o cronômetro quando todas as saídas necessárias estiverem disponíveis.

---

## 17. Esqueleto reutilizável de campanha

~~~python
def campanha(backend, dataset, indices, ciclos, warmup):
    validar_backend(backend, dataset, indices_validacao)

    for _ in range(warmup):
        preparada = backend.preparar(dataset[indices[0]])
        backend.carregar_entrada(preparada)
        backend.executar_sincrono()
        backend.ler_saida()

    for n in [100, 1000, 10000]:
        subset = indices[:n]
        latencias = carregar_checkpoint_ou_criar(ciclos, n)

        for ciclo in ciclos_pendentes(latencias):
            inicio_passagem = perf_counter_ns()

            for j, indice in enumerate(subset):
                preparada = backend.preparar(dataset[indice])
                backend.carregar_entrada(preparada)

                t0 = perf_counter_ns()
                backend.executar_sincrono()
                t1 = perf_counter_ns()

                saida = backend.ler_saida()
                predicao = backend.posprocessar(saida)
                latencias[ciclo, j] = (t1 - t0) / 1e6

            fim_passagem = perf_counter_ns()
            salvar_passagem_e_checkpoint()

        gerar_resumos_por_prefixo()
~~~

Esse esqueleto mostra a arquitetura, não uma fronteira universal. Se ler a saída for necessário para sincronizar, essa leitura deve ficar antes de t1.

---

## 18. Metadados mínimos

Cada execução deve registrar:

### Modelo

- caminho;
- SHA-256;
- formato;
- shape de entrada/saída;
- parâmetros;
- presença/ausência de softmax;
- precisão/quantização.

### Dataset

- nome e versão;
- split;
- SHA-256 do artefato local;
- número de amostras;
- seed;
- índices ou hash dos índices;
- preprocessamento.

### Host

- sistema operacional;
- kernel;
- Python;
- framework;
- bibliotecas numéricas;
- CPU/GPU;
- driver;
- threads;
- clocks/governor.

### FPGA/DPU

- placa;
- BIT/HWH/XSA ou xmodel;
- hashes;
- frequência;
- versão do IP;
- configuração/fingerprint do DPU;
- XRT/runtime;
- tamanho/layout dos buffers;
- método de sincronização.

### Benchmark

- cenário;
- descrição textual de t0 e t1;
- batch;
- concorrência;
- warm-up;
- Ns;
- ciclos;
- intervalo de telemetria;
- rail;
- baseline;
- data/hora e timezone.

---

## 19. Estrutura recomendada de resultados

~~~text
resultados/
├── metadados.json
├── validacao.json
├── matriz_confusao.npy
├── predicoes.npy
├── sha256sums.txt
├── inference_only/
│   ├── latencias_n100.npy
│   ├── latencias_n1000.npy
│   ├── latencias_n10000.npy
│   ├── passagens_n100.csv
│   ├── passagens_n1000.csv
│   ├── passagens_n10000.csv
│   ├── power_idle_raw.csv
│   ├── power_samples_nN.csv
│   ├── resultados.csv
│   └── RESULTADOS.md
├── end_to_end/
│   └── ...
└── saturado/
    ├── janelas.csv
    ├── latencias.npy
    ├── telemetria.csv
    └── RESULTADOS.md
~~~

Para aceleradores, copie também o artefato executável ou registre um caminho imutável e hash.

---

## 20. Checklist antes de publicar números

### Correção

- [ ] o modelo correto foi carregado;
- [ ] o hash foi salvo;
- [ ] softmax/logits estão identificados;
- [ ] argmax foi validado;
- [ ] acurácia completa foi executada;
- [ ] packing, padding, sinal e escala foram testados;
- [ ] DMA/runner terminou sem erro;
- [ ] fallback para CPU foi identificado.

### Comparabilidade

- [ ] mesmos índices;
- [ ] mesmo batch;
- [ ] mesma política de warm-up;
- [ ] mesma contagem de ciclos;
- [ ] fronteiras t0/t1 escritas;
- [ ] sincronização garantida;
- [ ] inference-only separado de end-to-end;
- [ ] serial separado de concorrente.

### Estatística

- [ ] latências brutas preservadas;
- [ ] sem remoção silenciosa de outliers;
- [ ] p95 calculado nos dados corretos;
- [ ] desvio-padrão com convenção declarada;
- [ ] FPS global calculado por imagens/tempo;
- [ ] prefixos cumulativos identificados;
- [ ] IC de acurácia calculado.

### Potência

- [ ] domínio/rail descrito;
- [ ] baseline idle coletado;
- [ ] timestamps preservados;
- [ ] intervalo de telemetria registrado;
- [ ] overhead avaliado;
- [ ] energia por imagem usa FPS da mesma fronteira;
- [ ] escopos físicos não foram tratados como equivalentes.

### Reprodutibilidade

- [ ] comando salvo;
- [ ] ambiente salvo;
- [ ] checkpoints íntegros;
- [ ] NPY e CSV têm ciclos coerentes;
- [ ] hashes gerados;
- [ ] relatório derivável dos dados crus;
- [ ] artefatos antigos não foram misturados.

---

## 21. Erros metodológicos comuns

1. **Parar o relógio antes da sincronização da GPU/runner.**  
   Resultado: latência artificialmente baixa.

2. **Chamar de inference-only um intervalo que inclui resize e argmax em apenas uma plataforma.**  
   Resultado: comparação desigual.

3. **Usar a média dos FPS como FPS global.**  
   Resultado: ponderação incorreta.

4. **Medir potência sem timestamp ou baseline.**  
   Resultado: energia não auditável.

5. **Comparar PkgWatt com rail da placa como se fossem o mesmo domínio.**  
   Resultado: conclusão energética exagerada.

6. **Usar conjunto diferente em cada backend.**  
   Resultado: acurácia e custo podem mudar com as amostras.

7. **Medir um acelerador quantizado sem validar seu argmax.**  
   Resultado: benchmark rápido de uma implementação errada.

8. **Chamar uma execução serial de throughput máximo.**  
   Resultado: confusão entre latência mínima e capacidade sob carga.

9. **Incluir carga do modelo somente em um backend.**  
   Resultado: end-to-end mal definido.

10. **Reutilizar nome de IP e trocar seu conteúdo.**  
    Resultado: bitstream, relatório e código deixam de ser rastreáveis.

---

## 22. Índice dos scripts relacionados

### CPU

- [benchmark_no_softmax_cpu.py](CPU/resnet8-no-softmax/benchmark_no_softmax_cpu.py)
- [validar_equivalencia.py](CPU/resnet8-no-softmax/validar_equivalencia.py)
- [enriquecer_rapl.py](CPU/resnet8-no-softmax/enriquecer_rapl.py)
- [benchmark_softmax_cpu.py](CPU/resnet8-softmax/benchmark_softmax_cpu.py)
- [enriquecer_rapl.py — softmax](CPU/resnet8-softmax/enriquecer_rapl.py)

### GPU

- [benchmark_no_softmax_gpu.py](GPU/resnet8-no-softmax/benchmark_no_softmax_gpu.py)
- [validar_equivalencia.py](GPU/resnet8-no-softmax/validar_equivalencia.py)
- [benchmark_softmax_gpu.py](GPU/resnet8-softmax/benchmark_softmax_gpu.py)

### hls4ml e hardware

- [scripts hls4ml](hls4ml/scripts)
- [wrapper AXI DMA](hardware/src/Accel_dma_wrapper.vhd)
- [script robusto ZCU104](hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/benchmark_resnet8_zcu104_ip11_robusto.py)
- [notebook intermediário](hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/benchmark_resnet8_hls_ip11.ipynb)
- [metodologia saturada](hls4ml/Resultados_ZCU104/resnet8_ip11_intermediario_2026-08-28/README-SATURADO.md)
- [notebook final 10k × 100](hls4ml/Resultados_ZCU104/resnet8_ip11_final_10k100_2026-08-28/benchmark_resnet8_hls_ip11.ipynb)

---

## 23. Referências oficiais

As referências abaixo servem para confirmar APIs e compatibilidade na versão efetivamente instalada:

- [hls4ml — documentação principal](https://fastmachinelearning.org/hls4ml/)
- [hls4ml — instalação e compatibilidade](https://fastmachinelearning.org/hls4ml/intro/setup.html)
- [hls4ml — backend Vitis](https://fastmachinelearning.org/hls4ml/backend/vitis.html)
- [hls4ml — profiling](https://fastmachinelearning.org/hls4ml/api/profiling.html)
- [Vitis AI — releases oficiais](https://github.com/Xilinx/Vitis-AI/releases)
- [VART Runner API](https://xilinx.github.io/Vitis-AI/3.5/html/doxygen/api/file/runner_8hpp.html)
- [Vitis AI Profiler — exemplos](https://github.com/Xilinx/Vitis-AI/blob/master/examples/vai_profiler/examples.md)

---

## 24. Regra final de interpretação

Para cada número publicado, deve ser possível responder sem olhar o código:

1. qual modelo e qual hash;
2. quais amostras;
3. qual dtype/layout/quantização;
4. onde o cronômetro começou;
5. onde terminou;
6. como a execução foi sincronizada;
7. qual batch e concorrência;
8. quantos warm-ups e ciclos;
9. qual domínio de potência;
10. quais dados brutos sustentam o resumo.

Se uma dessas respostas faltar, o número pode servir como observação exploratória, mas ainda não como comparação reproduzível.
