# LeNet/MNIST — Vitis AI Quantizer e AI Compiler para ZCU104

## Resultado

O fluxo foi executado com sucesso em 03/09/2026. A LeNet original foi convertida de Keras 3 para Keras 2.12 sem alterar nenhum tensor de pesos, quantizada por PTQ para INT8 com 10.000 imagens balanceadas e compilada para o alvo `DPUCZDX8G_ISA1_B4096` da ZCU104.

Artefatos principais:

- XModel compilado: `artifacts/compiled/lenet_mnist_zcu104_vai3_5/lenet_mnist_no_softmax.xmodel`
- pacote pronto para copiar à placa: `artifacts/deploy/lenet_mnist_zcu104_vai3_5/`
- modelo INT8 intermediário: `artifacts/quantized/lenet_mnist_no_softmax_quantized.h5`
- SHA-256 do XModel: `bfef577b297939e19cabf74b1ab8d30ab65d694cd6a1ec9b28197577a3ce52c5`
- tamanho do XModel: 171.444 bytes

## Arquitetura confirmada

O modelo recebe imagens MNIST normalizadas em formato NHWC. A última camada é `linear`: **não existe Softmax** no H5 nem no XModel. A decisão é obtida por `argmax` diretamente nos 10 logits. Isso é adequado para inferência, pois Softmax não altera a classe escolhida pelo argmax.

| Etapa | Saída | Ativação | Parâmetros |
|---|---:|---|---:|
| Entrada | 28 × 28 × 1 | — | 0 |
| Conv2D, 6 filtros 5 × 5, valid | 24 × 24 × 6 | ReLU | 156 |
| MaxPool 2 × 2 | 12 × 12 × 6 | — | 0 |
| Conv2D, 16 filtros 5 × 5, valid | 8 × 8 × 16 | ReLU | 2.416 |
| MaxPool 2 × 2 | 4 × 4 × 16 | — | 0 |
| Flatten | 256 | — | 0 |
| Dense | 120 | ReLU | 30.840 |
| Dense | 84 | ReLU | 10.164 |
| Dense de saída | 10 logits | Linear | 850 |
| **Total** | | | **44.426** |

O custo teórico das camadas com pesos é aproximadamente 281.640 MACs por imagem.

## Ambiente e alvo

- container: `xilinx/vitis-ai-tensorflow2-cpu:ubuntu2004-3.5.0.300`
- Vitis AI: 3.5
- TensorFlow/Keras no container: 2.12.0
- quantizador: Vitis AI Quantizer for TensorFlow 2
- estratégia: PTQ INT8 `pof2s`
- compilador: `vai_c_tensorflow2`
- placa: AMD/Xilinx ZCU104
- arquitetura compilada: `DPUCZDX8G_ISA1_B4096`
- batch compilado: 1
- layout: NHWC

O XModel somente pode ser executado em um bitstream cujo fingerprint/configuração do DPU corresponda ao alvo usado pelo Compiler. Antes do teste na placa, `xdputil query` deve confirmar um DPU compatível.

## Metodologia dos dados

O arquivo original é o MNIST oficial armazenado em `data/cache/keras/datasets/mnist.npz`, SHA-256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.

Calibração:

- divisão oficial de treino, sem usar imagens de teste;
- 10.000 imagens, sem reposição;
- amostragem estratificada: exatamente 1.000 imagens de cada classe;
- semente fixa `20260825`;
- preprocessamento: conversão `uint8 -> float32`, divisão por `255.0` e canal final;
- batch de calibração: 32, totalizando 313 passos (o último parcial).

Validação:

- todas as 10.000 imagens da divisão oficial de teste;
- nenhuma imagem de teste participou da calibração;
- mesma normalização usada no treinamento e nos benchmarks CPU/GPU;
- acurácia reportada com intervalo de confiança Wilson de 95%;
- comparação de argmax e erro dos logits entre float e INT8.

## Conversão Keras 3 para Keras 2

O Vitis AI 3.5 utiliza TensorFlow/Keras 2.12. Por isso, a arquitetura foi reconstruída com os mesmos nomes de camadas e os pesos do H5 original foram importados por nome.

A conversão foi aprovada com:

- 10 de 10 tensores de kernel/bias exatamente iguais, erro máximo `0.0`;
- 256 de 256 decisões de argmax iguais na prova de conversão;
- erro absoluto médio dos logits `0,0009955`;
- erro absoluto máximo dos logits `0,0061842`.

A pequena diferença nos logits decorre da execução em backends distintos (TensorFlow 2.21/GPU no modelo de origem e TensorFlow 2.12/CPU no container); os pesos foram comparados separadamente e são bit a bit idênticos.

## Resultados float e INT8

| Métrica, 10.000 testes | Float Keras 2.12 | INT8 Vitis Quantizer | Diferença INT8 − float |
|---|---:|---:|---:|
| Acertos | 9.898 | 9.895 | −3 |
| Acurácia | 98,98% | 98,95% | −0,03 p.p. |
| IC Wilson 95% | 98,7634%–99,1590% | 98,7306%–99,1318% | — |
| Concordância de argmax com float | — | 9.995/10.000 (99,95%) | — |
| MAE dos logits vs. float | — | 0,1075625 | — |
| RMSE dos logits vs. float | — | 0,1397367 | — |
| Erro absoluto máximo dos logits | — | 1,2333946 | — |

O critério de aceitação definido foi queda máxima de 1,0 ponto percentual. A queda observada foi somente 0,03 p.p.; portanto, a quantização foi aprovada.

| Dígito | Amostras | Float | INT8 |
|---:|---:|---:|---:|
| 0 | 980 | 99,286% | 99,286% |
| 1 | 1.135 | 99,648% | 99,648% |
| 2 | 1.032 | 99,322% | 99,322% |
| 3 | 1.010 | 99,109% | 99,109% |
| 4 | 982 | 99,695% | 99,695% |
| 5 | 892 | 98,991% | 98,991% |
| 6 | 958 | 99,165% | 98,956% |
| 7 | 1.028 | 97,665% | 97,763% |
| 8 | 974 | 98,460% | 98,255% |
| 9 | 1.009 | 98,414% | 98,414% |

As matrizes de confusão completas estão em `reports/float_validation.json` e `reports/quantized_validation.json`.

## Resultado do AI Compiler

O comando efetivamente executado no container foi:

```bash
vai_c_tensorflow2 \
  --model /workspace/artifacts/quantized/lenet_mnist_no_softmax_quantized.h5 \
  --arch /workspace/config/arch_zcu104_vai3.5.json \
  --output_dir /workspace/artifacts/compiled/lenet_mnist_zcu104_vai3_5 \
  --net_name lenet_mnist_no_softmax
```

O Compiler retornou código 0 e produziu três subgrafos:

| Subgrafo | Dispositivo | Operações | Entrada/saída |
|---|---|---:|---|
| `subgraph_input` | USER | 1 | saída INT8 `[1,28,28,1]`, fixed-point 6 |
| `subgraph_quant_conv1` | DPU | 20 | INT8 `[1,28,28,1]` → INT8 `[1,10]` |
| `subgraph_quant_output_fix_` | CPU | 1 | INT8 `[1,10]`, fixed-point 2 → float32 |

O runner VART criado diretamente para o subgrafo DPU recebe `int8` com escala `2^6` e devolve `int8` com escala `2^-2`. Os scripts da placa implementam essas escalas explicitamente.

## Execução na ZCU104

Copie todo o diretório abaixo para a placa:

```text
artifacts/deploy/lenet_mnist_zcu104_vai3_5/
```

No Linux da ZCU104 com Vitis AI Runtime/XIR/VART compatível:

```bash
cd lenet_mnist_zcu104_vai3_5
python3 00_inspect_dpu.py
python3 01_smoke_inference.py
python3 02_validate_accuracy.py --samples 10000
```

Os scripts executam, respectivamente:

1. inspeção do ambiente, `xdputil query`, `xdputil status`, XIR e VART;
2. teste rápido batch 1, após 20 aquecimentos, comparado à decisão INT8 do host;
3. validação nas 10.000 imagens, após 100 aquecimentos, com acurácia, concordância com o host, matriz de confusão, throughput e latências média, desvio padrão, mínimo, p50, p90, p95, p99 e máximo.

Os resultados da placa serão gravados em `results/`. A acurácia real do **XModel executado no DPU** só pode ser declarada depois dessa etapa na ZCU104; os 98,95% acima são a validação do modelo quantizado no host, antes da execução física no DPU.

## Organização dos arquivos

```text
Vitis AI/
├── artifacts/
│   ├── quantized/     # H5 INT8 produzido pelo AI Quantizer
│   ├── compiled/      # XModel, meta.json e MD5 do AI Compiler
│   └── deploy/        # pacote autocontido para a ZCU104
├── board/             # inspeção, smoke test e validação completa via VART
├── config/            # alvo DPU e configuração do experimento
├── data/
│   ├── cache/         # MNIST original local
│   └── prepared/      # calibração balanceada, teste e prova de conversão
├── docker/            # wrapper reprodutível do container Vitis AI 3.5
├── logs/compiler/     # saída integral do vai_c_tensorflow2
├── manifests/         # inventário e SHA-256 de todos os artefatos
├── models/float/      # H5 original e reconstrução Keras 2.12
├── reports/           # relatórios JSON por etapa
├── results/           # logits e predições float/INT8 de todas as imagens
└── scripts/           # estágios 00 a 09 do pipeline
```

## Reprodução

O orquestrador é:

```bash
cd '/home/miguel/Downloads/Plano testes TCC/LeNet/Vitis AI'
./run_full_flow.sh
```

Os scripts de compilação e empacotamento recusam diretórios de saída não vazios para não sobrescrever silenciosamente esta execução. Para refazer o experimento, preserve ou mova os artefatos existentes e utilize diretórios de saída novos.

## Relatórios e rastreabilidade

- `reports/keras3_reference.json`: identidade e arquitetura do H5 original;
- `reports/keras2_conversion.json`: equivalência de pesos e prova de logits;
- `reports/mnist_preparation.json`: hashes, semente, classes e separação dos dados;
- `reports/float_validation.json`: acurácia float e matriz de confusão;
- `reports/quantization.json`: parâmetros e artefato do Quantizer;
- `reports/quantized_validation.json`: acurácia INT8 e comparação com float;
- `reports/compilation.json`: comando, alvo, retorno e hash do XModel;
- `reports/xmodel_inspection.json`: subgrafos, tensores e fixed-points;
- `logs/compiler/compile_lenet_mnist.log`: log integral do AI Compiler;
- `manifests/HOST_FLOW_SHA256SUMS.txt`: checksums do fluxo completo;
- `artifacts/deploy/lenet_mnist_zcu104_vai3_5/SHA256SUMS.txt`: checksums do pacote da placa.

## Estado da implementação

| Etapa | Estado | Evidência |
|---|---|---|
| H5 original e arquitetura | concluída | `reports/keras3_reference.json` |
| Conversão Keras 3 → Keras 2.12 | concluída | pesos bit a bit iguais em `reports/keras2_conversion.json` |
| Preparação da calibração e teste | concluída | `reports/mnist_preparation.json` |
| PTQ INT8 com AI Quantizer | concluída | `reports/quantization.json` |
| Validação INT8 no host | concluída | 98,95% em `reports/quantized_validation.json` |
| AI Compiler para B4096 | concluída | retorno 0 em `reports/compilation.json` |
| Inspeção estrutural do XModel | concluída | um subgrafo DPU em `reports/xmodel_inspection.json` |
| Integridade do pacote de deploy | concluída | todos os SHA-256 verificados |
| Compatibilidade com o DPU físico | concluída | dois núcleos `DPUCZDX8G_ISA1_B4096` identificados |
| Smoke test VART | concluído | execução confirmada na ZCU104 |
| Validação das 10.000 imagens no DPU | concluída | 98,95% e concordância integral com a referência INT8 |
| Benchmark estatístico definitivo | concluído | campanhas preservadas em `results/zcu104_physical_20260908/` |
| Potência e energia da ZCU104 | concluída | telemetria e agregados preservados com os resultados físicos |

O script `02_validate_accuracy.py` é uma validação funcional de uma passagem. Ele não substitui a campanha definitiva de 100 ciclos e não deve fornecer os números principais de desempenho do TCC.

## Protocolo exigido para comparação com CPU e GPU

As campanhas CPU/GPU usaram precisão float32; o XModel utiliza INT8. Portanto, as comparações de desempenho são válidas como comparação entre implementações do mesmo classificador, mas a diferença de precisão numérica deve aparecer em todas as tabelas.

Parâmetros obrigatórios da campanha principal na ZCU104:

- batch 1, execução serial e síncrona, uma inferência em voo;
- 100 warm-ups excluídos das medições;
- seed `20260825`;
- tamanhos `N = 100`, `1.000` e `10.000`;
- 100 ciclos completos para cada N;
- resumos cumulativos dos primeiros 10, 20, 50 e 100 ciclos;
- mesma ordem determinística e aninhada usada nos scripts CPU/GPU;
- permutação interna diferente e determinística por ciclo;
- nenhum outlier removido;
- `time.perf_counter_ns()` e término do cronômetro somente após `runner.wait()`;
- nenhuma escrita em disco ou impressão por imagem dentro da região medida;
- checkpoint ao final de cada ciclo, sem sobrescrever campanhas anteriores.

O algoritmo exato dos subconjuntos é:

1. embaralhar separadamente os índices de cada classe usando `numpy.random.default_rng(20260825)`;
2. intercalar as primeiras 100 imagens de cada classe, produzindo as primeiras 1.000 imagens;
3. concatenar e embaralhar os índices restantes;
4. formar a ordem final com as 1.000 balanceadas seguidas do restante;
5. usar os prefixos 100, 1.000 e 10.000 dessa ordem.

Assim, N=100 contém 10 imagens por classe, N=1.000 contém 100 por classe e N=10.000 contém todo o teste oficial. Os hashes SHA-256 dos índices `int64`, em bytes contíguos, devem ser:

- N=100: `23b05c021258beb5fed85d820566f0e428e1033b80d93144bd029a13192e3994`;
- N=1.000: `d937fca9cab717946d8b85c5dc6a7363a17f392c800baef071c97e0ae4913661`;
- N=10.000: `65a5248b6e84b1b5bc6bd1068aba612e7884ac5c781f686d75d3843cc7ebd8a9`.

Para cada N, gerar as 100 permutações com um gerador inicializado por `seed + N × 1009`, exatamente como nos benchmarks de alto nível.

## Fronteiras temporais que devem permanecer separadas

### Inference-only

Entrada INT8 já normalizada, quantizada, no layout correto e em memória contígua. O cronômetro inclui somente `execute_async` e `runner.wait`. Normalização, quantização, cópia/preparação, dequantização, argmax e escrita ficam fora. É a fronteira mais próxima da chamada de inferência cronometrada em CPU/GPU, mas ainda pode incluir cópias internas do VART.

### Vazão efetiva batch 1

O dataset inteiro fica pré-quantizado antes do ciclo. O tempo de cada ciclo inclui laço Python, seleção/cópia da entrada para o buffer, `execute_async`, `wait`, leitura da saída, argmax e armazenamento da predição. Esta é a principal vazão comparável à vazão efetiva CPU/GPU.

### End-to-end

A imagem `uint8` já está residente em RAM. O tempo inclui conversão para float32, divisão por 255, quantização INT8, clipping, cópia, execução VART, dequantização e argmax. Carga do NPZ, carga do XModel, criação do runner, alocação e warm-up ficam fora.

### Saturado serial

Reutilizar a mesma entrada e os mesmos buffers, sem preprocessamento ou argmax na janela. Usar oito janelas de dez segundos para throughput e 50.000 inferências para a distribuição de latência. Continuar com batch 1 e uma inferência em voo. Publicar separadamente e não chamar de throughput máximo absoluto do DPU.

Experimentos concorrentes com múltiplos runners, threads ou jobs em voo são opcionais e nunca devem substituir a tabela principal serial batch 1.

## Estatística obrigatória da campanha ZCU104

Para cada cenário, N e prefixo de ciclos, preservar todas as latências individuais e calcular:

- número de medições, média, mediana, desvio-padrão amostral (`ddof=1`), coeficiente de variação, p90, p95, p99, mínimo e máximo;
- IC95 da média de latência usando as médias dos ciclos como observações independentes operacionais;
- FPS de cada ciclo, média, mediana, desvio-padrão amostral, IC95 e FPS efetivo global;
- FPS efetivo global como `total de inferências / soma dos tempos`, nunca como média simples dos FPS;
- acertos e acurácia somente sobre imagens únicas, IC95 Wilson e matriz de confusão;
- concordância e divergências do DPU contra a referência INT8 do host;
- hashes dos índices, predições e arquivos brutos.

Repetições caracterizam desempenho, mas não aumentam artificialmente o tamanho da amostra usado no intervalo de confiança da acurácia. Os prefixos 10/20/50/100 pertencem à mesma série e não são campanhas independentes.

## Energia e telemetria na ZCU104

Primeiro inventariar os sensores disponíveis em `/sys/class/hwmon`, `sensors` e, quando existir, `pynq.pmbus.get_rails()`. O rail preferencial para comparação entre implementações na mesma ZCU104 é `12V_power`.

Requisitos:

- registrar nome do rail, unidade, escopo físico e frequência real de atualização;
- medir baseline ocioso por pelo menos 5 segundos após inicializar DPU/runner e concluir warm-up;
- usar intervalo desejado de 1 segundo para PMBus, salvo limitação documentada;
- preservar timestamps monotônicos e amostras brutas;
- integrar potência pela sobreposição temporal com cada ciclo/janela;
- medir e documentar o overhead do coletor;
- calcular potência total, potência dinâmica, energia total e dinâmica por inferência;
- reportar cobertura temporal da telemetria e recusar ciclos sem cobertura suficiente;
- nunca somar rails sobrepostos nem substituir `12V_power` silenciosamente por outro rail.

As potências não têm automaticamente o mesmo limite físico: CPU usa RAPL package, GPU usa sensor da placa e ZCU104 deve declarar o rail PMBus. Energia só deve ser colocada lado a lado junto do escopo de cada sensor.

## Resultados físicos

A execução na ZCU104 obteve 98,95% de acurácia e concordância integral com a referência INT8. No cenário de inferência, a menor latência média foi 0,2256 ms com uma thread e a maior vazão foi 5.291,79 inferências por segundo com duas threads. No cenário saturado, o melhor resultado foi 9.292,01 inferências por segundo com três threads. Os dados completos estão em `results/zcu104_physical_20260908/` e a síntese comparativa está em `../../README_RESULTADOS_VITIS_AI_HLS4ML.md`.
