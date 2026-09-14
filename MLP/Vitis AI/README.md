# MLP Iris — fluxo Vitis AI 3.5 para ZCU104

Este diretório contém todo o fluxo reproduzível de conversão da MLP Iris para
um `.xmodel` executável pela DPU da ZCU104. A coleta na placa deve seguir
[`../METODOLOGIA_BENCHMARK_MLP_IRIS.md`](../METODOLOGIA_BENCHMARK_MLP_IRIS.md).

## Documentação técnica

Este diretório reúne os scripts de implantação, validação e benchmark do XModel na ZCU104. Os dados físicos estão em [`results/zcu104_final_20260909T204251Z/`](results/zcu104_final_20260909T204251Z/) e os valores consolidados constam em [`../../README_RESULTADOS_VITIS_AI_HLS4ML.md`](../../README_RESULTADOS_VITIS_AI_HLS4ML.md).

## Resultado final

| Item | Resultado |
|---|---:|
| Modelo | MLP 4→8→8→3, 139 parâmetros |
| Quantização | PTQ INT8, estratégia `pof2s` |
| AI Quantizer | Vitis AI 3.5 / TensorFlow 2.12 |
| AI Compiler | `vai_c_tensorflow2` |
| Alvo | `DPUCZDX8G_ISA1_B4096` |
| Calibração | 120 registros de treino, 40 por classe |
| Registros de teste usados na calibração | 0 |
| Acurácia float | 96,67% (29/30) |
| Acurácia quantizada | 96,67% (29/30) |
| Concordância INT8 × float | 100% (30/30) |
| Subgrafos DPU | 1 |
| Tamanho do XModel | 109.221 bytes |

O artefato compilado está em
[`artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel`](artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel).

O pacote para copiar à placa está em
[`artifacts/deploy/iris_mlp_zcu104_vai3_5/`](artifacts/deploy/iris_mlp_zcu104_vai3_5/).

## Calibração

Iris é um conjunto tabular, portanto o Quantizer recebe vetores de quatro
atributos, e não imagens. Foram usadas todas as 120 amostras do subconjunto de
treino, o máximo disponível sem contaminar o teste:

- 40 amostras da classe Setosa;
- 40 da classe Versicolor;
- 40 da classe Virginica;
- 120 registros no total, correspondentes a 119 vetores de atributos distintos;
- oito batches de calibração, com `batch_size=16`;
- 30 amostras exclusivas reservadas para validação.

Repetir artificialmente as mesmas amostras não acrescentaria novos intervalos
de ativação ao PTQ. Por isso, a calibração prioriza cobertura real e ausência de
vazamento em vez de duplicação do dataset.

Os dados e a auditoria da divisão estão em:

- [`data/prepared/iris_calibration_train.npz`](data/prepared/iris_calibration_train.npz);
- [`data/prepared/iris_test.npz`](data/prepared/iris_test.npz);
- [`reports/data_preparation.json`](reports/data_preparation.json).

## Softmax e equivalência

O modelo original produz probabilidades por softmax. O modelo entregue à DPU
termina em logits, pois `argmax(softmax(logits)) = argmax(logits)`. A remoção
do softmax:

- preservou as 30 decisões do modelo original;
- manteve a acurácia de 96,67%;
- permite que as três camadas densas sejam compiladas no subgrafo DPU;
- deixa o softmax como pós-processamento opcional no ARM quando probabilidades
  forem necessárias.

A reconstrução Keras 2.12 apresentou erro máximo de probabilidade de
`1,1920929 × 10⁻⁷` em relação ao modelo original.

## Validação da quantização

| Métrica | Resultado |
|---|---:|
| Acurácia float | 96,67% |
| Acurácia INT8 simulada | 96,67% |
| Concordância de classes | 100% |
| Erro absoluto máximo nos logits | 0,22719145 |
| Erro absoluto médio nos logits | 0,04820604 |
| Erro absoluto máximo nas probabilidades | 0,03152779 |
| Erro absoluto médio nas probabilidades | 0,00365869 |

Esses valores são obtidos pelo modelo quantizado do Vitis AI em software. A
validação bit a bit do XModel e as métricas de desempenho exigem a DPU física
da ZCU104.

## Grafo compilado

A inspeção XIR encontrou três subgrafos:

| Subgrafo | Dispositivo | Operações | Interface |
|---|---|---:|---|
| Quantização da entrada | USER | 1 | `float32 → int8` |
| MLP 4→8→8→3 | DPU | 13 | `int8[1,4] → int8[1,3]` |
| Dequantização da saída | CPU | 1 | `int8 → float32` |

Para execução direta com `vart.Runner`, o script fornecido quantiza e
dequantiza os tensores usando:

- entrada: shape `[1,4]`, `int8`, `fix_point=5`;
- saída: shape `[1,3]`, `int8`, `fix_point=3`.

SHA-256 do XModel:

```text
47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce
```

## Organização

```text
Vitis AI/
├── source/                 modelo e scaler originais usados no fluxo
├── config/                 arquitetura DPU e pré-processamento
├── data/prepared/          calibração e teste separados
├── models/float/           modelo Keras 2.12 com saída em logits
├── artifacts/quantized/    H5 produzido pelo AI Quantizer
├── artifacts/compiled/     XModel produzido pelo AI Compiler
├── artifacts/deploy/       pacote autocontido para a ZCU104
├── board/                  executor, validador, benchmark e sensores
├── scripts/                etapas reproduzíveis do fluxo
├── reports/                métricas e auditorias JSON
├── results/                saídas float e quantizadas
├── logs/                   log integral do compiler
├── manifests/              inventário e hashes
├── docker/                 wrapper do container Vitis AI
└── run_all.sh              executor sequencial
```

## Reprodução

A imagem já utilizada neste computador é:

```text
xilinx/vitis-ai-tensorflow2-cpu:ubuntu2004-3.5.0.300
```

Em uma cópia limpa, execute a partir da raiz deste diretório:

```bash
cd "Vitis AI"
./run_all.sh
```

O fluxo executa, em ordem:

1. preparação dos dados;
2. reconstrução compatível com Keras 2.12;
3. validação float;
4. AI Quantizer PTQ INT8;
5. validação quantizada;
6. AI Compiler para ZCU104;
7. inspeção XIR;
8. empacotamento para a placa;
9. geração do manifesto SHA-256.

Por segurança, a compilação não sobrescreve resultados não vazios. O empacotamento atualiza somente os arquivos gerenciados do pacote de deploy. Para uma nova compilação, preserve os diretórios anteriores e use uma saída vazia.

## Conteúdo do pacote de placa

Além do XModel, o pacote `artifacts/deploy/iris_mlp_zcu104_vai3_5/` agora contém os 30 vetores canônicos, referências float/INT8, executor unitário, validador completo, benchmark padronizado e descoberta de sensores. Todos são protegidos por `SHA256SUMS.txt`. A sintaxe dos scripts foi validada no host; a execução VART depende da DPU física.

Na placa, a ordem segura é:

```bash
sha256sum -c SHA256SUMS.txt
python3 validate_iris_mlp.py --output-dir validation_output
python3 discover_zcu104_sensors.py > sensors_inventory.json
python3 benchmark_iris_mlp.py --inferences 30000 --output-dir campaigns/30000/rep01
```

O benchmark bloqueia automaticamente se não obtiver 29/30 e concordância 30/30 com a referência quantizada. Sem um sensor calibrado informado por `--sensor NAME=PATH:MULTIPLIER_TO_WATTS` e `--primary-power-sensor NAME`, energia fica explicitamente como `not_measured`.

## Execução na ZCU104

Copie o conteúdo de
`artifacts/deploy/iris_mlp_zcu104_vai3_5/` para uma imagem de sistema da
ZCU104 que contenha Vitis AI Runtime compatível com a DPU
`DPUCZDX8G_ISA1_B4096`. Na placa:

```bash
python3 run_iris_mlp.py --features 5.1 3.5 1.4 0.2
```

A ordem das entradas é comprimento da sépala, largura da sépala, comprimento
da pétala e largura da pétala, todas em centímetros. O script aplica o mesmo
`StandardScaler`, faz a conversão INT8, executa a DPU, dequantiza os logits e
retorna classe e probabilidades.

## Relatórios principais

- [`reports/keras2_conversion.json`](reports/keras2_conversion.json)
- [`reports/float_validation.json`](reports/float_validation.json)
- [`reports/quantization.json`](reports/quantization.json)
- [`reports/quantized_validation.json`](reports/quantized_validation.json)
- [`reports/compilation.json`](reports/compilation.json)
- [`reports/xmodel_inspection.json`](reports/xmodel_inspection.json)
- [`logs/compiler/compile_iris_mlp.log`](logs/compiler/compile_iris_mlp.log)
- [`manifests/flow_manifest.json`](manifests/flow_manifest.json)
- [`manifests/SHA256SUMS.txt`](manifests/SHA256SUMS.txt)
