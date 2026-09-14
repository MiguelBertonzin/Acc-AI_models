# Inferência CPU ARM na ZCU104

Este diretório é autocontido e não altera os resultados consolidados de CPU,
GPU, hls4ml ou Vitis AI. Ele contém os modelos canônicos copiados, dados de
calibração/teste, conversores TFLite e o benchmark para os quatro Cortex-A53.

## Modelos finais

Cada rede possui duas variantes:

- `fp32`: comparação principal com CPU/GPU em ponto flutuante;
- `int8`: quantização inteira TFLite específica da CPU ARM.

A quantização TFLite INT8 é um experimento próprio e não é bit a bit equivalente
ao PTQ `pof2s` do Vitis AI.

## Estrutura

```text
ARM_CPU/
├── models/h5/          cópias imutáveis dos modelos canônicos
├── models/tflite/      modelos finais FP32 e INT8
├── data/               teste, calibração e scaler da MLP
├── scripts/            conversão, validação e benchmark ARM
├── results/            validações e futuras campanhas físicas
├── manifests/          configuração e hashes
├── deploy_zcu104/      pacote autocontido pronto para cópia à placa
└── docs/               documentação complementar
```

## Geração no host

```bash
cd ARM_CPU
python3 scripts/convert_models.py
python3 scripts/validate_models.py
python3 scripts/make_deploy.py
```

## Verificação na ZCU104

```bash
uname -m
python3 --version
ldd --version | head -1
python3 -c "import tflite_runtime.interpreter as tflite; print('OK')"
```

O runtime deve ser compatível com AArch64, Python 3.9 e a glibc da imagem.
Não é necessário instalar TensorFlow completo na placa.

Depois de copiar `deploy_zcu104/` à placa:

```bash
cd deploy_zcu104
sha256sum -c SHA256SUMS.txt
python3 scripts/check_board.py
python3 scripts/infer_arm.py --network mlp --precision fp32 --features 5.1 3.5 1.4 0.2
python3 scripts/infer_arm.py --network lenet --precision int8 --index 0
python3 scripts/infer_arm.py --network resnet8 --precision fp32 --index 0
```

## Benchmark CPU-only

Uma thread, restrita ao CPU 0:

```bash
taskset -c 0 python3 scripts/benchmark_arm.py \
  --network lenet --precision fp32 --threads 1 --inferences 10000
```

Quatro threads nos quatro Cortex-A53:

```bash
taskset -c 0-3 python3 scripts/benchmark_arm.py \
  --network lenet --precision int8 --threads 4 --inferences 10000
```

Redes válidas: `mlp`, `lenet`, `resnet8`.

O script não importa VART, não carrega delegate DPU e não programa overlay.
Assim, toda computação do modelo ocorre na CPU ARM. Ele registra latência de
`invoke()`, E2E, throughput, acurácia, afinidade, versões e hashes. Nenhum
outlier é removido.

## Campanha recomendada

Executar FP32 e INT8 com uma e quatro threads. Antes de cada configuração:

1. fixar e registrar governor/frequência;
2. executar 200 warm-ups fora da janela;
3. validar todas as amostras únicas;
4. executar cinco campanhas independentes;
5. coletar energia em processo separado com idle pareado;
6. registrar se o bitstream DPU permanece carregado, pois ele afeta a potência
   total da placa mesmo sem receber inferências.

Quantidades alinhadas aos ensaios existentes:

- MLP: cinco campanhas de 30.000 e cinco de 100.000;
- LeNet: 10.000 imagens × 100 ciclos, ou cinco campanhas de 100.000;
- ResNet8: 10.000 imagens × 100 ciclos após estimar a duração de uma campanha.

## Fronteiras

- inference-only: somente `Interpreter.invoke()`;
- E2E: preparação da amostra, quantização quando aplicável, cópia de entrada,
  `invoke()`, cópia da saída e `argmax`;
- throughput: tempo global do laço completo.

Para energia, manter separados potência total da placa e potência dinâmica.
Não comparar diretamente com RAPL da CPU Intel ou com o sensor da RTX 3050
sem explicitar a diferença de escopo.
