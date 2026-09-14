# LeNet — MNIST

Este diretório reúne os arquivos da rede LeNet utilizada na classificação do conjunto MNIST. O modelo recebe imagens de 28 × 28 pixels em escala de cinza e produz dez logits para classificação por `argmax`.

O conteúdo inclui o treinamento e os modelos de referência, benchmarks em CPU e GPU, quantização e compilação pelo Vitis AI, geração de acelerador pelo hls4ml e resultados físicos obtidos na ZCU104.

## Organização

- `CPU/`: scripts, latências, predições, telemetria e resultados no processador.
- `GPU/`: scripts, latências, predições e telemetria na GPU.
- `Vitis AI/`: conversão compatível com Keras 2.12, PTQ INT8, XModel, execução VART e resultados da DPU.
- `hls4ml/`: configuração em ponto fixo, IP, relatórios HLS/Vivado, overlay e resultados na placa.
- `scripts/`: utilitários de exportação e documentação.

## Arquivos de referência

- `lenet_mnist_final.h5` e `lenet_mnist_final.keras`: modelos treinados.
- `train_lenet_mnist.py`: treinamento do modelo.
- `hls4ml/hardware/deploy/`: bitstream, arquivo HWH e scripts do overlay.
- `hls4ml/resultados_zcu104/`: validação, desempenho e telemetria da implementação dedicada.

Os resultados consolidados dos fluxos Vitis AI e hls4ml estão em `../README_RESULTADOS_VITIS_AI_HLS4ML.md`.
