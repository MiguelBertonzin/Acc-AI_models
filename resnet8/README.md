# ResNet8 — CIFAR-10

Este diretório reúne os arquivos da ResNet8 utilizada na classificação do conjunto CIFAR-10. Foram preservadas as versões do modelo com e sem Softmax; as implementações em FPGA utilizam a saída em logits e classificação por `argmax`.

O trabalho incluiu benchmarks em CPU e GPU, quantização e compilação para DPU pelo Vitis AI, geração de um acelerador dedicado pelo hls4ml e integração completa no Vivado para a ZCU104.

## Organização

- `CPU/`: benchmarks das versões com e sem Softmax.
- `GPU/`: benchmarks das versões com e sem Softmax.
- `Vitis AI/`: preparação do CIFAR-10, PTQ INT8, compilação do XModel e pacote de resultados da ZCU104.
- `hls4ml/`: configurações, scripts, IP, relatórios de síntese, resultados físicos e overlays.
- `hardware/`: fontes auxiliares de hardware.
- `scripts modelo/`: transformação do modelo para remoção do Softmax.
- `readme/`: registros metodológicos e instruções detalhadas de reprodução.

## Arquivos de referência

- `resnet8_cifar10_keras3.h5`: modelo original.
- `resnet8_cifar10_keras3_no_softmax.h5`: versão utilizada nos fluxos FPGA.
- `Vitis AI/resnet8_final_v1_RESULTS_2026-09-01.tar.gz`: pacote da coleta Vitis AI.
- `hls4ml/Resultados_ZCU104/`: resultados brutos e agregados da implementação dedicada.

Os resultados consolidados dos dois fluxos FPGA estão em `../README_RESULTADOS_VITIS_AI_HLS4ML.md`.
