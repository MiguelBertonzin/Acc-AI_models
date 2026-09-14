# Inventário do snapshot

Data da curadoria: **2026-09-14**.

| Área | Arquivos | Tamanho aproximado | Destaques |
| --- | ---: | ---: | --- |
| `MLP/` | 944 | 93 MB | Iris, CPU/GPU, Vitis AI, hls4ml, Vivado e ZCU104 |
| `LeNet/` | 458 | 79 MB | MNIST, CPU/GPU, Vitis AI, IP/overlay hls4ml e ZCU104 |
| `resnet8/` | 438 | 96 MB | CIFAR-10, CPU/GPU, Vitis AI, IP/overlays hls4ml e ZCU104 |
| `H1/` | 1.033 | 218 MB | varreduras de reuse factor, relatórios, validações e DCPs |
| `ARM_CPU/` | 56 | 14 MB | TensorFlow Lite, implantação e resultados ARM/ZCU104 |
| `artifacts/` | 8 | 8,9 MB | exports IP HLS das variantes H1 |
| `docs/` | 4 + documentação do repositório | 112 KB + | consolidação e metodologia |

O snapshot completo tem cerca de **508 MB** antes da compactação interna do Git.

## Modelos de referência

- MLP/Iris: `MLP/iris_mlp_clean.h5` e `MLP/iris_scaler.joblib`.
- LeNet/MNIST: `LeNet/lenet_mnist_final.h5` e `LeNet/lenet_mnist_final.keras`.
- ResNet8/CIFAR-10: `resnet8/resnet8_cifar10_keras3.h5` e `resnet8/resnet8_cifar10_keras3_no_softmax.h5`.

## Artefatos de placa

- MLP hls4ml: `MLP/hls4ml/vivado_zcu104_mlp/output/`.
- LeNet hls4ml: `LeNet/hls4ml/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz/`.
- ResNet8 hls4ml: `resnet8/hls4ml/Resultados_ZCU104/`.
- Vitis AI: subdiretórios `artifacts/compiled`, `artifacts/deploy` e `results` de cada rede.
- CPU ARM: `ARM_CPU/deploy_zcu104/`.

## Exploração H1

- MLP: RF1, RF4, RF8, RF16 e RF32.
- LeNet: RF32, RF64 e RF128.
- Cada variante mantém configuração efetiva, fonte/firmware, teste, resumo de validação, relatórios Vivado e checkpoint pós-síntese quando disponível.
- Os IPs finais exportados também estão reunidos em `artifacts/hls-ip-exports/` para acesso rápido.

## Resultados consolidados

Os arquivos históricos copiados para `docs/` incluem:

- `RESULTADOS_FINAIS_TRES_REDES.txt`;
- `testes.txt`;
- `texto-testes.txt`;
- `RESNET8_ZCU104_BLOCK_DESIGN_REPLICATION_GUIDE.txt`.

Os resultados brutos permanecem junto ao respectivo modelo e backend, evitando perder o vínculo entre script, configuração e coleta.

