# MLP — Iris

Este diretório reúne os arquivos utilizados na avaliação da rede MLP aplicada ao conjunto Iris. A arquitetura possui quatro entradas, duas camadas ocultas com oito neurônios cada e três saídas, totalizando 139 parâmetros treináveis.

Foram preservados o modelo Keras, o normalizador dos dados, os scripts de benchmark em CPU e GPU, os arquivos dos fluxos Vitis AI e hls4ml e as coletas realizadas na ZCU104.

## Organização

- `CPU/`: resultados de latência, vazão, potência e energia obtidos no processador do computador de referência.
- `GPU/`: resultados e telemetria das execuções na GPU NVIDIA.
- `Vitis AI/`: preparação dos dados, quantização INT8, compilação do XModel e resultados na DPU da ZCU104.
- `hls4ml/`: conversão para ponto fixo, síntese HLS, integração no Vivado, bitstream e medições físicas na ZCU104.
- `scripts/`: scripts utilizados nos benchmarks e na consolidação das campanhas.
- `manifests/`: inventários e registros de integridade dos artefatos.

## Arquivos de referência

- `iris_mlp_clean.h5`: modelo treinado.
- `iris_scaler.joblib`: normalizador aplicado aos quatro atributos.
- `METODOLOGIA_BENCHMARK_MLP_IRIS.md`: protocolo experimental.
- `hls4ml/resultados_zcu104/`: dados brutos e agregados da implementação dedicada.

A acurácia de referência foi de 96,67% no conjunto de teste. Os resultados consolidados dos dois fluxos FPGA estão disponíveis em `../README_RESULTADOS_VITIS_AI_HLS4ML.md`.
