# Acc-AI_models

Repositório de dados, código-fonte e artefatos produzidos na etapa experimental do Trabalho de Conclusão de Curso sobre aceleração de redes neurais em diferentes plataformas computacionais.

O estudo considera três modelos — MLP/Iris, LeNet/MNIST e ResNet8/CIFAR-10 — executados em CPU, GPU e na plataforma AMD/Xilinx ZCU104. Para a implementação em FPGA foram empregados dois fluxos: Vitis AI, com execução em DPU, e hls4ml, com geração de aceleradores dedicados.

## Estrutura

| Diretório | Descrição |
| --- | --- |
| `MLP/` | Modelo MLP para Iris, benchmarks e implementações Vitis AI e hls4ml |
| `LeNet/` | Modelo LeNet para MNIST, benchmarks e implementações Vitis AI e hls4ml |
| `resnet8/` | Modelo ResNet8 para CIFAR-10, benchmarks e implementações Vitis AI e hls4ml |
| `H1/` | Exploração de fatores de reutilização, síntese HLS e análise de recursos |
| `artifacts/` | Pacotes de IP exportados das variantes avaliadas |
| `docs/` | Metodologia, inventário e resultados gerais do trabalho |

Cada diretório principal possui um README próprio com a descrição do conteúdo, dos procedimentos realizados e dos arquivos de referência.

## Resultados dos fluxos FPGA

Os resultados consolidados de Vitis AI e hls4ml estão em [README_RESULTADOS_VITIS_AI_HLS4ML.md](README_RESULTADOS_VITIS_AI_HLS4ML.md). O documento apresenta acurácia, latência, vazão, potência, energia e resultados de síntese dos três modelos, preservando as fronteiras de medição utilizadas em cada experimento.

## Artefatos preservados

O repositório contém modelos Keras, modelos quantizados, arquivos XModel, código de benchmark, configurações, relatórios de síntese e implementação, IPs exportados, checkpoints, bitstreams e arquivos de descrição de hardware.

Os pares `.bit` e `.hwh` devem ser mantidos juntos, pois o primeiro contém a configuração do FPGA e o segundo descreve a organização do hardware utilizada pelo PYNQ.

## Validação do repositório

Antes de incluir uma nova coleta, recomenda-se executar:

```bash
./scripts/validate_repository.sh
```

O procedimento verifica arquivos acima do limite do GitHub, caches indevidamente versionados e padrões comuns de credenciais.
