# Acc-AI_models

Acervo técnico da etapa experimental do TCC sobre aceleração de modelos de IA. O repositório reúne modelos, scripts, configurações, resultados medidos e artefatos de compilação/síntese para **MLP/Iris**, **LeNet/MNIST** e **ResNet8/CIFAR-10**.

Snapshot organizado em **14 de setembro de 2026** a partir do diretório de trabalho local. A preparação deste repositório foi feita por cópia: os arquivos-fonte originais não foram movidos, renomeados, editados ou excluídos.

## Organização

| Caminho | Conteúdo |
| --- | --- |
| `MLP/` | Modelo Iris, scaler, benchmarks CPU/GPU, fluxo Vitis AI, fluxo hls4ml, Vivado e resultados ZCU104 |
| `LeNet/` | Modelos MNIST, treino, benchmarks CPU/GPU, Vitis AI, IP hls4ml, bitstream e resultados ZCU104 |
| `resnet8/` | Modelos CIFAR-10 com e sem softmax, benchmarks CPU/GPU, Vitis AI, hls4ml, IP, bitstreams e resultados ZCU104 |
| `ARM_CPU/` | Conversões TensorFlow Lite, scripts e resultados para execução ARM na ZCU104 |
| `H1/` | Exploração hls4ml por fator de reutilização, validações C/RTL, relatórios e checkpoints pós-síntese |
| `artifacts/hls-ip-exports/` | Pacotes IP exportados das variantes H1 de MLP e LeNet |
| `docs/` | Resultados consolidados, plano/metodologia e documentação de apoio |

Veja [docs/INVENTARIO.md](docs/INVENTARIO.md) para o inventário detalhado e [docs/CRITERIOS_DE_CURADORIA.md](docs/CRITERIOS_DE_CURADORIA.md) para entender o que foi preservado ou deliberadamente deixado fora desta cópia.

## Fluxos preservados

### Benchmarks de software

Os diretórios `CPU/` e `GPU/` de cada rede preservam scripts de benchmark e as coletas em CSV, JSON, NPY/NPZ e logs de telemetria. O fluxo ARM consolidado está em `ARM_CPU/`.

### Vitis AI

Cada diretório `Vitis AI/` contém, conforme disponível:

- scripts de preparação, validação, quantização PTQ, compilação e empacotamento;
- modelos float, quantizados e compilados (`.h5`, `.keras`, `.xmodel`);
- configurações de arquitetura da ZCU104;
- scripts de execução na placa;
- resultados, telemetria, manifests e hashes.

### hls4ml / Vitis HLS / Vivado

Os diretórios `hls4ml/` e `H1/` preservam:

- scripts geradores, configurações YAML/JSON e testbenches;
- firmware C++ e RTL/IP exportado;
- relatórios de C synthesis, co-simulação, utilização, timing, potência, rota e DRC;
- pacotes de IP reutilizáveis;
- artefatos de implantação ZCU104 (`.bit` + `.hwh`);
- checkpoints pós-síntese (`.dcp`) das explorações H1.

## Sobre arquivos binários

Os arquivos `.h5`, `.keras`, `.joblib`, `.xmodel`, `.bit`, `.hwh`, `.zip` de IP e `.dcp` foram mantidos quando representam um modelo, entrega compilada, overlay ou checkpoint relevante. Em especial, **o `.bit` programa o FPGA e o `.hwh` descreve o hardware para o PYNQ; os dois devem ser versionados juntos quando o objetivo é reproduzir um overlay**.

Não há Git LFS configurado neste snapshot. Todos os arquivos estão abaixo do limite rígido do GitHub; o maior é um checkpoint DCP de aproximadamente 99 MB (94,4 MiB).

## Como começar

```bash
git clone https://github.com/MiguelBertonzin/Acc-AI_models.git
cd Acc-AI_models
```

Depois, siga o README do fluxo desejado. Os ambientes são distintos: benchmarks de host usam Python/TensorFlow; Vitis AI depende da versão e imagem correspondentes; hls4ml/HLS e Vivado dependem das ferramentas AMD/Xilinx e da placa-alvo ZCU104.

Alguns manifests e relatórios registram caminhos absolutos da máquina em que o experimento foi executado. Eles foram mantidos como metadados históricos; ao reproduzir o fluxo, ajuste os caminhos para o seu ambiente.

## Integridade e manutenção

Há manifests SHA-256 específicos dentro dos fluxos. Antes de novos commits, execute:

```bash
./scripts/validate_repository.sh
```

O script detecta arquivos acima do limite do GitHub, caches acidentais e padrões comuns de credenciais. Consulte [CONTRIBUTING.md](CONTRIBUTING.md) antes de adicionar novas coletas.

