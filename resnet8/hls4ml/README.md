# Fluxo hls4ml da ResNet8 para ZCU104

Esta pasta concentra scripts, configurações, dados, builds, logs, relatórios e artefatos do fluxo hls4ml. O alvo reproduz o projeto anterior: ZCU104, Vitis/Vivado 2024.2, `io_stream`, estratégia `Resource`, convoluções `LineBuffer`, `ap_fixed<22,12,AP_RND_CONV,AP_SAT>`, clock de 10 ns e `ReuseFactor` máximo de 288.

## Decisão principal

Use `../resnet8_cifar10_keras3_no_softmax.h5` para gerar o acelerador. O modelo original permanece como referência e como teste de equivalência. A justificativa completa está em `docs/DECISAO_SOFTMAX.md`.

O plano pedido replica o caso roteável do repositório anterior. Os valores solicitados 8/32/64/128 são ajustados pelo backend ao fator válido mais próximo (tipicamente 9/36/64/128); `conv2d_6` e `conv2d_7` permanecem em 288. O CSV gerado em cada build registra pedido, valor efetivo e fatores válidos.

## Estrutura

```text
hls4ml/
├── configs/       parâmetros do dispositivo e plano de reuse
├── data/          CIFAR-10 preparado para validação/testbench
├── docs/          decisões técnicas e procedimento FIFO
├── environment/   dependências e criação do ambiente Python
├── models/        inventário e checksums dos modelos-fonte
├── scripts/       auditoria, conversão, síntese e otimização
├── tcl/           exportação, validação de catálogo e síntese OOC
├── builds/        projetos gerados pelo hls4ml/Vitis
├── logs/          logs capturados pelo usuário
├── reports/       auditorias e resumos consolidados
├── ip_repo/       repositórios IP-XACT descompactados para o Vivado
├── ip_exports/    arquivos ZIP distribuíveis
└── ip_validation/ relatórios, logs e XCI dos testes de catálogo/OOC
```

## Ambiente detectado

```text
Python       3.12.7
TensorFlow   2.21.0
Keras        3.13.2
hls4ml       1.3.0
NumPy        1.26.4
Vitis HLS    2024.2 (/opt/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls)
Vivado       2024.2 (/opt/Xilinx/Vivado/2024.2/bin/vivado)
Part         xczu7ev-ffvc1156-2-e
Clock        10.0 ns (100 MHz)
```

Para criar um ambiente isolado do zero:

```bash
bash environment/create_venv.sh
source .venv/bin/activate
```

## Ordem de execução

Todos os comandos partem desta pasta:

```bash
cd hls4ml

# 0. Ferramentas, versões e modelos
python scripts/00_check_environment.py
python scripts/01_inspect_models.py

# 1. Dados de validação
python scripts/02_prepare_cifar10.py

# 2. Geração e validação C++ rápida, sem HLS synthesis
python scripts/03_build_baseline.py --validate-samples 100

# 3. C synthesis do baseline
python scripts/03_build_baseline.py --out builds/baseline_synth --synth --validate-samples 100

# 4. Profiling RTL e reescrita das profundidades FIFO
python scripts/04_optimize_fifo.py --tb-samples 2

# 5. Opcional: além do FIFO flow, exportar IP e Vivado synthesis
python scripts/04_optimize_fifo.py --out builds/fifo_opt_vsynth --tb-samples 2 --vsynth
```

Os scripts resolvem caminhos a partir desta pasta, portanto também funcionam se chamados de outro diretório. Um diretório de build existente nunca é apagado silenciosamente; use outro `--out` ou passe `--force` conscientemente.

Como o caminho pai contém espaços e o Vitis HLS 2024.2 os proíbe em projetos, os scripts criam automaticamente um symlink exclusivo em `/tmp/resnet8_hls4ml_<hash>`. O alias só fornece à ferramenta um nome sem espaços: todos os arquivos continuam fisicamente em `hls4ml/`.

## Critérios antes de avançar ao Vivado

- concordância top-1 Keras/hls4ml medida e documentada;
- reuse efetivo de todas as camadas `<= 288`;
- C synthesis concluída sem deadlock;
- `fifo_depths.json` sem canal no limite de profiling;
- C synthesis final feita depois da reescrita dos FIFOs;
- relatórios de LUT, FF, DSP, BRAM, URAM, latência e II preservados no build;
- somente depois disso, exportar/promover o IP para `artifacts/ip/`.

O empacotamento especial usado no repositório anterior (sobrepor `syn/verilog` no IP exportado para eliminar RTL stale com FIFOs `d4096`) não deve ser aplicado automaticamente. Primeiro verifique se o hls4ml 1.3.0/Vitis 2024.2 ainda apresenta o problema; o script FIFO grava uma auditoria das instâncias `d4096` para essa decisão.

## IP promovido

O export padrão do Vitis HLS 2024.2 foi testado e reintroduziu profundidade `4096` nos 42 FIFOs. Esse pacote 1.0 foi rejeitado. O pacote promovido foi reconstruído somente com o RTL de `solution1/syn/verilog`, recebeu VLNV `xilinx.com:hls:resnet8_resource_fifo_opt:1.1` e passou por:

- auditoria do top e dos módulos FIFO;
- registro no catálogo Vivado;
- `validate_ip`;
- geração de todos os targets;
- síntese Out-of-Context do `.xci` com progresso 100% e zero erros.

Artefatos finais:

```text
ip_repo/resnet8_resource_fifo_opt_v1_1/                    IP-XACT descompactado
ip_exports/xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip
ip_validation/resnet8_resource_fifo_opt_v1_1/              relatórios e XCI
```

Utilização da síntese OOC na ZCU104:

| Recurso | Uso | Disponível | Percentual |
|---|---:|---:|---:|
| CLB LUT | 165.564 | 230.400 | 71,86% |
| CLB Registers | 178.894 | 460.800 | 38,82% |
| BRAM tiles | 74,5 | 312 | 23,88% |
| URAM | 11 | 96 | 11,46% |
| DSP48E2 | 1.174 | 1.728 | 67,94% |

Os Tcl reproduzíveis estão em `tcl/`. O script `scripts/05_make_correct_ip.py` recebe o scaffold do export padrão e o diretório `syn/verilog` final, elimina VHDL e módulos residuais `d4096`, atualiza os file sets IP-XACT e gera o ZIP corrigido.
