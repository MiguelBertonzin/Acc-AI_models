# Exploração H1 — hls4ml

Este diretório contém a exploração de configurações hls4ml para os modelos MLP e LeNet. O objetivo foi avaliar o efeito do fator de reutilização sobre latência estimada, consumo de recursos e fechamento de timing na ZCU104.

## Configurações avaliadas

- MLP: fatores de reutilização 1, 4, 8, 16 e 32.
- LeNet: fatores de reutilização 32, 64 e 128.

Cada variante preserva a configuração solicitada e efetiva, firmware gerado, testbench, validação C/RTL, relatório de síntese, utilização de recursos, timing e checkpoint pós-síntese quando disponível.

Os scripts `run_h1.py`, `audit_h1.py`, `collect_vivado.py` e `run_vivado_ooc.tcl` automatizam a geração, auditoria e coleta dos resultados. Os arquivos `resultados_hls.csv` e `vivado_ooc_summary.json` apresentam os dados consolidados por modelo.

Os pacotes IP exportados também foram reunidos em `../artifacts/hls-ip-exports/`.
