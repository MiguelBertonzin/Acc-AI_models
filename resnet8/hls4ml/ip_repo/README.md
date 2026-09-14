# Repositórios IP

Cada subpasta desta pasta é uma raiz de repositório IP-XACT que pode ser adicionada diretamente a `IP_REPO_PATHS` no Vivado.

O IP promovido atualmente é `resnet8_resource_fifo_opt_v1_1`, com VLNV:

```text
xilinx.com:hls:resnet8_resource_fifo_opt:1.1
```

O arquivo `ip_manifest.json` registra checksums, módulos FIFO usados e a confirmação de que o top não instancia `d4096`.
