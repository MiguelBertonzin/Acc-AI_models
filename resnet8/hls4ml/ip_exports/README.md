# Pacotes distribuíveis

Esta pasta contém os ZIPs prontos para transporte. O pacote válido é:

```text
xilinx_com_hls_resnet8_resource_fifo_opt_1_1_corrected.zip
```

SHA-256:

```text
4e148c361116d19430aabd95e00b0dc3ca6789821e83602f1cd5d70313095191
```

Não use o ZIP 1.0 produzido diretamente por `export_design`: a auditoria mostrou que ele reintroduz `d4096` nos 42 FIFOs.
