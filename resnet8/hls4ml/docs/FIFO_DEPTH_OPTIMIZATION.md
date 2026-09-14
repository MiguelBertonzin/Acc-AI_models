# FIFO depth optimization

Com `IOType=io_stream`, as camadas são ligadas por FIFOs. Profundidades conservadoras podem consumir BRAM/LUTRAM sem benefício. O flow oficial do hls4ml usa FIFOs temporariamente grandes, executa síntese e RTL co-simulation, lê a ocupação máxima de cada canal e reescreve cada profundidade.

Este projeto usa o flow `vitis:fifo_depth_optimization` do hls4ml 1.3.0. Duas precauções são obrigatórias:

- o testbench chama o top pelo menos duas vezes; por isso `04_optimize_fifo.py` rejeita menos de duas amostras;
- nenhuma ocupação otimizada pode atingir a profundidade temporária de profiling. Se atingir, o profiling é considerado saturado e deve ser repetido com `--profiling-depth` maior.

O valor inicial é 4096 porque o modelo e o plano de reuse são os mesmos do fluxo anterior, cujo maior FIFO observado foi 1026. O valor oficial genérico do hls4ml é 100000; use-o se a arquitetura, scheduling, precisão ou reuse forem alterados de forma material.

Comando:

```bash
python scripts/04_optimize_fifo.py --tb-samples 2
```

Essa etapa é pesada: inclui C synthesis e RTL co-simulation de profiling, seguida por C synthesis final. Adicione `--vsynth` somente quando quiser também exportar o IP e rodar Vivado synthesis.

Referência oficial: <https://fastmachinelearning.org/hls4ml/advanced/fifo_depth.html>

