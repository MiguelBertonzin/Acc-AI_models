# Benchmark em GPU — ResNet-8 com softmax

Este diretório contém o benchmark batch 1 do modelo
`resnet8_cifar10_keras3.h5` no conjunto de teste oficial do CIFAR-10.

O protocolo usa subconjuntos estratificados, aninhados e determinados pela seed:
100 imagens (10 por classe), 1.000 imagens (100 por classe) e 10.000 imagens
(1.000 por classe). Para cada tamanho são executados 100 ciclos completos. Os
resultados de 10, 20, 50 e 100 ciclos são prefixos cumulativos da mesma execução,
portanto são comparações pareadas.

Execução completa:

```bash
CUDA_VISIBLE_DEVICES=0 python3 GPU/benchmark_softmax_gpu.py
```

O script exige uma GPU visível ao TensorFlow e desativa fallback de operações para
CPU. Ele faz 100 inferências de aquecimento antes da medição e salva um checkpoint
ao final de cada ciclo. Se uma execução for interrompida, basta repetir o comando
para continuar. Use `--restart` somente quando quiser descartar os checkpoints dos
tamanhos solicitados.

Os resultados ficam em `GPU/resultados_softmax_gpu/`:

- `RESULTADOS.md`: tabela pronta para leitura;
- `resultados.csv`: 12 combinações agregadas;
- `passagens_n*.csv`: métricas de cada ciclo completo;
- `latencias_n*.npy`: cada latência individual, em milissegundos;
- `metadados.json`: ambiente, seed, GPU, versões e protocolo.

A latência inclui a chamada batch 1 e a sincronização necessária para trazer a
saída ao host. O modelo é traçado uma vez com `tf.function`, sem XLA, para remover a
sobrecarga do interpretador Python sem alterar a rede. Normalização e escolha da
imagem não entram no cronômetro. A potência é a potência total da placa reportada
pelo `nvidia-smi`; a potência dinâmica subtrai a linha de base ociosa medida depois
do aquecimento.
