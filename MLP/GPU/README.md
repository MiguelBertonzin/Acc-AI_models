# Benchmark da MLP Iris em GPU

A coleta GPU está completa: cinco campanhas independentes de 30.000 e cinco de 100.000 inferências, todas com batch 1 serial/síncrono, 200 warm-ups, baseline de 5 s, telemetria a 100 ms, semente 20260831 e nenhum outlier removido.

A metodologia normativa é [`../METODOLOGIA_BENCHMARK_MLP_IRIS.md`](../METODOLOGIA_BENCHMARK_MLP_IRIS.md).

## Resultado entre campanhas

| Métrica | 30.000 × 5 | 100.000 × 5 |
|---|---:|---:|
| Inference-only média | 0,176134 ms | 0,168663 ms |
| IC95% t | [0,168712; 0,183557] ms | [0,157735; 0,179591] ms |
| CV entre campanhas | 3,39% | 5,22% |
| End-to-end efetiva | 0,264565 ms | 0,255899 ms |
| Vazão efetiva | 3.784,46 inf/s | 3.911,81 inf/s |
| Energia total da placa | 7,8758 mJ/inf | 7,5211 mJ/inf |
| Acurácia | 29/30 = 96,67% | 29/30 = 96,67% |

A unidade experimental é a campanha (`n=5`), não cada inferência. As repetições medem desempenho e determinismo e não aumentam o holdout de 30 amostras.

## Arquivos

- `resultados/` e `resultados_100000/`: `rep01` histórica de cada tamanho.
- `replicas/30000/rep02..rep05/` e `replicas/100000/rep02..rep05/`: novas campanhas de 2026-09-03.
- [`replicas/RESUMO_5_CAMPANHAS.md`](replicas/RESUMO_5_CAMPANHAS.md): relatório agregado.
- `replicas/campanhas_gpu.csv`: uma linha por campanha.
- `replicas/resumo_5_campanhas.json`: estatística completa.

## Reprodução

```bash
python3 scripts/benchmark_mlp_iris_gpu.py --cycles 1000 --warmup 200 \
  --idle-seconds 5 --telemetry-ms 100 --seed 20260831 \
  --output-dir GPU/nova_replica_30000

python3 scripts/benchmark_mlp_iris_gpu_100k.py --inferences 100000 \
  --samples-per-passage 30 --warmup 200 --idle-seconds 5 \
  --telemetry-ms 100 --seed 20260831 \
  --output-dir GPU/nova_replica_100000
```

A janela inference-only começa após a criação do tensor e termina após `.numpy()`. A janela efetiva inclui o laço de aplicação/argmax por passagem. Potência é a leitura da placa GPU via `nvidia-smi`, não da tomada nem do computador inteiro; não compare diretamente com RAPL ou trilhos da ZCU104 sem rotular o escopo.
