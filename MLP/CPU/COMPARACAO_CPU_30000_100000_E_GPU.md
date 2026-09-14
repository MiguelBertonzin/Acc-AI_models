# MLP Iris batch 1 — comparação CPU 30.000 × 100.000 e CPU × GPU

> **Nota de atualização:** as tabelas CPU×GPU abaixo usam as campanhas históricas individuais. A consolidação GPU atual com cinco campanhas por tamanho está em [`../GPU/replicas/RESUMO_5_CAMPANHAS.md`](../GPU/replicas/RESUMO_5_CAMPANHAS.md).


> **Atualização de energia CPU:** o RAPL foi posteriormente habilitado e validado em cinco campanhas de 30.000 e cinco de 100.000 inferências. Os valores e IC95% estão em [VALIDACAO_ENERGIA_RAPL.md](VALIDACAO_ENERGIA_RAPL.md). As marcações de energia indisponível abaixo descrevem apenas as duas execuções históricas originais.


## Resultado principal

As campanhas foram repetidas no Intel Core i7-13700 com **batch 1, execução serial e síncrona**, usando exatamente o mesmo modelo, scaler, holdout, seed e definições temporais das campanhas GPU.

| Métrica CPU | 30.000 | 100.000 | Variação |
|---|---:|---:|---:|
| Inference-only média | 0,111120 ms | 0,108883 ms | −2,01% |
| Inference-only mediana | 0,103388 ms | 0,103043 ms | −0,33% |
| Inference-only p95 | 0,153314 ms | 0,143820 ms | −6,19% |
| Inference-only p99 | 0,232181 ms | 0,184443 ms | −20,56% |
| End-to-end média efetiva | 0,135388 ms | 0,132613 ms | −2,05% |
| Vazão efetiva | 7.386,17 inf/s | 7.540,74 inf/s | +2,09% |

Nenhum outlier foi removido. As 30.000 e 100.000 inferências produziram sempre as mesmas classes da validação funcional.

## Comparação CPU × GPU

| Campanha | Janela | CPU | GPU | Diferença CPU | Speedup de vazão CPU/GPU |
|---|---|---:|---:|---:|---:|
| 30.000 | Inference-only média | 0,111120 ms | 0,185747 ms | −40,18% | — |
| 30.000 | End-to-end efetiva | 0,135388 ms | 0,280414 ms | −51,72% | 2,071× |
| 100.000 | Inference-only média | 0,108883 ms | 0,183677 ms | −40,72% | — |
| 100.000 | End-to-end efetiva | 0,132613 ms | 0,271231 ms | −51,11% | 2,045× |

Para esta rede de apenas 139 parâmetros em batch 1, o i7-13700 apresentou menor latência e aproximadamente o dobro da vazão efetiva da RTX 3050. Isso não significa que a CPU seja mais rápida para redes maiores ou execução em lote: neste ensaio, o custo de lançamento, sincronização e transferência da GPU é grande em relação às somente 120 multiplicações densas da MLP.

## Comparação estatística por passagens

O teste t de Welch foi aplicado às médias de passagens completas de 30 inferências, sem assumir variâncias iguais. A unidade do teste é a passagem, não cada chamada individual.

| Comparação | Janela | Diferença | IC95% | p bilateral | Efeito padronizado |
|---|---|---:|---:|---:|---:|
| CPU 100k − CPU 30k | Inference-only | −2,239 µs | [−3,379; −1,099] µs | 1,23×10⁻⁴ | −0,156 |
| CPU 100k − CPU 30k | End-to-end | −2,778 µs | [−4,175; −1,380] µs | 1,02×10⁻⁴ | −0,158 |
| CPU 30k − GPU 30k | Inference-only | −74,627 µs | [−78,036; −71,218] µs | 2,36×10⁻²⁴⁶ | −1,921 |
| CPU 30k − GPU 30k | End-to-end | −145,026 µs | [−148,744; −141,308] µs | <10⁻³⁰⁰ | −3,422 |
| CPU 100k − GPU 100k | Inference-only | −74,796 µs | [−76,321; −73,271] µs | <10⁻³⁰⁰ | −2,355 |
| CPU 100k − GPU 100k | End-to-end | −138,621 µs | [−140,435; −136,806] µs | <10⁻³⁰⁰ | −3,669 |

A redução CPU 30k→100k é estatisticamente detectável, mas pequena em magnitude. A diferença CPU×GPU é grande neste protocolo. Como as campanhas foram sequenciais e têm autocorrelação temporal, os testes devem ser interpretados como exploratórios, não como prova causal isolada.

## Validação funcional

- Arquitetura confirmada: entrada 4, Dense(8/ReLU), Dense(8/ReLU), Dense(3/softmax), 139 parâmetros.
- Holdout reconstruído: 120 treino e 30 teste, `test_size=0.2`, `random_state=42`, estratificado.
- Acurácia CPU: 29/30 = 96,67%.
- Concordância de classes CPU×GPU: 100%.
- Diferença máxima de probabilidade CPU×GPU: 1,192×10⁻⁷.
- Saída TensorFlow confirmada em `/CPU:0`.
- Nenhuma GPU ficou visível ao processo CPU; soft placement foi desabilitado.
- Nenhum resultado não finito ou divergência de classe foi encontrado.

## Definições temporais

### Inference-only

O cronômetro começa depois de `tf.convert_to_tensor` e termina depois de `output.numpy()`:

```text
tensor pronto -> grafo TensorFlow/oneDNN -> sincronização -> saída NumPy no host
```

Inclui execução do grafo, sincronização e materialização da saída. Não inclui a criação explícita do tensor nem o `argmax`.

### End-to-end individual

O cronômetro começa antes de criar o tensor e termina depois do `argmax`:

```text
amostra NumPy -> criação do tensor -> inferência -> saída NumPy -> argmax
```

### End-to-end efetiva e throughput

Cada passagem possui um cronômetro externo que inclui o laço Python, criação dos tensores, inferência, sincronização, `argmax` e armazenamento. A vazão é:

```text
throughput = total de inferências / soma dos tempos das passagens
```

Essa é a métrica recomendada para comparar o comportamento observado pela aplicação.

## Protocolo completo

1. Definiu `CUDA_VISIBLE_DEVICES` vazio antes da importação do TensorFlow.
2. Removeu GPUs visíveis, desabilitou soft placement e fixou modelo e função em `/CPU:0`.
3. Usou TensorFlow 2.21.0/oneDNN, `tf.function`, `float32`, batch 1, `jit_compile=False`.
4. Não usou XLA, TensorRT, mixed precision, quantização, batching ou inferências concorrentes.
5. Usou o mesmo `.h5`, `StandardScaler`, divisão e ordem de classes da GPU.
6. Executou 200 inferências de aquecimento, excluídas dos resultados.
7. Mediu 5 s de baseline depois do aquecimento.
8. Executou 1.000×30 inferências na campanha 30k.
9. Executou 3.333×30 + 10 inferências na campanha 100k.
10. Permutou deterministicamente as 30 amostras em cada passagem com seed 20260831.
11. Materializou toda saída com `.numpy()` antes da próxima chamada.
12. Não removeu outliers.
13. Calculou média, mediana, desvio-padrão amostral, CV, p90, p95, p99, mínimo e máximo.
14. Calculou IC95% usando distribuição t sobre médias das passagens completas.
15. Coletou telemetria a cada 100 ms em uma thread auxiliar.

Batch 1 significa uma amostra por chamada. O TensorFlow/oneDNN permaneceu livre para usar várias threads; não se trata de um benchmark single-core.

## Telemetria CPU

| Métrica | CPU 30.000 | CPU 100.000 |
|---|---:|---:|
| Amostras durante benchmark | 42 | 138 |
| Utilização média do processo | 110,89% | 110,87% |
| Núcleos lógicos equivalentes | 1,109 | 1,109 |
| Utilização global média | 5,98% | 5,10% |
| Frequência média entre políticas | 2.061,4 MHz | 1.764,8 MHz |
| Temperatura média/máxima do pacote | 65,7/75 °C | 71,9/76 °C |
| RSS médio | 721,3 MiB | 722,4 MiB |
| Fan médio | 933 RPM | 932 RPM |

O processo TensorFlow criou 107 threads, mas consumiu em média aproximadamente 1,109 CPUs lógicas equivalentes durante a campanha de 100.000. A frequência é a média simples de `scaling_cur_freq` das 24 políticas, não uma média ponderada somente pelos núcleos ocupados.

## Potência e energia CPU

Potência e energia CPU ficaram **indisponíveis**, não iguais a zero:

- `turbostat` não conseguiu acessar `/dev/cpu/0/msr`;
- `/sys/class/powercap` não expôs um domínio RAPL;
- `perf_event_paranoid=4` bloqueou `power/energy-pkg/`;
- `sudo -n` informou que uma senha é obrigatória.

Por isso, não é metodologicamente válido comparar os 8,193 mJ/inferência medidos na placa GPU com um valor CPU estimado. Os erros completos estão preservados nos arquivos `diagnostico_turbostat_acesso.txt` e `diagnostico_perf_energia.txt`.

## Ambiente e limitações

- CPU: Intel Core i7-13700, 16 núcleos físicos, 24 CPUs lógicas.
- Governador registrado: `powersave`, com escalonamento Intel P-state ativo.
- GPU de referência: NVIDIA GeForce RTX 3050 OEM.
- As campanhas ocorreram sequencialmente, não de forma alternada ou pareada por reinicialização.
- Temperatura CPU e temperatura GPU têm escopos físicos diferentes e não devem ser comparadas diretamente.
- Utilização do processo CPU e utilização GPU do `nvidia-smi` também têm denominadores diferentes.
- O desktop e outros processos permaneceram ativos.
- Os resultados caracterizam throughput sustentado batch 1, não throughput máximo com batching.

## Evidências

- `CPU/resultados_30000/README.md`: campanha CPU de 30.000.
- `CPU/resultados_100000/README.md`: campanha CPU de 100.000.
- `CPU/resultados_*/latencias_inference_only_cpu.npy`: latências inference-only individuais.
- `CPU/resultados_*/latencias_end_to_end_cpu.npy`: latências end-to-end individuais.
- `CPU/resultados_*/passagens_cpu.csv`: métricas por passagem.
- `CPU/resultados_*/telemetria_cpu.csv`: telemetria bruta.
- `CPU/comparacao_estatistica_passagens.json`: resultados estruturados dos testes de Welch.
- `GPU/COMPARACAO_30000_100000.md`: campanhas GPU de referência.
- `scripts/benchmark_mlp_iris_cpu.py`: script reproduzível.
