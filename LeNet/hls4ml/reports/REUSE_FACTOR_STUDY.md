# Estudo de ReuseFactor — LeNet/MNIST no hls4ml para ZCU104

Data: 2026-09-04  
Estado: análise estática concluída; nenhuma síntese ou geração de IP foi iniciada.

## Conclusão executiva

O hls4ml oferece mecanismos para orientar a escolha — configuração por camada,
lista de RFs válidos, `TargetCycles`, profiling numérico e relatórios de síntese —,
mas não fornece um otimizador automático que encontre sozinho o melhor conjunto
de RFs para uma placa e um modelo fixos.

Não é adequado escrever `ReuseFactor=64` globalmente nesta LeNet. O valor 64 é
válido somente em `dense1`. Na versão instalada, o backend escolheria os fatores
válidos mais próximos `75/75/64/60/84`; isso produz um limite estrutural de 696
multiplicadores, mas deixa `conv1` com um proxy de 46.656 ciclos, piorando muito o
balanceamento do pipeline.

Para testar a hipótese de RF máximo 64, o candidato correto é:

```text
conv1=5, conv2=50, dense1=64, dense2=60, output=42
```

Ele usa fatores válidos, tem limite estrutural estimado de 746 multiplicadores e
pior proxy de 8.400 ciclos. Isso representa 43,2% dos 1.728 DSP48E2 da ZCU104
somente se cada multiplicador lógico mapear para um DSP. Com Q22.12, a hipótese de
dois DSPs por multiplicador sobe para 86,3%, antes da infraestrutura e da lógica;
por isso ele deve ser sintetizado somente depois do candidato conservador.

O primeiro candidato recomendado para síntese passa a ser:

```text
conv1=5, conv2=150, dense1=128, dense2=120, output=84
```

Ele reduz de 444 para 380 o limite estrutural do baseline original sem aumentar o
pior proxy analítico, que permanece em 16.080 ciclos.

## O que o hls4ml realmente fornece

1. `granularity='name'`: permite configurar precisão e RF individualmente.
2. `get_valid_reuse_factors(n_in, n_out)`: enumera os RFs aceitos pela
   implementação de multiplicação do backend.
3. `set_closest_reuse_factor`: substitui um RF inválido pelo válido mais próximo.
4. `TargetCycles`: converte uma meta de ciclos em RF antes de ajustá-lo ao fator
   válido mais próximo.
5. Profiling numérico: orienta as larguras fixas, que afetam o custo real dos
   multiplicadores e a acurácia.
6. C simulation, C synthesis, RTL co-simulation e relatórios: fornecem a validação
   efetiva. O relatório `csynth` é o primeiro valor confiável de DSP/latência/II.

`TargetCycles` é uma boa semente de busca, mas não substitui a síntese. Na
implementação local 1.3.0, o backend usa seis ciclos de movimentação e deriva:

```text
proxy_ciclos = (ReuseFactor + 6) * unidades
```

Para convoluções, `unidades = altura_saida * largura_saida`; para densas,
`unidades = n_out`. Como o RF calculado é arredondado ao fator válido mais próximo,
uma meta rígida deve ser transformada em RF explícito para evitar ultrapassagem.

## Regra estrutural correta

Para `Strategy=Resource`, o limite de multiplicadores usado na validação do backend
é:

```text
limite = ceil((n_in * n_out) / min(n_in, ReuseFactor))
```

Consequência: para `RF >= n_in`, aumentar o RF não reduz mais esse limite. Por
exemplo, em `dense2` os RFs 120 e 240 mantêm limite 84; RF 240 apenas aumenta o
tempo. Isso corrigiu a estimativa anterior do candidato compacto de 189 para 236.

O limite não é uma previsão de DSP pós-síntese. Precisão, operandos constantes,
zeros, compartilhamento, implementação em LUT e mapeamento do Vitis HLS alteram o
resultado.

## RFs válidos relevantes

| Camada | `n_in × n_out` | RFs válidos até a região útil |
|---|---:|---|
| `conv1` | 25 × 6 | 1, 5, 25; acima: 50, 75, 150 |
| `conv2` | 150 × 16 | 1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150; acima: 300 |
| `dense1` | 256 × 120 | 1, 2, 4, 8, 16, 32, 64, 128, 256; acima: 512… |
| `dense2` | 120 × 84 | 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120; acima: 240… |
| `output` | 84 × 10 | 1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84; acima: 168… |

Valores acima de `n_in` são aceitos, porém dominados do ponto de vista do limite
estrutural utilizado aqui.

## Comparação dos candidatos

| Candidato | RFs C1/C2/D1/D2/Out | Limite | % de 1.728 em 1:1 | % em cenário 2:1 | Pior proxy |
|---|---|---:|---:|---:|---:|
| Teto 16 por recursos | 5/15/16/15/14 | 2.842 | 164,5% | 329,0% | 6.336 |
| Teto 32 por recursos | 25/30/32/30/28 | 1.412 | 81,7% | 163,4% | 17.856 |
| Teto 64 por recursos | 25/50/64/60/42 | 722 | 41,8% | 83,6% | 17.856 |
| RF 64 global ajustado | 75/75/64/60/84 | 696 | 40,3% | 80,6% | 46.656 |
| **Teto 64 balanceado** | **5/50/64/60/42** | **746** | **43,2%** | **86,3%** | **8.400** |
| Baseline original | 5/30/128/120/84 | 444 | 25,7% | 51,4% | 16.080 |
| **Baseline refinado** | **5/150/128/120/84** | **380** | **22,0%** | **44,0%** | **16.080** |
| Compacto corrigido | 25/150/256/240/168 | 236 | 13,7% | 27,3% | 31.440 |

Os percentuais 1:1 e 2:1 são cenários de sensibilidade, não resultados de síntese.
O DSP48E2 possui multiplicador 27 × 18; portanto Q22 × Q22 não satisfaz diretamente
o menor operando de 18 bits. Já pesos/entradas de até 18 bits tornam o cenário 1:1
mais plausível, ainda sujeito às decisões do HLS.

## Técnica recomendada de busca

1. Fixar modelo, dados, backend, clock, IO e precisão.
2. Remover RFs inválidos e soluções dominadas (`RF > n_in` sem economia estrutural).
3. Usar `TargetCycles`/proxy para localizar gargalos, mas gravar RFs explícitos.
4. Sintetizar primeiro `A_refined_same_proxy` em Q22.12.
5. Sintetizar `E_cap64_rate_balanced` com a mesma Q22.12 para isolar apenas o RF.
6. Se o RF<=64 exceder recursos, reduzir pesos/resultados para Q16.6 e manter
   acumuladores largos, após profiling e C simulation nas 10.000 imagens.
7. Comparar `DSP`, `LUT`, `FF`, `BRAM`, latência min/max, `II`, frequência estimada
   e warnings em uma tabela única.
8. Levar ao Vivado apenas os candidatos aprovados; decidir por recursos e timing
   pós-route, não apenas pelo `csynth`.
9. Medir vazão/latência/energia na placa com batch 1, warm-up fixo e repetições
   idênticas às usadas nos benchmarks CPU/GPU.

O pequeno sweep mínimo, portanto, é de três pontos: baseline refinado, RF<=64
balanceado e compacto. Não é necessário testar centenas de combinações antes de
obter os primeiros relatórios HLS.

## Arquivos reproduzíveis

- `../scripts/02_analyze_reuse_space.py`: replica as regras do backend e gera as tabelas;
- `reuse_factor_space.json`: fatores válidos, metodologia e detalhes por camada;
- `reuse_factor_candidates.csv`: comparação tabular pronta para análise;
- `../configs/reuse_refined_same_proxy.json`: primeira síntese recomendada;
- `../configs/reuse_cap64_rate_balanced.json`: hipótese RF máximo 64.

## Referências primárias

- hls4ml 1.3.0, configuração por camada e ReuseFactor:
  <https://fastmachinelearning.org/hls4ml/api/configuration.html>
- hls4ml 1.3.0, conceitos de RF, estratégias e `io_stream`:
  <https://fastmachinelearning.org/hls4ml/api/concepts.html>
- hls4ml 1.3.0, profiling numérico:
  <https://fastmachinelearning.org/hls4ml/advanced/profiling.html>
- hls4ml 1.3.0, atributos das camadas:
  <https://fastmachinelearning.org/hls4ml/ir/attributes.html>
- Código do backend FPGA (`get_valid_reuse_factors`, `TargetCycles`):
  <https://github.com/fastmachinelearning/hls4ml/blob/v1.3.0/hls4ml/backends/fpga/fpga_backend.py>
- AMD ZCU104 UG1267, 1.728 DSP slices:
  <https://docs.amd.com/api/khub/documents/g76en4BfB6qIp2gHDRkQxg/content>
- AMD UG579, DSP48E2 com multiplicador 27 × 18:
  <https://docs.amd.com/api/khub/documents/pTysoma4TYgNH95BrY1Sbw/content>
