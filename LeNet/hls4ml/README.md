# LeNet/MNIST — início do fluxo hls4ml para ZCU104

## Configuração inicial congelada

O primeiro candidato é um baseline numérico conservador e equilibrado, pensado para validar a conversão antes da síntese e do empacotamento do IP.

| Propriedade | Primeira tentativa |
|---|---|
| Modelo | `lenet_mnist_final.h5`, sem Softmax |
| Backend | `Vitis` |
| hls4ml | 1.3.0 |
| Vitis HLS/Vivado | 2024.2 |
| FPGA | `xczu7ev-ffvc1156-2-e` |
| Clock | 10 ns / 100 MHz |
| IO | `io_stream` |
| Estratégia | `Resource` |
| Convolução | `LineBuffer` |
| ParallelizationFactor | 1 |
| Precisão inicial | `ap_fixed<22,12,AP_RND_CONV,AP_SAT>` |
| Batch | 1 |
| Trace | desabilitado |
| Otimização FIFO | somente depois do baseline |

ReuseFactors:

| Camada | Multiplicações estruturais | RF | Multiplicadores paralelos estimados |
|---|---:|---:|---:|
| `conv1` | 150 | 5 | 30 |
| `conv2` | 2.400 | 30 | 80 |
| `dense1` | 30.720 | 128 | 240 |
| `dense2` | 10.080 | 120 | 84 |
| `output` | 840 | 84 | 10 |
| **Total estimado** | **44.190** | — | **444** |

Os fatores foram conferidos com `VitisBackend.get_valid_reuse_factors()`. A contagem de 444 usa o limite estrutural do backend, `ceil(n_in*n_out/min(n_in, RF))`, e não é uma previsão de DSP: o mapeamento real depende da largura fixa e deve ser obtido na síntese. Em particular, aumentar RF acima de `n_in` não reduz esse limite e apenas pode aumentar os ciclos.

## Estado atual: candidato RF<=64 validado e sintetizado

O candidato `5/50/64/60/42`, Q22.12, já passou pelas seguintes etapas:

| Etapa | Resultado |
|---|---|
| Validação C++ | 10.000 imagens; hls4ml 98,99%; Keras 98,98%; concordância top-1 99,99% |
| Vitis HLS C synthesis | concluída; latência estimada de 9.412–9.460 ciclos |
| Export do IP | concluído em `ip_repo/lenet_mnist_cap64_hls_v1_0/` |
| Vivado `synth_design` + `opt_design` | concluído sem erros |
| Utilização Vivado | LUT 61,38%; FF 17,59%; DSP 43,17%; BRAM 36,54%; URAM 0% |

O Vitis HLS havia estimado 121% de LUT, mas o Vivado pós-síntese mediu 61,38%. A comparação auditável está em `reports/cap64_q22_12_rate_balanced/HLS_VS_VIVADO_UTILIZATION.md`. Ainda falta place-and-route OOC e, depois, a implementação do block design completo.

## Estudo do ReuseFactor

O estudo reproduzível está em `reports/REUSE_FACTOR_STUDY.md`. A conclusão inicial é que não se deve aplicar `ReuseFactor=64` globalmente. O valor 64 só é válido diretamente em `dense1`; o backend ajustaria as demais camadas para outros fatores, e a primeira convolução ficaria desnecessariamente lenta.

Dois candidatos foram acrescentados. O candidato RF<=64 já foi validado, sintetizado e exportado; o refinado conservador permanece como alternativa:

| Candidato | RFs `conv1/conv2/dense1/dense2/output` | Limite estrutural | Pior proxy de ciclos |
|---|---|---:|---:|
| Refinado conservador | `5/150/128/120/84` | 380 | 16.080 |
| RF<=64 balanceado — sintetizado | `5/50/64/60/42` | 746 | 8.400 |

O primeiro é a recomendação para a primeira síntese. O segundo testa especificamente a hipótese de RF máximo 64 e troca mais recursos por uma vazão analítica melhor. O script `scripts/02_analyze_reuse_space.py` regenera o JSON e o CSV do estudo sem executar HLS.

## Motivos das escolhas

- `io_stream` e `LineBuffer` são adequados à CNN e mantêm continuidade com o fluxo ResNet/ZCU104 já validado.
- `Resource` evita o paralelismo inviável que `Latency`/RF=1 produziria nas camadas densas.
- 100 MHz é o ponto inicial já conhecido no toolchain e deixa margem para integração AXI/DMA.
- Q22.12 oferece 10 bits fracionários e grande margem para somas parciais. Nas 10.000 imagens, as saídas observadas ficaram entre −26,4215 e +26,4595, mas o acumulador pode exceder a ativação final.
- `AP_RND_CONV` e `AP_SAT` evitam truncamento enviesado e wrap-around silencioso no baseline.
- O modelo permanece sem Softmax; a decisão é `argmax(logits)`, como nos benchmarks CPU/GPU e no XModel.

## Ordem dos experimentos

1. Auditar ambiente, modelo, dados e intervalos numéricos.
2. Gerar o C++ do baseline e validar todas as 10.000 imagens.
3. Se aprovado, executar C synthesis e conferir timing/recursos.
4. Se os recursos forem altos, testar o candidato compacto com os mesmos Q22.12.
5. Depois testar precisão mista, mantendo acumuladores largos e reduzindo pesos/resultados.
6. Só explorar RF mais baixo após conhecer a margem pós-route.
7. Executar RTL co-simulation, exportar e auditar o IP.
8. Fazer síntese/place-and-route OOC no Vivado antes do block design.
9. Integrar ao AXI DMA e validar na ZCU104.

Não serão alteradas simultaneamente precisão e ReuseFactor ao comparar candidatos; isso preserva a atribuição causal das diferenças.

## Arquivos

- `configs/design_initial_zcu104.json`: invariantes do primeiro design;
- `configs/reuse_initial_balanced.json`: RFs da primeira tentativa;
- `configs/reuse_refined_same_proxy.json`: baseline refinado recomendado;
- `configs/reuse_cap64_rate_balanced.json`: candidato balanceado com RF máximo 64;
- `configs/experiments_plan.json`: candidatos e ordem de aceitação;
- `scripts/00_audit_environment_model.py`: auditoria e preparação do teste;
- `scripts/01_generate_validate_baseline.py`: geração/compilação C++ e validação completa;
- `scripts/02_analyze_reuse_space.py`: análise analítica reproduzível, sem gerar IP;
- `reports/model_profile_initial.json`: versões, ranges e RFs válidos;
- `builds/baseline_q22_12_resource_stream_balanced/`: projeto baseline preservado;
- `builds/cap64_q22_12_rate_balanced/`: C++ gerado e evidências da validação de 10.000 imagens;
- `reports/cap64_q22_12_rate_balanced/`: relatórios e logs do Vitis HLS e Vivado;
- `ip_repo/lenet_mnist_cap64_hls_v1_0/`: IP exportado para o catálogo do Vivado;
- `README_CHATGPT_WEB_LENET_ZCU104_VIVADO.md`: guia completo para integração
  PS–DMA–wrapper–LeNet no Vivado e continuidade assistida pelo ChatGPT Web.

## Comandos iniciais

```bash
cd '/home/miguel/Downloads/Plano testes TCC/LeNet'
python3 hls4ml/scripts/00_audit_environment_model.py
python3 hls4ml/scripts/01_generate_validate_baseline.py
```

A síntese HLS, o export do IP e a síntese Vivado foram executados separadamente, depois que a validação C++ cumpriu os critérios de acurácia e concordância top-1. Os artefatos permanentes estão listados acima.
