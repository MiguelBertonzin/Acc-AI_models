# Metodologia canônica de benchmark — MLP Iris

Estado: **normativo para este projeto**. Data de congelamento inicial: 2026-09-03.

Este documento define como comparar CPU, GPU, DPU/Vitis AI e IP/HLS4ML na ZCU104. Em caso de divergência, ele prevalece sobre `metodologia-benchmark.txt`, que foi escrito para ResNet/CIFAR-10 e permanece apenas como referência histórica.

## 1. Objeto da comparação

- Rede: MLP `4 -> 8 -> 8 -> 3`, batch 1.
- Dataset: Iris; divisão estratificada fixa, `random_state=42`, com 120 amostras de treino e holdout de 30 amostras, 10 por classe.
- Ordem das entradas: comprimento e largura da sépala, comprimento e largura da pétala, em centímetros.
- Pré-processamento: `StandardScaler` ajustado **somente no treino**; `(x - mean) / scale`.
- Decisão: `argmax` das três saídas.
- Acurácia é calculada apenas nas 30 amostras únicas do holdout. Repetições nunca aumentam artificialmente o tamanho amostral da acurácia.

## 2. Parâmetros fixos

| Item | Valor |
|---|---:|
| Batch | 1 |
| Execução | serial, síncrona, uma inferência em voo |
| Aquecimento | 200 inferências, fora das métricas |
| Baseline de potência | 5 s, sem inferência |
| Telemetria | alvo de 100 ms |
| Amostras por passagem | 30, em permutação determinística |
| Semente | 20260831 |
| Campanha curta | exatamente 30.000 inferências = 1.000 passagens |
| Campanha longa | exatamente 100.000 = 3.333 passagens completas + 10 amostras |
| Réplicas independentes | 5 por plataforma e por tamanho |
| Outliers | nenhum removido |

Cada réplica deve ser executada como um novo processo. Registrar data, versões, temperatura inicial/final, configuração de clocks/governor, imagem do sistema e hashes dos artefatos. Resfriar até condição inicial comparável ou registrar explicitamente a diferença.

## 3. Janelas de tempo

Todas as plataformas devem produzir as janelas abaixo. Os limites exatos precisam constar nos metadados.

1. **Kernel/acelerador:** somente a execução síncrona do mecanismo de inferência. Na DPU: `execute_async` + `wait`; no IP: do `ap_start` ao `ap_done`, ou ciclos medidos; em CPU/GPU: chamada do modelo e sincronização da saída.
2. **Chamada de dispositivo:** inclui quantização/empacotamento, comunicação com o acelerador, espera e leitura/desempacotamento.
3. **Aplicação end-to-end:** entrada já carregada em RAM, pré-processamento, chamada de dispositivo, pós-processamento e `argmax`.
4. **Passagem:** execução consecutiva de 30 amostras, usada como unidade estatística para IC dentro de uma campanha.

O número de 5 ciclos/50 ns do HLS é latência estimada do kernel e **não** deve ser comparado diretamente com o end-to-end de VART, MMIO/PYNQ, CPU ou GPU. A comparação principal entre plataformas usa `application_end_to_end`; as demais janelas explicam onde o tempo é gasto.

Política de softmax:

- CPU/GPU: saída softmax do modelo original.
- HLS4ML: softmax faz parte do IP.
- Vitis AI: DPU entrega logits; softmax opcional no ARM apenas quando probabilidades forem necessárias.
- Para classificação, `argmax(logits) == argmax(softmax(logits))`. A janela end-to-end deve incluir o pós-processamento realmente usado.

## 4. Estatística

Para latências individuais e passagens, registrar: média, mediana, desvio-padrão amostral, CV, mínimo, máximo, p90, p95 e p99.

- Dentro de uma campanha, o IC95% da média é t de Student calculado sobre as médias das passagens completas, não sobre 30.000/100.000 observações tratadas como independentes.
- Entre plataformas, a unidade experimental é a **réplica/campanha** (`n=5`), usando uma métrica agregada por campanha.
- Não declarar superioridade estatística usando apenas uma campanha.
- Comparações feitas em imagens de sistema, fontes de alimentação ou instrumentos diferentes são independentes; não chamá-las de pareadas.
- Relatar efeito absoluto, efeito percentual, IC95% e teste usado; não reportar apenas valor-p.
- Acurácia: acertos/30 e intervalo de Wilson de 95%. Também registrar concordância de classes com a referência float.
- Determinismo: comparar as predições das repetições; divergência deve ser preservada e investigada.

## 5. Vazão

Vazão efetiva é `inferências_totais / soma_das_durações_das_passagens`. Registrar separadamente média e distribuição da vazão por passagem. Não usar o inverso da média de uma janela diferente.

## 6. Potência e energia

- Amostrar continuamente baseline e benchmark com timestamps monotônicos.
- Integrar energia pela regra trapezoidal sobre potência e tempo.
- Energia dinâmica: integral de `max(P(t) - P_baseline, 0)`; energia total: integral de `P(t)`.
- Relatar joules totais e mJ/inferência, potência média, mediana, desvio, p95, pico e cobertura temporal da telemetria.
- Nunca misturar escopos sem rótulo: RAPL package (CPU), board power do `nvidia-smi` (GPU), trilhos/PMBus da ZCU104 ou wattímetro externo.
- Para comparação energética principal, preferir o mesmo wattímetro externo no ponto de alimentação de cada plataforma. Sensores internos são resultados complementares.
- Antes da campanha FPGA, executar descoberta de sensores. Se não houver potência calibrada com unidade e escala confirmadas, marcar energia como `não mensurada`; não inferir watts de valores desconhecidos.

## 7. Regras específicas da ZCU104

### Vitis AI

- Validar o fingerprint/arquitetura DPU antes do benchmark.
- Confirmar tensores: entrada `int8[1,4]`, `fix_point=5`; saída `int8[1,3]`, `fix_point=3`.
- Quantização de entrada: `clip(round(x_norm * 2^5), -128, 127)`.
- Desquantização: `output_int8 / 2^3`.
- Parar se a validação funcional das 30 amostras não for aprovada.

### HLS4ML

- Confirmar o IP por SHA-256 e interface descrita no pacote golden.
- Entrada `ap_fixed<16,6>` com `AP_TRN/AP_WRAP`; 10 bits fracionários.
- Palavra AXI/MMIO de 64 bits: `x0[15:0]`, `x1[31:16]`, `x2[47:32]`, `x3[63:48]`.
- Saídas: três palavras de 16 bits, também `ap_fixed<16,6>`.
- Reset ativo em nível alto. Respeitar `ap_start`, `ap_ready`, `ap_done` e sinais `ap_vld`.
- A implementação completa deve registrar utilização e timing pós-route, não apenas síntese OOC do IP.

## 8. Critérios de liberação

Uma implementação só entra no benchmark se:

1. hashes do modelo, scaler e artefato de hardware forem verificados;
2. as 30 entradas forem executadas na ordem canônica;
3. a acurácia for 29/30 (96,67%) e a concordância com sua referência de fluxo for documentada;
4. não houver timeout, NaN, saturação inesperada ou erro de protocolo;
5. clock/timing estiverem atendidos e a configuração da plataforma estiver registrada;
6. a telemetria de potência tiver unidade/escala validadas, ou energia for declarada indisponível.

## 9. Estrutura mínima de saída por réplica

Cada diretório de campanha deve ser imutável depois de concluído e conter:

- `metadata.json`: ambiente, hashes, parâmetros e limites das janelas;
- `validation.json` e predições das 30 amostras;
- `latencies_*.npy` ou CSV equivalente;
- `passages.csv`;
- `telemetry.csv` e `power_summary.json`, quando disponível;
- `metrics_summary.json`;
- logs brutos e `SHA256SUMS.txt`.

Use nomes `rep01` a `rep05`, separados por plataforma e por `30000`/`100000`. Após cinco réplicas, gerar um resumo que trate campanhas como unidade experimental.

## 10. Estado de comparabilidade em 2026-09-03

- CPU: cinco campanhas independentes de 30k e cinco de 100k já disponíveis.
- GPU: uma campanha de cada tamanho estava consolidada antes desta revisão; réplicas adicionais devem ser listadas no README da GPU quando concluídas.
- Vitis AI/ZCU104: XModel e referências host estão prontos; faltam validação e campanhas na placa.
- HLS4ML/ZCU104: IP, C/RTL co-sim e OOC estão prontos; faltam block design, bitstream, validação e campanhas na placa.

Resultados anteriores continuam válidos como descritivos, mas qualquer comparação inferencial final deve obedecer a este documento.
