# Instruções para o ChatGPT Web — implantar e medir o XModel na ZCU104

> Entregue este arquivo ao ChatGPT Web junto com a pasta `Vitis AI/`, ou
> forneça os arquivos solicitados por ele. Este texto é o contrato técnico e
> metodológico do trabalho. O chat deve conduzir a execução interativamente,
> usando as saídas reais do terminal e sem inventar resultados.

## 1. Missão do chat

Ajude o usuário a:

1. preparar uma ZCU104 com uma imagem Linux que tenha DPU e Vitis AI Runtime
   compatíveis;
2. transferir e validar o `.xmodel` já quantizado e compilado;
3. executar inferência com VART;
4. validar as 30 amostras Iris contra as referências float e INT8;
5. criar um benchmark batch 1, serial e síncrono;
6. coletar latência, vazão, potência, energia, temperatura e telemetria;
7. executar campanhas independentes de 30.000 e 100.000 inferências;
8. produzir dados brutos, relatórios e hashes que possam ser comparados com CPU,
   GPU e com o IP hls4ml na mesma ZCU104.

Não refaça o treinamento, a quantização ou a compilação sem uma evidência
concreta de incompatibilidade. O XModel já foi gerado e validado no host.

## 2. Como o chat deve trabalhar

Siga obrigatoriamente estas regras:

- Trabalhe por fases e forneça poucos comandos por vez.
- Depois de cada bloco de comandos, espere o usuário colar a saída completa.
- Explique antes de qualquer comando que use `sudo`, altere a imagem da placa,
  instale pacotes ou sobrescreva arquivos.
- Comece sempre com comandos somente de leitura.
- Não suponha versão da imagem, endereço IP, usuário SSH, caminho de montagem,
  nomes de rails, frequência da DPU ou versão do runtime.
- Não suponha que uma imagem PYNQ genérica contenha uma DPU. O fluxo Vitis AI
  requer uma imagem/overlay com DPUCZDX8G e VART/XIR compatíveis.
- Não tente executar um XModel B4096 em uma DPU com fingerprint incompatível.
- Não altere o modelo nem o pré-processamento para “fazer funcionar”.
- Não use as 30 amostras de teste para nova calibração.
- Não remova outliers.
- Não transforme repetições em novas observações de acurácia.
- Registre comandos, versões, stdout, stderr, timestamps e hashes.
- Tudo que voltar para o repositório deve ficar dentro de `Vitis AI/`.
- Ao consultar a internet, priorize documentação oficial AMD/Xilinx, Vitis AI,
  VART, XIR e documentação da imagem efetivamente instalada.
- Se uma saída divergir do esperado, pare a fase, diagnostique e preserve a
  evidência; não avance silenciosamente.
- Não reporte medição física que não foi realmente executada.

O chat deve distinguir sempre:

- resultado já obtido no host;
- resultado observado na placa;
- estimativa;
- inferência baseada em evidência;
- item ainda pendente.

## 3. Contexto imutável do projeto

### Modelo

- Dataset: Iris, tabular, não composto por imagens.
- Entrada: quatro atributos nesta ordem:
  1. comprimento da sépala em cm;
  2. largura da sépala em cm;
  3. comprimento da pétala em cm;
  4. largura da pétala em cm.
- Arquitetura: `4 → Dense(8, ReLU) → Dense(8, ReLU) → Dense(3)`.
- Parâmetros: 139.
- Split: 120 treino e 30 teste, estratificado, `random_state=42`.
- Acurácia de referência: 29/30 = 96,67%.
- Classes: Setosa, Versicolor e Virginica, índices 0, 1 e 2.
- O StandardScaler foi ajustado apenas no treino.

### Quantização e compilação já concluídas

- Container: `xilinx/vitis-ai-tensorflow2-cpu:ubuntu2004-3.5.0.300`.
- TensorFlow do Quantizer: 2.12.0.
- Quantização: PTQ INT8, estratégia `pof2s`.
- Calibração: todas as 120 amostras de treino, 40 por classe.
- Registros de teste usados na calibração: zero.
- Alvo do Compiler: `DPUCZDX8G_ISA1_B4096`.
- Acurácia INT8 simulada: 29/30 = 96,67%.
- Concordância de classes INT8 × float: 30/30 = 100%.
- O modelo DPU produz logits. Softmax é pós-processamento opcional no ARM.

### XModel

Arquivo principal:

```text
artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel
```

SHA-256 esperado:

```text
47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce
```

Grafo compilado:

- graph: `iris_mlp_logits`;
- três subgrafos no total;
- um subgrafo USER para quantização da entrada;
- um subgrafo DPU com 13 operações;
- um subgrafo CPU para dequantização da saída;
- entrada DPU: `int8[1,4]`, `fix_point=5`;
- saída DPU: `int8[1,3]`, `fix_point=3`.

### Pacote de implantação

Use prioritariamente:

```text
artifacts/deploy/iris_mlp_zcu104_vai3_5/
```

Ele contém:

- `iris_mlp.xmodel`;
- `arch_zcu104_vai3.5.json`;
- `preprocessing.json`;
- `run_iris_mlp.py`;
- `MANIFEST.json`;
- `SHA256SUMS.txt`.

Dados adicionais necessários para validação completa:

```text
data/prepared/iris_test.npz
results/float_test_outputs.npz
results/quantized_test_outputs.npz
```

Relatórios de evidência:

```text
reports/data_preparation.json
reports/keras2_conversion.json
reports/float_validation.json
reports/quantization.json
reports/quantized_validation.json
reports/compilation.json
reports/xmodel_inspection.json
logs/compiler/compile_iris_mlp.log
```

## 4. Critérios de conclusão

O trabalho só estará concluído quando existirem evidências reais de:

- DPU identificada e compatível com `DPUCZDX8G_ISA1_B4096`;
- VART e XIR importáveis na placa;
- hashes do pacote aprovados;
- smoke test executado;
- 30/30 decisões da placa comparadas com a referência INT8;
- acurácia da placa calculada apenas sobre as 30 flores únicas;
- determinismo verificado em todas as passagens;
- cinco campanhas independentes de 30.000 inferências;
- cinco campanhas independentes de 100.000 inferências;
- latências individuais preservadas;
- passagens/ciclos preservados;
- vazão efetiva calculada pela razão dos totais;
- baseline de potência e telemetria ativa preservados;
- energia integrada no tempo e dividida pelo número exato de inferências;
- escopo físico da potência explicitamente documentado;
- temperatura e frequência registradas quando disponíveis;
- ambiente, comandos, versões, logs e hashes arquivados;
- resultados copiados de volta para uma subpasta de `Vitis AI/`.

Não considere a execução no host do modelo quantizado como validação física da
DPU.

## 5. Fase 0 — inventário do computador e da placa

Antes de transferir arquivos, peça ao usuário para identificar:

### No computador host

- diretório absoluto do projeto;
- interface de rede disponível;
- IP/hostname da ZCU104;
- método de transferência: `scp`, SFTP, pendrive ou cartão;
- hash local do XModel e do pacote;
- espaço livre para receber os resultados.

### Na ZCU104

Colete e salve, sem modificar o sistema:

- `uname -a`;
- `cat /etc/os-release`;
- arquitetura de CPU;
- memória e espaço em disco;
- versão do Python;
- módulos Python `xir`, `vart` e, se aplicável, `pynq`;
- localização de `xdputil`;
- saída integral de `xdputil query`;
- processos em segundo plano relevantes;
- governor e frequências da CPU ARM;
- clocks expostos da DPU/PL;
- sensores em `/sys/class/hwmon`;
- ferramentas `sensors`, `iio_info`, `xbutil` ou APIs PYNQ disponíveis;
- data, timezone e sincronização do relógio.

Grave o inventário em um arquivo, não apenas na tela.

## 6. Fase 1 — barreira de compatibilidade da DPU

O chat deve interpretar a saída real de `xdputil query` e conferir:

- família `DPUCZDX8G`;
- arquitetura/ISA compatível;
- configuração B4096;
- fingerprint;
- quantidade de cores DPU;
- frequência;
- versão do runtime;
- versão da imagem/overlay.

A string do `arch.json` é:

```json
{
  "target": "DPUCZDX8G_ISA1_B4096"
}
```

Se a DPU da placa não for compatível, pare. Oriente a instalar/usar uma imagem
oficial Vitis AI para ZCU104 ou um overlay DPU PYNQ comprovadamente compatível.
Não recompile para outro alvo sem autorização explícita, pois isso criaria uma
nova variante experimental.

## 7. Fase 2 — transferência e integridade

O chat deve:

1. criar uma pasta de trabalho dedicada na placa;
2. transferir o pacote de deploy;
3. transferir `iris_test.npz` e as saídas de referência;
4. executar `sha256sum -c SHA256SUMS.txt` dentro do pacote;
5. conferir separadamente o SHA-256 do XModel;
6. registrar tamanho, permissões e timestamps;
7. nunca renomear arquivos sem atualizar o manifesto.

Se o runtime da placa não possuir NumPy ou outro requisito, diagnostique a
imagem antes de instalar pacotes.

## 8. Fase 3 — smoke test

Use primeiro o executor fornecido:

```bash
python3 run_iris_mlp.py --features 5.1 3.5 1.4 0.2
```

O chat deve conferir e salvar:

- quatro atributos brutos;
- quatro valores normalizados;
- quatro inteiros INT8;
- `input_fix_point=5`;
- três inteiros INT8 de saída;
- `output_fix_point=3`;
- três logits;
- probabilidades reconstruídas;
- índice e nome da classe;
- ausência de erro do runner.

Não use a saída de um único exemplo como prova de acurácia.

## 9. Fase 4 — validação funcional completa

Ajude a criar um script separado para a placa. Ele deve:

- carregar uma única vez o XModel;
- encontrar exatamente um subgrafo DPU;
- criar um único `vart.Runner`;
- carregar `iris_test.npz`;
- usar os 30 vetores já normalizados;
- quantizar entrada com o `fix_point` lido do tensor, não hardcoded;
- saturar corretamente em INT8;
- executar batch 1;
- dequantizar a saída usando o `fix_point` lido;
- calcular logits, softmax e argmax;
- comparar cada amostra com:
  - rótulo verdadeiro;
  - predição float;
  - predição do H5 quantizado;
- salvar tabela por amostra;
- salvar matriz de confusão;
- salvar acurácia e IC95% Wilson;
- salvar erros de logits/probabilidades quando a comparação for válida;
- falhar se o número ou a ordem das amostras não coincidir;
- registrar qualquer divergência, sem arredondá-la para esconder o erro.

A expectativa é 29/30 acertos e 30/30 de concordância com o modelo INT8
simulado. Uma divergência deve ser investigada antes do benchmark.

## 10. Contrato comum de benchmark

O protocolo original em `../METODOLOGIA_BENCHMARK_MLP_IRIS.md` foi escrito para
ResNet-8/CIFAR-10. Adapte somente a terminologia:

- “imagem” passa a ser “amostra Iris”;
- “FPS” passa a ser “inferências por segundo”;
- conjunto único passa a ser o holdout estratificado de 30 flores;
- nenhuma repetição aumenta o tamanho estatístico da acurácia.

Use exatamente:

- batch 1;
- execução serial;
- execução síncrona;
- uma única inferência em voo;
- 200 inferências de aquecimento;
- baseline ocioso de 5 s depois do aquecimento;
- telemetria nominal de 100 ms;
- 30 amostras por passagem;
- ordem permutada deterministicamente com seed `20260831`;
- campanhas exatas de 30.000 e 100.000 inferências;
- cinco campanhas independentes para cada tamanho;
- nenhum outlier removido;
- nenhum print dentro da região cronometrada;
- modelo e runner carregados antes do aquecimento;
- dados carregados e normalizados antes do benchmark.

Para 30.000 inferências:

```text
1.000 passagens × 30 amostras
```

Para 100.000 inferências:

```text
3.333 passagens completas × 30 + uma passagem final × 10
```

A passagem parcial entra nas estatísticas globais, mas não no IC calculado sobre
médias de passagens completas.

## 11. Janelas de tempo obrigatórias

Registre pelo menos três níveis, com `time.perf_counter_ns()`.

### 11.1 DPU execute/wait

Começa imediatamente antes de `runner.execute_async()` e termina depois de
`runner.wait(job_id)`.

Inclui:

- despacho VART;
- execução DPU;
- sincronização.

Não inclui:

- normalização;
- quantização da entrada;
- preenchimento do buffer;
- dequantização;
- softmax;
- argmax.

Nome sugerido:

```text
latency_dpu_execute_wait_ms
```

### 11.2 Chamada completa da inferência

Começa antes de quantizar/copiar a entrada para o buffer e termina depois de:

- `execute_async`;
- `wait`;
- dequantização dos três logits;
- softmax em software;
- argmax;
- materialização da classe no Python.

A entrada já deve estar normalizada, como nos benchmarks CPU/GPU.

Nome sugerido:

```text
latency_application_end_to_end_ms
```

Esta é a principal janela para comparação prática com CPU, GPU e hls4ml.
Documente que o softmax roda no ARM neste fluxo.

### 11.3 Tempo da passagem

Começa antes da primeira amostra e termina depois da trigésima classe
materializada e armazenada. Inclui laço Python e todas as chamadas completas.

A vazão principal é:

```text
throughput_effective = total_inferences / sum(passage_duration_seconds)
```

Também reporte a média aritmética da vazão das passagens, sem confundi-la com a
razão dos totais.

## 12. Estatística obrigatória

Para cada vetor de latência, preserve todos os valores e calcule:

- contagem;
- média;
- mediana;
- desvio-padrão amostral;
- coeficiente de variação;
- mínimo;
- máximo;
- p90;
- p95;
- p99.

Para o IC95% da média:

- use as médias das passagens completas como observações;
- use distribuição t de Student;
- informe quantidade de passagens;
- não trate 100.000 inferências repetidas como independentes para acurácia;
- informe possível autocorrelação entre passagens sequenciais.

Para acurácia:

- use somente as 30 amostras únicas;
- reporte acertos/30;
- use IC95% Wilson;
- reporte F1 macro, MCC, kappa e matriz de confusão se disponíveis;
- verifique determinismo em todas as passagens.

Para comparar as cinco campanhas:

- a campanha independente é a unidade estatística;
- reporte média, mediana, desvio-padrão, CV e IC95%;
- não aplique teste pareado se as campanhas não forem realmente pareadas;
- registre temperatura inicial, ordem e tempo de resfriamento;
- trate testes de significância como exploratórios quando houver autocorrelação ou condições não controladas.

Situação atual das referências: CPU e GPU já possuem cinco campanhas independentes de 30k e cinco de 100k. Não repita os testes de host; colete cinco campanhas por tamanho na ZCU104 e trate a campanha como unidade experimental.

## 13. Potência, energia e temperatura

Primeiro descubra quais sensores a imagem expõe. Não invente nomes de rails.

Prioridade de medição:

1. wattímetro externo para consumo total da placa;
2. sensores INA226/PMBus da ZCU104 com rails documentados;
3. APIs PYNQ de rails, se realmente disponíveis;
4. estimativas de software apenas como dado auxiliar.

O chat deve pedir e interpretar:

- nomes exatos dos rails;
- unidade e escala de cada arquivo;
- se a leitura representa tensão, corrente, potência ou energia;
- frequência máxima de amostragem;
- se os rails se sobrepõem;
- quais rails cobrem PL, PS, DDR e placa;
- incerteza/resolução do sensor.

Nunca some rails sem provar que são domínios não sobrepostos.

Colete:

- timestamp monotônico;
- timestamp civil;
- potência por rail;
- tensão e corrente quando disponíveis;
- temperatura;
- frequência DPU/PL;
- frequência e utilização do ARM, quando possível.

Integre potência pelo tempo, preferencialmente por regra trapezoidal:

```text
E_total = integral de P(t) dt
E_por_inferencia = E_total / N
```

Energia dinâmica:

```text
P_dinamica(t) = max(P_ativa(t) - P_baseline, 0)
E_dinamica = integral de P_dinamica(t) dt
```

O baseline deve ser medido por 5 s:

- com a mesma imagem carregada;
- runtime e XModel já carregados;
- depois do aquecimento;
- sem inferências ativas.

A amostragem de 100 ms é muito mais lenta que uma inferência. Portanto, energia
é válida para a janela longa da campanha, não para uma inferência isolada.

Reporte separadamente, quando possível:

- placa inteira;
- domínio PL/DPU;
- domínio PS/ARM;
- DDR;
- consumo externo na tomada.

Não compare diretamente energia de rail FPGA, package RAPL e sensor de placa
GPU sem declarar os diferentes limites físicos.

## 14. Campanhas independentes

Execute cinco campanhas de 30k e cinco de 100k. Para cada uma:

1. registrar data/hora e ambiente;
2. registrar temperatura antes do aquecimento;
3. carregar overlay/runtime/XModel;
4. confirmar fingerprint da DPU;
5. realizar 200 warm-ups;
6. medir 5 s de baseline;
7. iniciar telemetria persistente;
8. iniciar a janela energética;
9. executar exatamente N inferências;
10. encerrar a janela energética;
11. aguardar pelo menos duas amostras de telemetria;
12. parar o sampler;
13. validar determinismo e contagens;
14. gravar dados em diretório novo;
15. calcular hashes;
16. permitir resfriamento ou reinicialização conforme o plano registrado.

Não sobrescreva uma campanha anterior.

## 15. Estrutura mínima dos resultados

Ao devolver dados ao repositório, use algo como:

```text
Vitis AI/results_zcu104/
├── 30000_rep1/
├── 30000_rep2/
├── 30000_rep3/
├── 30000_rep4/
├── 30000_rep5/
├── 100000_rep1/
├── 100000_rep2/
├── 100000_rep3/
├── 100000_rep4/
├── 100000_rep5/
├── campaigns.csv
├── aggregate_statistics.json
├── comparison_cpu_gpu_hls4ml.md
└── SHA256SUMS.txt
```

Cada campanha deve conter:

- `metadata.json`;
- `environment.txt`;
- `xdputil_query.txt`;
- `commands.log`;
- `stdout.log`;
- `stderr.log`;
- `validation_unique_30.json`;
- `predictions_unique_30.csv`;
- `latencies_dpu_execute_wait_ms.npy`;
- `latencies_application_end_to_end_ms.npy`;
- `passages.csv`;
- `telemetry.csv`;
- `telemetry_summary.json`;
- `benchmark_summary.json`;
- `SHA256SUMS.txt`.

Os CSVs devem informar unidades nos nomes das colunas ou em metadados.

## 16. Comparação final

A tabela final deve separar:

- acurácia;
- latência DPU execute/wait;
- latência completa batch 1;
- throughput efetivo batch 1;
- potência por escopo;
- energia por inferência por escopo;
- temperatura;
- frequência;
- versão do software;
- número de campanhas.

Não chame `latency_dpu_execute_wait` de latência end-to-end.

Não compare diretamente a latência DPU sem softmax com o núcleo hls4ml que
contém softmax. Para comparação primária, use a janela de aplicação completa,
incluindo softmax em software no fluxo Vitis AI e saída/argmax no fluxo hls4ml.

Se também medir throughput saturado com múltiplas requisições, coloque-o em uma
seção secundária. Ele não substitui o benchmark batch 1 serial.

## 17. Primeira resposta esperada do ChatGPT Web

Ao receber este documento, o chat deve:

1. resumir em poucas linhas o que já está pronto;
2. declarar que não irá requantizar/recompilar inicialmente;
3. explicar que precisa confirmar imagem, DPU e runtime;
4. fornecer somente os comandos de inventário da Fase 0;
5. pedir que o usuário cole a saída integral;
6. não avançar para transferência ou execução antes da barreira de
   compatibilidade.

## 18. Referências locais que prevalecem

Leia antes de orientar:

- `README.md` desta pasta;
- `../METODOLOGIA_BENCHMARK_MLP_IRIS.md`;
- `../CPU/README.md`;
- `../GPU/README.md`;
- `reports/xmodel_inspection.json`;
- `reports/quantized_validation.json`;
- `artifacts/deploy/iris_mlp_zcu104_vai3_5/MANIFEST.json`.

Em caso de conflito, preserve os artefatos já validados e peça decisão ao
usuário antes de criar uma nova variante experimental.

## Atualização de prontidão de 2026-09-03

A metodologia normativa é `../METODOLOGIA_BENCHMARK_MLP_IRIS.md`. CPU e GPU já possuem cinco campanhas de 30k e cinco de 100k; a GPU foi completada nesta revisão. Não solicite novas réplicas de host antes da placa. Preserve uma campanha por diretório e trate `n=5` campanhas como unidade experimental.
