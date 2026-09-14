# MLP / Iris — Vitis AI / ZCU104

## Benchmark experimental completo, validação, desempenho, potência e energia

Este diretório contém a implementação, validação e campanha experimental completa da rede neural MLP treinada sobre o dataset Iris e implantada fisicamente na placa AMD/Xilinx ZCU104 por meio do fluxo Vitis AI.

O objetivo desta campanha foi obter uma caracterização reproduzível da execução da MLP na DPU da ZCU104, medindo separadamente:

- acurácia da rede implementada;
- concordância entre referência INT8 e execução física;
- latência pura da DPU;
- latência da chamada de dispositivo;
- latência de aplicação end-to-end;
- vazão efetiva;
- comportamento sob concorrência;
- potência idle;
- potência ativa;
- potência dinâmica;
- energia total por inferência;
- energia dinâmica por inferência;
- estabilidade e repetibilidade das medições;
- comportamento das distribuições de latência;
- integridade dos artefatos e resultados.

A campanha foi dividida em dois protocolos complementares:

1. **Campanha serial de plataforma**
   - 5 campanhas de 30.000 inferências;
   - 5 campanhas de 100.000 inferências;
   - destinada principalmente à comparação futura CPU × GPU × ZCU104.

2. **Campanha robusta Vitis AI**
   - `inference_only`;
   - `end_to_end`;
   - `saturated`;
   - T1, T2, T3 e T4;
   - 10.000 inferências por configuração;
   - destinada a caracterizar concorrência, vazão e comportamento da DPU.

Ao todo, foram realizadas:

```text
5 × 30.000   = 150.000
5 × 100.000  = 500.000
12 × 10.000  = 120.000
-----------------------
TOTAL        = 770.000 inferências medidas
```

A avaliação de acurácia, entretanto, utiliza somente as **30 amostras únicas do conjunto de teste Iris**. As centenas de milhares de execuções repetidas não são tratadas como novas amostras de acurácia.

---

# 1. Rede neural

## 1.1 Dataset

Dataset utilizado:

```text
Iris
```

A entrada possui quatro características:

```text
1. sepal length
2. sepal width
3. petal length
4. petal width
```

As três classes são:

```text
Iris Setosa
Iris Versicolor
Iris Virginica
```

A divisão utilizada foi:

```text
Treino: 120 amostras
Teste : 30 amostras
```

A divisão foi estratificada e utilizou:

```text
seed = 42
```

---

# 2. Pré-processamento

Foi utilizado `StandardScaler`.

O scaler foi ajustado **somente sobre o conjunto de treinamento**, evitando vazamento de informação do conjunto de teste.

Portanto:

```text
dados de treino
    ↓
fit StandardScaler
    ↓
transform treino

dados de teste
    ↓
transform usando o scaler já ajustado
```

Na campanha de plataforma, a entrada do benchmark da ZCU104 já é considerada a representação `float32` normalizada.

Isso foi uma escolha metodológica importante para manter a mesma fronteira experimental utilizada nas implementações CPU e GPU.

---

# 3. Arquitetura da MLP

A arquitetura utilizada foi:

```text
Entrada: 4 features

Dense(8)
ReLU

Dense(8)
ReLU

Dense(3)
Logits
```

Representação resumida:

```text
4 → 8 → 8 → 3
```

Número total de parâmetros:

```text
139 parâmetros
```

A última camada fornece **logits**.

A Softmax não faz parte do hardware DPU da rede utilizada nesta campanha.

Quando necessária no cenário de aplicação, a Softmax é executada em software após a inferência da DPU.

---

# 4. Referência em alto nível

Antes da implementação física foram mantidas duas referências:

```text
modelo float
modelo INT8 simulado/quantizado
```

Resultados no conjunto de teste de 30 amostras:

```text
Float:
29 / 30 corretas
96,6667 %

INT8 de referência:
29 / 30 corretas
96,6667 %
```

A decisão de classificação entre Float e INT8 permaneceu consistente para o conjunto utilizado.

---

# 5. Quantização Vitis AI

A rede foi quantizada para:

```text
INT8
```

Foi utilizado o fluxo PTQ — Post-Training Quantization.

Características relevantes:

```text
quantização      : INT8
método           : PTQ
calibração       : conjunto de treino
amostras calib.  : 120
saída            : logits
target DPU       : B4096
```

O modelo implantado na placa é um `.xmodel`.

Arquivo utilizado:

```text
iris_mlp.xmodel
```

SHA-256:

```text
47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce
```

---

# 6. Tensores da DPU

O subgrafo DPU apresenta:

## Entrada

```text
shape      = [1, 4]
dtype      = int8
fix_point  = 5
```

A escala correspondente é:

```text
2^5 = 32
```

A quantização da entrada é equivalente a:

```python
q = round(x_normalizado * 32)
q = clip(q, -128, 127)
```

## Saída

```text
shape      = [1, 3]
dtype      = int8
fix_point  = 3
```

A dequantização é equivalente a:

```python
logits = output_int8 / 8
```

pois:

```text
2^3 = 8
```

---

# 7. Teste funcional inicial

Uma das amostras utilizadas no smoke test foi:

```text
entrada original:
[5.1, 3.5, 1.4, 0.2]
```

Após normalização:

```text
[-0.8856622,
  1.0112290,
 -1.3457223,
 -1.3232756]
```

Entrada INT8:

```text
[-28, 32, -43, -42]
```

Saída bruta INT8:

```text
[29, -30, -17]
```

Logits dequantizados:

```text
[3.625, -3.750, -2.125]
```

Probabilidades após Softmax em software:

```text
[0.996205,
 0.000624,
 0.003171]
```

Classe prevista:

```text
Iris Setosa
```

O teste confirmou o funcionamento correto da cadeia:

```text
normalização
→ quantização INT8
→ DPU
→ dequantização
→ Softmax
→ argmax
```

---

# 8. Validação física completa

A validação física foi realizada nas 30 amostras únicas do conjunto de teste.

Resultado:

```text
Amostras únicas       : 30
Corretas              : 29
Acurácia              : 96,6667 %
Referência INT8       : 29/30
Concordância DPU/INT8 : 30/30
Concordância          : 100 %
Divergências          : 0
```

Portanto, a execução física da DPU reproduziu exatamente as decisões da referência INT8.

O intervalo de Wilson de 95% para a acurácia, considerando somente as 30 amostras independentes, foi aproximadamente:

```text
83,33 % – 99,41 %
```

Isso é importante porque:

```text
30.000, 100.000 ou 770.000 execuções repetidas
≠
30.000, 100.000 ou 770.000 amostras independentes de acurácia
```

Para a acurácia:

```text
n = 30
```

As execuções repetidas foram utilizadas para medir desempenho, potência, energia e determinismo.

---

# 9. Plataforma física

Placa:

```text
AMD/Xilinx ZCU104
Zynq UltraScale+ MPSoC
```

Sistema operacional:

```text
PetaLinux 2022.2
honister
```

Kernel:

```text
5.15.36-xilinx-v2022.2
```

Arquitetura:

```text
aarch64
```

Python:

```text
Python 3.9.9
```

NumPy:

```text
1.21.2
```

---

# 10. DPU física

O comando:

```bash
xdputil query
```

confirmou:

```text
DPU Core Count = 2
```

Arquitetura dos dois cores:

```text
DPUCZDX8G_ISA1_B4096
```

Frequência:

```text
DPU Frequency = 300 MHz
XRT Frequency = 300 MHz
```

Fingerprint:

```text
0x101000056010407
```

Os dois cores físicos apresentaram o mesmo fingerprint.

DPU 0:

```text
DPUCZDX8G:DPUCZDX8G_1
```

DPU 1:

```text
DPUCZDX8G:DPUCZDX8G_2
```

Versão do IP:

```text
v4.1.0
```

Versão do VART Runner:

```text
3.0.0
```

A execução do XModel compilado funcionou corretamente neste ambiente.

---

# 11. Medição de potência

Foi utilizado o sensor INA226 disponível na ZCU104.

Caminhos principais:

```text
/sys/class/hwmon/hwmon0/name
/sys/class/hwmon/hwmon0/power1_input
/sys/class/hwmon/hwmon0/curr1_input
/sys/class/hwmon/hwmon0/in2_input
```

Identificação:

```text
ina226
```

Amostragem do subsistema IIO:

```text
sampling_frequency = 114 Hz
oversampling_ratio = 4
```

A potência medida corresponde ao:

```text
consumo da entrada da placa ZCU104
aproximadamente no rail de 12 V
```

Portanto:

> **os valores de potência não representam somente o consumo interno da DPU.**

Eles representam o consumo da placa dentro do escopo monitorado pelo INA226.

---

# 12. Definições energéticas

Foi adotado:

```text
P_dynamic = P_active - P_idle
```

Energia total:

```text
E_total = integral da potência ativa durante o benchmark
```

Energia dinâmica:

```text
E_dynamic =
E_total - P_idle × tempo_do_benchmark
```

Energia por inferência:

```text
E_total_per_inf =
E_total / número_de_inferências
```

e:

```text
E_dynamic_per_inf =
E_dynamic / número_de_inferências
```

A integração da potência foi realizada numericamente sobre a telemetria medida, utilizando integração trapezoidal.

As amostras foram explicitamente verificadas para garantir que existiam pontos antes/depois dos limites da região de benchmark (`bracketed=true`).

---

# 13. Campanha experimental A — benchmark serial de plataforma

Esta campanha foi projetada para futura comparação:

```text
CPU
GPU
ZCU104 / DPU
```

O benchmark serial utiliza:

```text
batch                       = 1
execução                    = síncrona
requisições simultâneas     = 1
warm-up                     = 200
baseline                    = 5 s
telemetria                  = 100 ms
seed                        = 20260831
outlier removal             = nenhum
```

A sequência das amostras é determinística.

A rede e o Runner são criados antes da janela medida.

---

# 14. Três fronteiras de latência da campanha serial

## 14.1 Accelerator / DPU

Representa a região mais próxima da execução pura do acelerador.

Antes do cronômetro:

```text
entrada já normalizada
→ quantização
→ cópia para o buffer persistente da DPU
```

Região medida:

```text
t0
↓
runner.execute_async()
runner.wait()
↓
t1
```

Portanto:

```text
accelerator latency
≈ execute_async + wait
```

Esta é a métrica principal para representar o custo da inferência na DPU.

## 14.2 Device call

Inclui mais operações de movimentação e tratamento da chamada ao dispositivo.

De forma conceitual:

```text
entrada normalizada
→ quantização
→ cópia entrada
→ DPU
→ cópia saída
→ dequantização
```

## 14.3 Application end-to-end

A fronteira da aplicação foi definida como:

```text
entrada float normalizada
→ quantização INT8
→ cópia para a DPU
→ DPU
→ cópia da saída
→ dequantização
→ Softmax em software
→ argmax
```

O `StandardScaler` não é recalculado nesta região.

A entrada já está normalizada porque esta é a fronteira adotada para comparação de plataforma.

---

# 15. Campanha serial — 5 × 30.000 inferências

Foram realizadas cinco campanhas.

Cada campanha possui:

```text
30.000 inferências
1.000 passagens completas × 30 amostras
```

Todas apresentaram:

```text
acurácia física = 29/30
concordância = 30/30
divergências = 0
```

## Resultados individuais — 30k

| Rep | Throughput (inf/s) | DPU mean (ms) | DPU p95 (ms) | DPU p99 (ms) | Device (ms) | E2E (ms) | P idle (W) | P active (W) | P dyn (W) | E total (mJ/inf) | E dyn (mJ/inf) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1362.497 | 0.179790 | 0.192320 | 0.224340 | 0.487597 | 0.690730 | 14.467398 | 14.678305 | 0.210906 | 10.773094 | 0.154794 |
| 02 | 1351.753 | 0.180256 | 0.193300 | 0.226792 | 0.490458 | 0.695996 | 14.470866 | 14.684357 | 0.213492 | 10.863192 | 0.157937 |
| 03 | 1361.405 | 0.179382 | 0.191311 | 0.226690 | 0.486702 | 0.691080 | 14.472687 | 14.675830 | 0.203143 | 10.779911 | 0.149215 |
| 04 | 1355.061 | 0.179440 | 0.192320 | 0.226261 | 0.489437 | 0.694473 | 14.415138 | 14.621648 | 0.206510 | 10.790395 | 0.152399 |
| 05 | 1350.259 | 0.180770 | 0.194200 | 0.229220 | 0.490556 | 0.696799 | 14.412420 | 14.626087 | 0.213667 | 10.832059 | 0.158241 |

---

# 16. Estatística entre campanhas — 5 × 30k

Nesta análise, cada uma das cinco campanhas é tratada como uma unidade experimental:

```text
n = 5 campanhas
```

O desvio-padrão desta tabela é o desvio entre campanhas e não o desvio das 150.000 latências individuais agrupadas.

| Métrica | Média | DP | CV | IC95 |
|---|---:|---:|---:|---:|
| Throughput (inf/s) | **1356.195** | 5.548 | 0.409% | 1349.307 – 1363.084 |
| DPU mean (ms) | **0.179928** | 0.000586 | 0.325% | 0.179201 – 0.180655 |
| Device mean (ms) | **0.488950** | 0.001730 | 0.354% | 0.486802 – 0.491098 |
| E2E mean (ms) | **0.693816** | 0.002788 | 0.402% | 0.690354 – 0.697278 |
| DPU p95 médio (ms) | **0.192690** | 0.001099 | 0.570% | 0.191326 – 0.194054 |
| DPU p99 médio (ms) | **0.226661** | 0.001740 | 0.768% | 0.224500 – 0.228822 |
| P idle (W) | **14.447702** | 0.031040 | 0.215% | 14.409160 – 14.486243 |
| P active (W) | **14.657245** | 0.030667 | 0.209% | 14.619167 – 14.695324 |
| P dynamic (W) | **0.209543** | 0.004599 | 2.195% | 0.203832 – 0.215254 |
| E total (mJ/inf) | **10.807730** | 0.038542 | 0.357% | 10.759874 – 10.855586 |
| E dynamic (mJ/inf) | **0.154517** | 0.003816 | 2.469% | 0.149780 – 0.159255 |

Os resultados mostram excelente repetibilidade.

O throughput apresentou CV de aproximadamente:

```text
0,41 %
```

A latência média da DPU apresentou CV entre campanhas de aproximadamente:

```text
0,33 %
```

---

# 17. Campanha serial — 5 × 100.000 inferências

Para avaliar maior duração e estabilidade, foram realizadas mais cinco campanhas com:

```text
100.000 inferências por campanha
```

Como 100.000 não é divisível exatamente por 30:

```text
3333 passagens completas × 30 = 99.990
passagem final parcial        = 10
total                         = 100.000
```

Para o cálculo de intervalo de confiança baseado em passagens dentro de cada campanha, foram utilizadas somente as 3333 passagens completas.

Novamente:

```text
29/30 corretas
30/30 concordância INT8
0 divergências
```

em todas as campanhas.

## Resultados individuais — 100k

| Rep | Throughput (inf/s) | DPU mean (ms) | DPU p95 (ms) | DPU p99 (ms) | Device (ms) | E2E (ms) | P idle (W) | P active (W) | P dyn (W) | E total (mJ/inf) | E dyn (mJ/inf) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1362.395 | 0.178222 | 0.190090 | 0.223960 | 0.486836 | 0.690843 | 14.413782 | 14.625386 | 0.211604 | 10.735056 | 0.155317 |
| 02 | 1366.695 | 0.178818 | 0.192760 | 0.227020 | 0.485453 | 0.688298 | 14.423478 | 14.628245 | 0.204767 | 10.703372 | 0.149826 |
| 03 | 1358.146 | 0.180314 | 0.193770 | 0.231660 | 0.488630 | 0.693217 | 14.426253 | 14.631477 | 0.205224 | 10.773127 | 0.151106 |
| 04 | 1359.306 | 0.180124 | 0.193590 | 0.237040 | 0.488212 | 0.692109 | 14.424548 | 14.635867 | 0.211320 | 10.767162 | 0.155461 |
| 05 | 1362.168 | 0.178522 | 0.191790 | 0.238050 | 0.486474 | 0.691093 | 14.420652 | 14.634275 | 0.213622 | 10.743370 | 0.156825 |

---

# 18. Estatística entre campanhas — 5 × 100k

| Métrica | Média | DP | CV | IC95 |
|---|---:|---:|---:|---:|
| Throughput (inf/s) | **1361.742** | 3.317 | 0.244% | 1357.623 – 1365.860 |
| DPU mean (ms) | **0.179200** | 0.000956 | 0.534% | 0.178013 – 0.180387 |
| Device mean (ms) | **0.487121** | 0.001299 | 0.267% | 0.485508 – 0.488734 |
| E2E mean (ms) | **0.691112** | 0.001831 | 0.265% | 0.688838 – 0.693386 |
| DPU p95 médio (ms) | **0.192400** | 0.001511 | 0.785% | 0.190524 – 0.194276 |
| DPU p99 médio (ms) | **0.231546** | 0.006135 | 2.649% | 0.223929 – 0.239163 |
| P idle (W) | **14.421743** | 0.004893 | 0.034% | 14.415667 – 14.427819 |
| P active (W) | **14.631050** | 0.004294 | 0.029% | 14.625719 – 14.636381 |
| P dynamic (W) | **0.209307** | 0.004038 | 1.929% | 0.204293 – 0.214322 |
| E total (mJ/inf) | **10.744417** | 0.027906 | 0.260% | 10.709767 – 10.779067 |
| E dynamic (mJ/inf) | **0.153707** | 0.003050 | 1.985% | 0.149920 – 0.157495 |

O benchmark de 100k apresentou excelente estabilidade.

Em particular:

```text
Throughput CV ≈ 0,24 %
E2E CV        ≈ 0,26 %
P idle CV     ≈ 0,034 %
```

---

# 19. Comparação 30k × 100k

Aumentar a duração da campanha não alterou substancialmente os resultados.

Diferença da média 100k em relação à média 30k:

```text
Throughput       ≈ +0,409 %
DPU mean         ≈ -0,404 %
Device mean      ≈ -0,374 %
E2E mean         ≈ -0,390 %
P idle           ≈ -0,180 %
P active         ≈ -0,179 %
P dynamic        ≈ -0,113 %
E total / inf    ≈ -0,586 %
E dynamic / inf  ≈ -0,524 %
```

Portanto, as campanhas de 30k e 100k fornecem resultados coerentes e reforçam a repetibilidade da implementação.

---

# 20. Observação sobre outliers

Nenhum outlier foi removido.

Foram preservados picos ocasionais de latência.

Exemplos observados incluem valores máximos na ordem de vários milissegundos em algumas campanhas, mesmo com medianas próximas de:

```text
~0,176–0,178 ms na DPU
```

Por esse motivo são reportadas, além da média:

```text
mediana
desvio-padrão
mínimo
máximo
p90
p95
p99
CV
IC95
```

Os p95 e p99 permaneceram muito mais estáveis que os máximos individuais e são mais úteis para caracterizar a cauda da distribuição.

---

# 21. Campanha experimental B — matriz robusta Vitis AI

A segunda campanha foi criada para caracterizar a concorrência da aplicação Vitis AI.

Foram avaliados:

```text
inference_only
end_to_end
saturated
```

com:

```text
T1
T2
T3
T4
```

Total:

```text
3 cenários × 4 configurações = 12 configurações
```

Cada configuração executou:

```text
10.000 inferências medidas
10 blocos × 1.000 inferências
```

Total da campanha:

```text
120.000 inferências
```

---

# 22. Significado de T1–T4

É fundamental não interpretar T1–T4 como quantidade de cores físicos da DPU.

A placa possui:

```text
2 cores físicos DPU
```

T1–T4 representam:

```text
T1 = 1 VART Runner / thread
T2 = 2 VART Runners / threads
T3 = 3 VART Runners / threads
T4 = 4 VART Runners / threads
```

Portanto T3 e T4 representam concorrência de software superior ao número de cores físicos disponíveis.

---

# 23. Parâmetros da campanha robusta

Para cada configuração:

```text
batch                 = 1
inferências           = 10.000
blocos                = 10
inferências/bloco     = 1.000
warm-up               = 100 por runner
baseline              = 5 s
telemetria nominal    = 10 ms
seed                   = 20260831
outliers removidos    = 0
```

Foi utilizado cooldown entre configurações para reduzir dependência térmica entre medições sucessivas.

---

# 24. Cenário `inference_only`

Objetivo:

> medir a execução da DPU com entrada INT8 preparada, evitando incluir quantização e pós-processamento na latência individual principal.

Antes do cronômetro:

```text
entrada INT8
→ cópia para input buffer
```

Região individual medida:

```text
execute_async()
+
wait()
```

Depois da região individual medida, quando necessário para verificação:

```text
cópia da saída
dequantização
argmax
```

A vazão global do cenário ainda inclui o comportamento real do loop da aplicação entre chamadas.

---

# 25. Cenário `end_to_end`

Fronteira:

```text
entrada float já normalizada
→ quantização INT8
→ cópia da entrada
→ execute_async
→ wait
→ cópia da saída
→ dequantização
→ Softmax
→ argmax
```

Neste cenário, a concorrência pode aumentar a vazão agregada, mas aumenta a latência percebida por uma inferência individual devido à espera e contenção entre runners.

---

# 26. Cenário `saturated`

Objetivo:

> determinar a maior vazão que a infraestrutura DPU consegue sustentar com requisições concorrentes e mínimo trabalho de software ao redor do acelerador.

Características:

```text
entrada INT8 persistente
sem nova quantização por inferência
sem dequantização
sem Softmax
sem argmax
```

O cenário representa uma condição de saturação do subsistema de aceleração e não uma aplicação completa.

---

# 27. Resultados completos da matriz robusta

## 27.1 Inference-only

| Config. | FPS | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | P idle (W) | P active (W) | P dyn (W) | E total (mJ/inf) | E dyn (mJ/inf) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 3462.664 | 0.176710 | 0.170290 | 0.226818 | 0.254843 | 14.4360 | 14.6507 | 0.214722 | 4.231053 | 0.062011 |
| T2 | **4894.829** | 0.271703 | 0.237120 | 0.385523 | 0.441383 | 14.4369 | 14.8138 | 0.376878 | **3.026410** | 0.076995 |
| T3 | 4231.657 | 0.377215 | 0.254440 | 0.695084 | 1.395617 | 14.4407 | 14.7621 | 0.321487 | 3.488503 | 0.075972 |
| T4 | 4060.092 | 0.600500 | 0.422950 | 1.559822 | 3.448956 | 14.4369 | 14.7875 | 0.350546 | 3.642156 | 0.086339 |

### Interpretação

A menor latência individual ocorreu em:

```text
T1
mean = 0.176710 ms
```

O maior throughput ocorreu em:

```text
T2
4894.829 inferências/s
```

Ao passar para T3 e T4, a concorrência adicional passa a provocar contenção.

T4 apresenta:

```text
mean = 0.600500 ms
p99  = 3.448956 ms
```

mostrando forte aumento da cauda de latência.

---

# 28. Resultados End-to-End

| Config. | FPS | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | P idle (W) | P active (W) | P dyn (W) | E total (mJ/inf) | E dyn (mJ/inf) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 1412.794 | **0.693515** | 0.689050 | 0.745040 | 0.827690 | 14.4364 | 14.6690 | 0.232573 | 10.382938 | 0.164619 |
| T2 | **1597.396** | 1.197868 | 1.201280 | 1.237015 | 2.912472 | 14.4362 | 14.6892 | 0.253037 | **9.195714** | **0.158406** |
| T3 | 1511.088 | 1.807040 | 1.617595 | 3.909189 | 5.700093 | 14.4345 | 14.7063 | 0.271859 | 9.732284 | 0.179910 |
| T4 | 1570.833 | 1.847936 | 1.204255 | 2.163321 | 5.623213 | 14.4315 | 14.6878 | 0.256299 | 9.350348 | 0.163161 |

### Interpretação

Menor latência:

```text
T1
0.693515 ms
```

Maior throughput:

```text
T2
1597.396 inf/s
```

Menor energia total por inferência:

```text
T2
9.195714 mJ/inf
```

Menor energia dinâmica entre essas configurações:

```text
T2
0.158406 mJ/inf
```

O aumento de concorrência acima de T2 não produz ganho monotônico.

T3, por exemplo, apresentou:

```text
1511.088 FPS
mean = 1.807040 ms
p95  = 3.909189 ms
p99  = 5.700093 ms
```

demonstrando aumento expressivo de contenção.

---

# 29. Resultados Saturated

| Config. | FPS | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | P idle (W) | P active (W) | P dyn (W) | E total (mJ/inf) | E dyn (mJ/inf) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 5484.878 | **0.169353** | 0.166660 | 0.186461 | 0.206042 | 14.4322 | 14.6610 | 0.228746 | 2.672981 | **0.041705** |
| T2 | 8917.201 | 0.207781 | 0.206970 | 0.268893 | 0.311944 | 14.4360 | 14.8257 | 0.389617 | 1.662591 | 0.043693 |
| T3 | **10470.977** | 0.264071 | 0.239340 | 0.386975 | 0.547558 | 14.4350 | 14.9041 | 0.469125 | **1.423372** | 0.044802 |
| T4 | 9639.873 | 0.365382 | 0.321700 | 0.708038 | 1.208060 | 14.4359 | 14.8774 | 0.441535 | 1.543322 | 0.045803 |

### Interpretação

Menor latência:

```text
T1
0.169353 ms
```

Maior throughput:

```text
T3
10470.977 inf/s
```

Menor energia total por inferência:

```text
T3
1.423372 mJ/inf
```

Menor energia dinâmica:

```text
T1
0.041705 mJ/inf
```

T4 não melhora a vazão:

```text
T3 = 10470.977 FPS
T4 =  9639.873 FPS
```

Portanto, para este modelo e esta plataforma, três runners produziram o maior throughput observado no cenário saturado.

Isso não significa que existem três cores DPU.

A ZCU104 continua utilizando:

```text
2 cores DPU físicos
```

T3 representa somente três runners concorrentes sendo agendados sobre a infraestrutura disponível.

---

# 30. Melhores resultados observados

## Menor latência pura

```text
Saturated T1
0.169353 ms
```

## Melhor inference-only throughput

```text
Inference-only T2
4894.829 FPS
```

## Melhor end-to-end throughput

```text
End-to-end T2
1597.396 FPS
```

## Maior throughput absoluto

```text
Saturated T3
10470.977 FPS
```

## Menor energia total no cenário saturated

```text
Saturated T3
1.423372 mJ/inf
```

## Menor energia dinâmica observada

```text
Saturated T1
0.041705 mJ/inf
```

## Melhor latência end-to-end

```text
End-to-end T1
0.693515 ms
```

---

# 31. Por que throughput e 1/latência não são necessariamente iguais?

O throughput principal foi calculado como:

```text
número exato de inferências
--------------------------
tempo global da campanha
```

ou:

```text
throughput =
N / wall_time
```

Já a latência média individual representa a duração de chamadas específicas.

Em um sistema concorrente:

```text
1 / mean_latency
```

não precisa ser igual a:

```text
throughput global
```

porque várias requisições podem estar em execução ou aguardando simultaneamente.

Por esse motivo, a vazão global é utilizada como a métrica principal de capacidade do sistema.

---

# 32. Diferença entre cenário serial e `inference_only T1`

Pode parecer estranho que a campanha serial apresente aproximadamente:

```text
~1360 FPS
```

enquanto o robusto `inference_only T1` apresenta:

```text
3462.7 FPS
```

Isso ocorre porque as fronteiras do throughput global não são iguais.

Na campanha serial, o loop global também executa operações auxiliares entre as inferências para reproduzir o protocolo CPU/GPU/ZCU104.

Já o cenário robusto `inference_only` foi construído especificamente para minimizar essas operações e estudar o comportamento do acelerador.

Por esse motivo:

> **não se deve misturar diretamente o throughput serial com o throughput saturated ou inference-only como se fossem exatamente a mesma métrica experimental.**

---

# 33. Determinismo

Nas campanhas em que a classificação foi verificada:

```text
divergências = 0
```

A saída física da DPU permaneceu consistente com a referência INT8.

No cenário `saturated`, a classificação não faz parte da janela principal de medição porque o objetivo é exclusivamente avaliar capacidade de processamento.

---

# 34. Arquivos produzidos por campanha serial

Cada diretório válido contém, entre outros:

```text
metrics_summary.json
power_summary.json
metadata.json
benchmark_summary.json

latencies_accelerator_ms.npy
latencies_device_call_ms.npy
latencies_application_e2e_ms.npy

passages.csv
telemetry.csv

sample_indices.npy
predictions.npy
divergent_positions.npy

validation_unique_30.json
validation_predictions_unique30.npy
validation_raw_output_int8_unique30.npy

preflight.json
SHA256SUMS.txt
```

Esses arquivos permitem reproduzir análises posteriormente sem necessidade de rodar novamente a placa.

---

# 35. Arquivos produzidos pela campanha robusta

Para cada configuração:

```text
inference_only_T1
inference_only_T2
...
saturated_T4
```

foram preservados:

```text
latencies_ms.npy
telemetry.csv
blocks.csv

metrics_summary.json
power_summary.json
metadata.json
config_summary.json

request_sample_indices.npy

predictions.npy
divergent_positions.npy

SHA256SUMS.txt
```

Além dos arquivos globais:

```text
preflight.json
validation_unique_30.json
selected_request_indices_10000.npy
all_configs_summary.json
integrity_summary.json
SHA256SUMS_ROOT.txt
```

O `integrity_summary.json` final confirmou:

```text
expected_configs        = 12
completed_configs       = 12
all_inferences_10000    = true
all_latency_count_10000 = true
all_positive_fps        = true
all_power_measured      = true
all_power_bracketed     = true
passed                  = true
```

---

# 36. Scripts principais

## Benchmark serial final

```text
benchmark_iris_mlp_v5.py
```

SHA-256:

```text
19c1be6f8072555cf3dd77317559982df405ac1965beb92908e974fca26b8a1e
```

Este é o script oficial utilizado nas campanhas:

```text
5 × 30k
5 × 100k
```

## Benchmark robusto final

```text
benchmark_iris_mlp_robusto.py
```

SHA-256:

```text
02ff166a1a1a21e349822f0c95edd16c48b49918b55b175a06b7f37f11294c84
```

Este script executa:

```text
inference_only T1-T4
end_to_end T1-T4
saturated T1-T4
```

## Consolidador

Versão original:

```text
summarize_iris_mlp_results.py
```

SHA-256:

```text
b6647b7d7fd6747a5647dcd0212135beb287b6d37bad8c649365d7bf4d0eb64a
```

Versão corrigida:

```text
summarize_iris_mlp_results_v2.py
```

SHA-256:

```text
91c7ea7b5e218a1616c68168baa6c319ca6b5c5db86afe8dca745cb45c6dcd9c
```

A versão `v2` corrige apenas a seleção do diretório robusto para impedir que o arquivo:

```text
robust_vitis_ai_....console.log
```

seja confundido com um diretório.

Os dados experimentais não foram alterados por essa correção.

---

# 37. Outros hashes importantes

Dataset de teste:

```text
iris_test.npz

SHA256:
3972a24323d73613dfca346b56333a5843d45e399cdf80d05ac04b1257ed86e1
```

Referência INT8:

```text
quantized_test_outputs.npz

SHA256:
b0ab40e38bd10bf685badc8552e2a66030c247447df626ab05a12e796364ba0e
```

XModel:

```text
iris_mlp.xmodel

SHA256:
47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce
```

---

# 38. Organização final dos resultados

Estrutura principal:

```text
results/
│
├── serial_official_20260908/
│   │
│   ├── 30000/
│   │   ├── rep01/
│   │   ├── rep02/
│   │   ├── rep03/
│   │   ├── rep04/
│   │   ├── rep05/
│   │   └── rep01_aborted_20260908T203231Z/
│   │
│   └── 100000/
│       ├── rep01/
│       ├── rep02/
│       ├── rep03/
│       ├── rep04/
│       └── rep05/
│
├── robust_vitis_ai_20260909T202607Z/
│   ├── inference_only_T1/
│   ├── inference_only_T2/
│   ├── inference_only_T3/
│   ├── inference_only_T4/
│   ├── end_to_end_T1/
│   ├── end_to_end_T2/
│   ├── end_to_end_T3/
│   ├── end_to_end_T4/
│   ├── saturated_T1/
│   ├── saturated_T2/
│   ├── saturated_T3/
│   └── saturated_T4/
│
├── final_summary/
│   ├── summary_all.csv
│   └── summary_all.json
│
├── FINAL_ENVIRONMENT.txt
├── FINAL_CAMPAIGN_INDEX.txt
└── FINAL_SHA256SUMS.txt
```

---

# 39. Execução abortada preservada

Durante a preparação ocorreu uma execução interrompida:

```text
rep01_aborted_20260908T203231Z
```

Essa tentativa foi preservada para rastreabilidade, porém:

```text
NÃO foi utilizada
nas médias
nos IC95
nas tabelas oficiais
ou nas conclusões experimentais
```

Ela não deve ser confundida com a `rep01` válida.

---

# 40. Resumo consolidado

O script final gerou:

```text
results/final_summary/summary_all.json
results/final_summary/summary_all.csv
```

O resumo contém:

```text
5 campanhas × 30k
agregado das 5 campanhas 30k

5 campanhas × 100k
agregado das 5 campanhas 100k

12 configurações robustas
```

Total:

```text
24 registros no summary_all
```

---

# 41. Backup final

Após o encerramento da campanha, todo o diretório da ZCU104 foi copiado para o computador Ubuntu.

Diretório de backup:

```text
results/zcu104_final_20260909T204251Z/
```

O projeto completo copiado possuía aproximadamente:

```text
23 MB
```

---

# 42. Manifesto de integridade

Foi criado:

```text
results/FINAL_SHA256SUMS.txt
```

Número de arquivos registrados:

```text
354
```

SHA-256 do manifesto:

```text
989cc92515eb0bca444aceb7c017ff6c7937a7ccf0fa660c056a7b0d8ac1dd43
```

O manifesto foi validado:

```text
na ZCU104
e
depois da transferência para o Ubuntu
```

Resultado no Ubuntu:

```text
Total OK:
354

Falhas:
Nenhuma falha
```

Portanto, os arquivos presentes no backup local são byte a byte consistentes com os arquivos registrados na placa no encerramento da campanha.

---

# 43. Arquivo compactado final

Foi criado:

```text
MLP_IRIS_ZCU104_FINAL_20260909.tar.gz
```

Tamanho aproximado:

```text
7,4 MB
```

SHA-256:

```text
3b0c491313e625892613e86e46e16dde2098ce92014b8639e2471912df4a647d
```

Também foi preservado:

```text
MLP_IRIS_ZCU104_FINAL_20260909.tar.gz.sha256
```

---

# 44. Resultado experimental principal

A implementação física apresentou:

```text
Acurácia DPU             = 96,6667 %
Concordância DPU/INT8    = 100 %
Divergências             = 0
```

No protocolo serial de maior duração:

```text
Throughput médio         = 1361,742 inf/s
DPU mean                 = 0,179200 ms
Device mean              = 0,487121 ms
Application E2E mean     = 0,691112 ms

P idle                   = 14,421743 W
P active                 = 14,631050 W
P dynamic                = 0,209307 W

E total                  = 10,744417 mJ/inf
E dynamic                = 0,153707 mJ/inf
```

No cenário de maior saturação foi atingido:

```text
Saturated T3
Throughput = 10470,977 inf/s
```

---

# 45. Conclusões da campanha

A campanha confirma que a MLP/Iris foi implantada corretamente na DPU da ZCU104.

A quantização INT8 não alterou as decisões da referência usada na implementação física:

```text
30/30 decisões concordantes
```

As campanhas seriais apresentaram alta repetibilidade, com variação muito pequena entre execuções independentes.

Aumentar a quantidade de runners pode aumentar significativamente a vazão, mas esse ganho não é monotônico.

Em `inference_only`:

```text
T1 → T2 aumenta a vazão
T3 e T4 introduzem contenção
```

Em `end_to_end`:

```text
T2 oferece o maior throughput
T1 oferece a menor latência individual
```

Em `saturated`:

```text
T3 fornece a maior vazão
T4 já apresenta degradação
```

Isso demonstra claramente o trade-off entre:

```text
latência individual
throughput agregado
concorrência
potência
energia por inferência
```

Também evidencia que o número ideal de runners não deve ser confundido com a quantidade de cores físicos DPU.

A placa possui dois cores físicos B4096, enquanto os experimentos T3/T4 utilizam mais runners concorrentes do que cores disponíveis.

---

# 46. Uso destes resultados no TCC

Para comparação entre plataformas CPU, GPU e ZCU104, a recomendação é utilizar principalmente a campanha serial, pois ela possui uma fronteira controlada e foi desenhada especificamente para essa finalidade.

Como resultado principal de desempenho da ZCU104 para essa comparação, a campanha de 100k fornece uma referência robusta:

```text
DPU latency       = 0,179200 ms
Device latency    = 0,487121 ms
End-to-end        = 0,691112 ms
Throughput        = 1361,742 inf/s
P dynamic         = 0,209307 W
E dynamic         = 0,153707 mJ/inf
```

Para análise específica do Vitis AI e capacidade da DPU, utilizar a matriz robusta T1–T4.

Não misturar diretamente:

```text
serial throughput
inference-only throughput
saturated throughput
```

sem explicar que cada um corresponde a uma fronteira experimental diferente.

---

# 47. Reprodutibilidade

Para reproduzir corretamente este trabalho, devem ser preservados em conjunto:

```text
XModel
dataset de teste
referência INT8
preprocessing
scripts de benchmark
logs
latências brutas
telemetria
resumos JSON
metadados
hashes SHA256
informações do ambiente
versão do sistema
configuração da DPU
```

Os arquivos `FINAL_ENVIRONMENT.txt`, `FINAL_CAMPAIGN_INDEX.txt` e `FINAL_SHA256SUMS.txt` foram criados justamente para permitir a reconstrução futura do contexto experimental.

---

# 48. Status final

```text
MLP / Iris
Vitis AI
ZCU104
INT8
DPUCZDX8G B4096
```

Status:

```text
Treinamento / referência      ✅
Quantização INT8              ✅
Compilação para DPU           ✅
Execução física               ✅
Validação de acurácia         ✅
Validação DPU × referência    ✅
5 × 30k                       ✅
5 × 100k                      ✅
Inference-only T1–T4          ✅
End-to-end T1–T4              ✅
Saturated T1–T4               ✅
Latências brutas              ✅
Throughput                    ✅
Potência                      ✅
Energia                       ✅
Telemetria                    ✅
Hashes                        ✅
Backup local                  ✅
Validação de 354 arquivos     ✅
Arquivo compactado final      ✅
```

**Campanha experimental Vitis AI da MLP/Iris concluída.**
