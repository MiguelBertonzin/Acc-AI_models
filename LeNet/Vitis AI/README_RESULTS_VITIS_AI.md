# LeNet/MNIST — Resultados Vitis AI na ZCU104

## 1. Status da campanha

**Status: FINALIZADA E APROVADA PARA COMPARAÇÃO COM AS DEMAIS REDES DO TCC.**

Esta campanha caracteriza a execução física de uma LeNet para MNIST no acelerador DPU da AMD/Xilinx ZCU104 utilizando o fluxo Vitis AI.

O experimento foi encerrado após:

- diagnóstico do sistema embarcado e do DPU;
- confirmação de integridade dos artefatos por SHA-256;
- smoke test físico do XModel;
- validação funcional nas 10.000 imagens oficiais de teste do MNIST;
- confirmação de 100% de concordância entre a decisão do DPU e a referência INT8 do host;
- execução de 12 configurações de benchmark;
- 10.000 inferências medidas por configuração;
- avaliação com 1, 2, 3 e 4 runners/threads;
- cenários `inference_only`, `end_to_end` e `saturated`;
- coleta de latência, throughput, potência, energia, tensão, corrente e temperatura;
- verificação de integridade final da campanha.

Para comparação com MLP/Iris, ResNet8/CIFAR-10 e implementações hls4ml, recomenda-se reutilizar a mesma lógica experimental sempre que aplicável, mantendo explícitas as diferenças de modelo, dataset, precisão, API e fronteira de medição.

---

# 2. Objetivo

O objetivo desta campanha foi medir, em hardware real, o comportamento de uma LeNet/MNIST compilada para o DPU da ZCU104 e obter uma base reproduzível para comparação com:

- CPU Intel Core i7-13700;
- GPU NVIDIA RTX 3050;
- outras redes implantadas via Vitis AI;
- implementações customizadas via hls4ml;
- diferentes estratégias de implantação em FPGA.

As métricas principais são:

- acurácia;
- concordância DPU × referência INT8;
- latência;
- throughput;
- escalabilidade com múltiplos runners;
- potência idle;
- potência ativa;
- potência dinâmica;
- energia total por inferência;
- energia dinâmica por inferência;
- temperatura;
- integridade e repetibilidade do fluxo.

---

# 3. Plataforma

## 3.1 FPGA / SoC

Placa:

- **AMD/Xilinx ZCU104**
- família: Zynq UltraScale+ MPSoC
- DPU: `DPUCZDX8G`
- arquitetura: `DPUCZDX8G_ISA1_B4096`
- número de cores DPU: **2**
- frequência dos cores DPU: **300 MHz**
- XRT/DPU clock observado: **300 MHz**
- DPU IP version: **v4.1.0**
- fingerprint observado: **`0x101000056010407`**

Os dois cores físicos foram reportados pelo `xdputil query`.

## 3.2 Sistema embarcado

Ambiente observado na placa:

- PetaLinux/Xilinx image: 2022.2
- kernel: `5.15.36-xilinx-v2022.2`
- arquitetura: `aarch64`
- glibc: 2.34
- Python: 3.9.9
- NumPy: 1.21.2
- VART runtime: 3.0.0
- target-factory: 3.0.0
- XIR disponível em `/usr/lib/python3.9/site-packages/xir.so`
- VART disponível em `/usr/lib/python3.9/site-packages/vart.so`

## 3.3 Compatibilidade do XModel

O XModel foi gerado no host usando Vitis AI 3.5, enquanto a imagem da ZCU104 utiliza runtime Vitis AI 3.0.

Essa diferença de versão foi mantida documentada.

Não foi assumida compatibilidade apenas pelo nome `B4096`.

A compatibilidade operacional deste artefato foi demonstrada empiricamente por:

1. desserialização do XModel na placa;
2. criação bem-sucedida do `vart.Runner`;
3. smoke test físico;
4. execução completa das 10.000 imagens;
5. concordância de 100% do argmax do DPU com a referência INT8;
6. campanha de 120.000 inferências medidas sem erro de runner.

Portanto, a afirmação válida é:

> O XModel específico utilizado nesta campanha, compilado no fluxo Vitis AI 3.5 para `DPUCZDX8G_ISA1_B4096`, foi executado com sucesso no runtime VART 3.0 desta imagem da ZCU104.

Não se generaliza essa conclusão para qualquer XModel Vitis AI 3.5.

---

# 4. Modelo

Modelo:

- LeNet
- dataset: MNIST
- entrada lógica: `28 × 28 × 1`
- layout: NHWC
- saída: 10 logits
- Softmax: **não utilizado**
- decisão: `argmax(logits)`
- parâmetros: **44.426**
- MACs teóricos por imagem: **281.640**

Arquitetura:

| Camada | Saída | Observação |
|---|---:|---|
| Input | 28×28×1 | imagem MNIST |
| Conv2D | 24×24×6 | kernel 5×5, valid, ReLU |
| MaxPool | 12×12×6 | 2×2 |
| Conv2D | 8×8×16 | kernel 5×5, valid, ReLU |
| MaxPool | 4×4×16 | 2×2 |
| Flatten | 256 | — |
| Dense | 120 | ReLU |
| Dense | 84 | ReLU |
| Dense | 10 | linear/logits |

---

# 5. Quantização e compilação

## 5.1 Ambiente do host

Fluxo de quantização/compilação:

- Vitis AI: 3.5
- container: `xilinx/vitis-ai-tensorflow2-cpu:ubuntu2004-3.5.0.300`
- TensorFlow/Keras: 2.12.0
- quantização: PTQ
- precisão: INT8
- estratégia: `pof2s`
- alvo do Compiler: `DPUCZDX8G_ISA1_B4096`

## 5.2 Calibração

Dataset de calibração:

- split oficial de treino MNIST;
- 10.000 imagens;
- sem reposição;
- exatamente 1.000 imagens por classe;
- seed: `20260825`;
- batch: 32;
- preprocessamento: `float32 / 255.0`;
- nenhuma imagem do conjunto oficial de teste usada na calibração.

## 5.3 Interface compilada do subgrafo DPU

Entrada:

- nome: `quant_input`
- shape: `[1, 28, 28, 1]`
- dtype: INT8
- fix point: 6
- escala: `2^6 = 64`

Quantização manual equivalente:

```python
image_float = image_uint8.astype(np.float32) / 255.0
input_int8 = np.clip(
    np.rint(image_float * 64),
    -128,
    127,
).astype(np.int8)
```

Saída:

- nome: `quant_output_fix`
- shape: `[1, 10]`
- dtype: INT8
- fix point: 2

Dequantização:

```python
logits = output_int8.astype(np.float32) / 4.0
prediction = np.argmax(logits)
```

Como a escala é positiva e comum aos logits, o `argmax` pode também ser realizado diretamente no vetor INT8.

---

# 6. Artefatos e hashes

## 6.1 XModel

Arquivo:

```text
lenet_mnist_no_softmax.xmodel
```

SHA-256:

```text
bfef577b297939e19cabf74b1ab8d30ab65d694cd6a1ec9b28197577a3ce52c5
```

## 6.2 Dataset de teste

Arquivo:

```text
mnist_test_uint8.npz
```

SHA-256:

```text
be1c7a4955d7da22ea2fa55f9c0350a34f33cdee1697a6da500153c2eafa73cd
```

Conteúdo:

```text
images       shape=(10000, 28, 28)  dtype=uint8
labels       shape=(10000,)         dtype=int64
test_indices shape=(10000,)         dtype=int64
```

## 6.3 Referência INT8 do host

Arquivo:

```text
quantized_test_outputs.npz
```

SHA-256:

```text
cb7f13cff28ae247001a638d7635dea8fd7066d12078dd049beb93bd856b0d32
```

Conteúdo:

```text
logits      shape=(10000, 10) dtype=float32
predictions shape=(10000,)    dtype=int64
labels      shape=(10000,)    dtype=int64
```

## 6.4 Índices usados na campanha

Seed:

```text
20260825
```

Hash dos índices N=10.000:

```text
65a5248b6e84b1b5bc6bd1068aba612e7884ac5c781f686d75d3843cc7ebd8a9
```

---

# 7. Conexão física e acesso à ZCU104

## 7.1 UART

Após a conexão USB, a interface utilizada para console serial foi:

```text
/dev/ttyUSB1
```

Comando:

```bash
picocom -b 115200 /dev/ttyUSB1
```

Prompt observado:

```text
root@xilinx-zcu104-20222:~#
```

## 7.2 Ethernet direta PC ↔ ZCU104

Foi usada rede privada direta:

```text
Ubuntu: 10.77.0.1/24
ZCU104: 10.77.0.2/24
```

Na ZCU104:

```bash
ip addr add 10.77.0.2/24 dev eth0
ip -br addr show eth0
```

No Ubuntu:

```bash
sudo ip addr replace 10.77.0.1/24 dev enp3s0
sudo ip link set enp3s0 up
sudo ip route replace 10.77.0.0/24 dev enp3s0 src 10.77.0.1 metric 10
```

Validação da rota:

```bash
ip route get 10.77.0.2
```

Resultado esperado:

```text
10.77.0.2 dev enp3s0 src 10.77.0.1
```

Teste:

```bash
ping -c 4 10.77.0.2
ssh root@10.77.0.2
```

## 7.3 SCP

A imagem embarcada não disponibilizava o `sftp-server` esperado pelo modo SCP moderno.

Por isso foi utilizado o modo SCP legado:

```bash
scp -O
```

Exemplo:

```bash
scp -O -r \
  artifacts/deploy/lenet_mnist_zcu104_vai3_5 \
  root@10.77.0.2:/home/root/
```

## 7.4 Correção do relógio

Após reboot, a placa retornava para uma data antiga.

O relógio foi sincronizado manualmente a partir do Ubuntu:

```bash
ssh root@10.77.0.2 \
  "date -u -s '$(date -u '+%Y-%m-%d %H:%M:%S')' && date -u"
```

Isso foi necessário para preservar timestamps corretos nos resultados.

---

# 8. Verificação de integridade na placa

Após transferência:

```bash
cd /home/root/lenet_mnist_zcu104_vai3_5
sha256sum -c SHA256SUMS.txt
```

Todos os arquivos retornaram `OK`.

Arquivos verificados:

- `lenet_mnist_no_softmax.xmodel`
- `mnist_test_uint8.npz`
- `quantized_test_outputs.npz`
- `00_inspect_dpu.py`
- `01_smoke_inference.py`
- `02_validate_accuracy.py`
- `arch_zcu104_vai3.5.json`
- `MANIFEST.json`

---

# 9. Diagnóstico do DPU

Comando principal:

```bash
xdputil query
```

Resultado relevante:

```text
DPU Core Count: 2
DPU Arch: DPUCZDX8G_ISA1_B4096
DPU Frequency: 300 MHz
fingerprint: 0x101000056010407
IP version: v4.1.0
VART: 3.0.0
```

Também foi utilizado:

```bash
xdputil status
```

Após as execuções, os cores retornavam ao estado `idle`.

Na validação serial inicial, o Core 0 foi utilizado enquanto o Core 1 permaneceu ocioso.

Na campanha com múltiplos runners, os dois cores apresentaram estado/endereço de execução, coerente com utilização concorrente.

---

# 10. Smoke test físico

Script:

```bash
python3 01_smoke_inference.py
```

Resultado:

```text
status: passed
sample_index: 0
label: 7
prediction_dpu: 7
prediction_quantized_host: 7
prediction_matches_reference: true
```

Interface em runtime:

```text
input_shape: [1,28,28,1]
input_fix_point: 6
output_shape: [1,10]
output_fix_point: 2
```

Saída INT8 observada:

```text
[-18, -6, 1, 1, -33, -14, -61, 44, -6, 2]
```

Predição:

```text
argmax = 7
```

O smoke foi repetido após reboot e voltou a passar.

---

# 11. Validação funcional — 10.000 imagens

Script:

```bash
python3 02_validate_accuracy.py --samples 10000
```

Resultado:

```text
samples: 10000
correct: 9895
accuracy: 98.95%
host INT8 agreement: 10000/10000
agreement: 100.0%
```

IC95 Wilson da acurácia:

```text
98.7306% a 99.1318%
```

Referência float do host:

```text
9898/10000
98.98%
```

Referência INT8 do host:

```text
9895/10000
98.95%
```

Diferença INT8 vs float:

```text
-3 acertos
-0,03 ponto percentual
```

A execução física do DPU reproduziu exatamente o argmax INT8 do host nas 10.000 imagens.

---

# 12. Benchmark definitivo

Script:

```text
03_benchmark_vart_robusto.py
```

SHA-256 da versão utilizada:

```text
a712edf15d45498236102e713f4cb918ae3b0b228503729e30256ef6cf8959d6
```

## 12.1 Matriz experimental

Foram executadas 12 configurações.

| Cenário | Threads/runners | Inferências |
|---|---:|---:|
| inference-only | 1 | 10.000 |
| inference-only | 2 | 10.000 |
| inference-only | 3 | 10.000 |
| inference-only | 4 | 10.000 |
| end-to-end | 1 | 10.000 |
| end-to-end | 2 | 10.000 |
| end-to-end | 3 | 10.000 |
| end-to-end | 4 | 10.000 |
| saturated | 1 | 10.000 |
| saturated | 2 | 10.000 |
| saturated | 3 | 10.000 |
| saturated | 4 | 10.000 |

Total:

```text
120.000 inferências medidas
```

Warm-up:

```text
100 inferências por runner
```

Os warm-ups foram excluídos das regiões medidas.

## 12.2 Divisão estatística

Cada configuração de 10.000 inferências foi dividida em:

```text
10 blocos × 1.000 inferências
```

Essa divisão permite:

- IC95 da média a partir das médias dos blocos;
- avaliação de estabilidade;
- FPS por bloco;
- análise temporal da potência.

Os blocos não aumentam o número de inferências.

## 12.3 Distribuição por thread

2 runners:

```text
5000 + 5000
```

3 runners:

```text
3334 + 3333 + 3333
```

4 runners:

```text
2500 + 2500 + 2500 + 2500
```

Cada thread utilizou:

- seu próprio `vart.Runner`;
- seu próprio input buffer;
- seu próprio output buffer.

## 12.4 Outliers

Nenhum outlier foi removido.

Todos os valores extremos permaneceram nos resultados.

---

# 13. Fronteiras de medição

## 13.1 Inference-only

Pergunta:

> Quanto custa a execução síncrona do XModel quando a entrada INT8 já está pronta?

Antes de `t0`:

- imagem selecionada;
- normalização concluída;
- quantização concluída;
- input INT8 preparado;
- buffer preenchido.

Região de latência individual:

```python
t0 = perf_counter_ns()
job = runner.execute_async([input_buffer], [output_buffer])
runner.wait(job)
t1 = perf_counter_ns()
```

Fora da latência individual:

- carga do modelo;
- carga do dataset;
- preprocessamento;
- quantização;
- cópia/preparação da próxima entrada;
- dequantização;
- argmax;
- escrita em disco.

Observação importante:

O **FPS global do bloco `inference_only`** utiliza o wall time completo do bloco. Portanto, ele inclui o laço de aplicação, cópia da entrada preparada e pós-processamento entre inferências.

Assim:

- `latency_mean_ms` do inference-only representa a fronteira VART `execute_async + wait`;
- `fps_global` representa o throughput do laço de aplicação correspondente.

## 13.2 End-to-end

Pergunta:

> Quanto custa transformar uma imagem uint8 residente em RAM na classe final?

Dentro da região medida por imagem:

```text
uint8
→ float32
→ /255
→ ×64
→ round
→ clip
→ int8
→ copiar para input buffer
→ execute_async
→ wait
→ dequantização /4
→ argmax
```

Fora:

- leitura do NPZ;
- desserialização do XModel;
- criação dos runners;
- alocações permanentes;
- warm-up;
- escrita de resultados.

## 13.3 Saturated

Objetivo:

> Medir a capacidade de execução com entrada INT8 e buffers reutilizados, sem preprocessamento ou argmax na região crítica.

Região:

```python
t0
execute_async()
wait()
t1
```

A mesma entrada INT8 é reutilizada.

O cenário é chamado de **saturated throughput** e não deve ser confundido com end-to-end.

---

# 14. Estatísticas calculadas

Para as 10.000 latências de cada configuração:

- count;
- mean;
- median;
- desvio-padrão amostral `ddof=1`;
- CV;
- p90;
- p95;
- p99;
- mínimo;
- máximo.

IC95 da média:

- observações: média de cada um dos 10 blocos;
- fórmula:

```text
mean_blocks ± 1.9599639845 × sd_blocks / sqrt(10)
```

Throughput:

```text
FPS_global = 10000 / wall_time_total
```

O FPS global não foi calculado como média simples dos FPS dos blocos.

Acurácia:

- calculada apenas sobre as 10.000 imagens únicas;
- repetições de desempenho não foram tratadas como novas amostras de acurácia.

---

# 15. Telemetria

## 15.1 Sensor de potência

Sensor:

```text
INA226
```

Interface hwmon:

```text
/sys/class/hwmon/hwmon0/
```

Arquivos:

```text
power1_input
curr1_input
in2_input
```

Interface IIO associada:

```text
/sys/bus/iio/devices/iio:device1
```

Características observadas:

```text
in_sampling_frequency = 114 Hz
in_oversampling_ratio = 4
in_voltage*_integration_time = 0.001100 s
```

Teste empírico de atualização:

```text
duração ≈ 5 s
leituras = 910
mudanças de valor = 383
intervalo mediano entre mudanças ≈ 10,983 ms
intervalo médio ≈ 13,071 ms
```

Sampler do benchmark:

```text
intervalo solicitado = 10 ms
```

## 15.2 Escopo de potência

A tensão medida fica em torno de:

```text
12,02 V
```

A corrente idle observada fica em torno de:

```text
1,20 A
```

A potência idle da placa ficou tipicamente em:

```text
≈14,49 W
```

Portanto, os valores são interpretados como potência do **rail de entrada de aproximadamente 12 V da ZCU104**.

Não devem ser chamados de “potência isolada da DPU”.

## 15.3 Temperatura

Sensor:

```text
AMS / IIO
```

Diretório:

```text
/sys/bus/iio/devices/iio:device0
```

Canais:

- PS temperature;
- PL temperature;
- remote temperature.

Temperaturas típicas durante a campanha:

```text
PS:     ~44–46 °C
PL:     ~43–45 °C
remote: ~45–47 °C
```

Não foram observadas temperaturas que indiquem condição térmica anormal.

---

# 16. Fórmulas de potência e energia

Potência dinâmica:

```text
P_dynamic = max(P_active - P_idle, 0)
```

Energia total:

```text
E_total = integral P_active(t) dt
```

Energia dinâmica:

```text
E_dynamic = max(E_total - P_idle × Δt, 0)
```

Energia total por inferência:

```text
E_total_per_inf = 1000 × E_total_J / 10000
```

Energia dinâmica por inferência:

```text
E_dynamic_per_inf = 1000 × E_dynamic_J / 10000
```

Unidade final:

```text
mJ/inferência
```

---

# 17. Resultado completo — Inference-only

| Threads | Lat. média ms | Mediana ms | SD ms | CV % | p90 ms | p95 ms | p99 ms | Min ms | Max ms | IC95 média ms | FPS global | P idle W | P ativa W | P dinâmica W | E total mJ/img | E dinâmica mJ/img | Cobertura |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,225585 | 0,216300 | 0,026103 | 11,57 | 0,262441 | 0,284173 | 0,329211 | 0,210960 | 0,466690 | 0,224978–0,226192 | 3066,89 | 14,487 | 14,767 | 0,280 | 4,8151 | 0,09140 | 99,67% |
| 2 | 0,253197 | 0,221870 | 0,067611 | 26,70 | 0,351910 | 0,391881 | 0,496332 | 0,210480 | 1,001780 | 0,251535–0,254858 | **5291,79** | 14,489 | 14,988 | 0,499 | **2,8323** | 0,09422 | 98,82% |
| 3 | 0,411653 | 0,360975 | 0,236262 | 57,39 | 0,639831 | 0,836992 | 1,324196 | 0,211140 | 3,693280 | 0,391515–0,431791 | 4487,86 | 14,491 | 14,977 | 0,486 | 3,3372 | 0,10820 | 98,85% |
| 4 | 0,612246 | 0,497340 | 0,450170 | 73,53 | 1,137437 | 1,452042 | 2,302124 | 0,211160 | 8,043640 | 0,591170–0,633322 | 4351,15 | 14,492 | 14,973 | 0,480 | 3,4411 | 0,11034 | 99,18% |

Acurácia nas quatro configurações:

```text
98,95%
```

Concordância DPU × host INT8:

```text
100,00%
```

Divergências:

```text
0
```

### Interpretação

O menor valor de latência individual ocorre com 1 runner.

O maior throughput do laço de aplicação ocorre com **2 runners**:

```text
5291,79 FPS
```

Speedup vs 1 runner:

```text
1,725×
```

Com 3 e 4 runners, a latência cresce e o throughput cai, indicando contenção acima do ponto ótimo.

---

# 18. Resultado completo — End-to-end

| Threads | Lat. média ms | Mediana ms | SD ms | CV % | p90 ms | p95 ms | p99 ms | Min ms | Max ms | IC95 média ms | FPS global | P idle W | P ativa W | P dinâmica W | E total mJ/img | E dinâmica mJ/img | Cobertura |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0,650648** | 0,588185 | 0,099138 | 15,24 | 0,808984 | 0,835760 | 0,910871 | 0,577660 | 1,635630 | 0,648524–0,652772 | 1436,19 | 14,494 | 14,748 | 0,253 | 10,2685 | **0,17624** | 99,72% |
| 2 | 1,206213 | 1,123250 | 0,388942 | 32,24 | 1,532943 | 1,682290 | 2,091478 | 0,577250 | 13,058540 | 1,198387–1,214040 | 1565,82 | 14,496 | 14,827 | 0,331 | 9,4692 | 0,21153 | 99,66% |
| 3 | 1,820691 | 1,730345 | 0,588907 | 32,35 | 2,556760 | 2,880463 | 3,592201 | 0,582790 | 8,940910 | 1,804811–1,836572 | **1570,13** | 14,498 | 14,844 | 0,346 | **9,4538** | 0,22038 | 99,57% |
| 4 | 2,472644 | 2,129700 | 1,368729 | 55,35 | 4,205438 | 5,207392 | 7,188725 | 0,578270 | 13,595510 | 2,435200–2,510087 | 1495,12 | 14,497 | 14,841 | 0,345 | 9,9265 | 0,23053 | 99,68% |

Acurácia:

```text
98,95%
```

Concordância DPU × host INT8:

```text
100,00%
```

Divergências:

```text
0
```

### Interpretação

O throughput máximo foi obtido com 3 runners:

```text
1570,13 FPS
```

Porém, 2 runners entregaram:

```text
1565,82 FPS
```

Diferença T3 vs T2:

```text
≈0,28%
```

Enquanto a latência média aumentou de:

```text
1,206 ms → 1,821 ms
```

Portanto, para operação end-to-end, **2 runners representam um ponto mais equilibrado entre latência e throughput**, mesmo que o maior FPS absoluto tenha aparecido em 3 runners.

O ganho de throughput em relação a 1 runner foi pequeno porque preprocessamento, conversões e pós-processamento passam a representar parcela importante da execução.

---

# 19. Resultado completo — Saturated

| Threads | Lat. média ms | Mediana ms | SD ms | CV % | p90 ms | p95 ms | p99 ms | Min ms | Max ms | IC95 média ms | FPS global | P idle W | P ativa W | P dinâmica W | E total mJ/img | E dinâmica mJ/img | Cobertura |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0,221690** | 0,212770 | 0,037141 | 16,75 | 0,235159 | 0,277813 | 0,342952 | 0,205980 | 2,464840 | 0,220521–0,222859 | 4167,43 | 14,494 | 14,791 | 0,297 | 3,5492 | **0,07123** | 99,17% |
| 2 | 0,235349 | 0,218680 | 0,073085 | 31,05 | 0,279454 | 0,318799 | 0,428182 | 0,208180 | 3,509670 | 0,232326–0,238372 | 7750,04 | 14,497 | 15,050 | 0,553 | 1,9420 | 0,07140 | 98,28% |
| 3 | 0,281798 | 0,242270 | 0,165393 | 58,69 | 0,376101 | 0,439513 | 0,679217 | 0,204250 | 6,315540 | 0,276647–0,286950 | **9292,01** | 14,491 | **15,184** | **0,694** | **1,6341** | 0,07468 | 98,22% |
| 4 | 0,419940 | 0,373295 | 0,209210 | 49,82 | 0,620141 | 0,745540 | 1,093435 | 0,207820 | 7,247620 | 0,408320–0,431559 | 8281,59 | 14,491 | 15,134 | 0,643 | 1,8275 | 0,07769 | 98,76% |

### Interpretação

Maior throughput de toda a campanha:

```text
9292,01 FPS
```

com:

```text
3 runners
```

Speedup vs saturated/1:

```text
2,230×
```

O ganho acima de 2× não representa a existência de três cores DPU.

A plataforma possui apenas dois cores físicos.

A explicação plausível é que múltiplos runners sobrepõem overhead de submissão/espera e mantêm os cores mais alimentados.

Com 4 runners, a contenção passa a superar esse benefício.

---

# 20. Escalabilidade

Speedup de throughput relativo à configuração de 1 runner do mesmo cenário:

| Threads | Inference-only | End-to-end | Saturated |
|---:|---:|---:|---:|
| 1 | 1,000× | 1,000× | 1,000× |
| 2 | **1,725×** | 1,090× | 1,860× |
| 3 | 1,463× | **1,093×** | **2,230×** |
| 4 | 1,419× | 1,041× | 1,987× |

---

# 21. Energia

## 21.1 Métricas coletadas

Foram coletadas diretamente ou calculadas a partir da telemetria:

- `P_idle`;
- `P_active`;
- `P_dynamic`;
- energia total da janela;
- energia dinâmica da janela;
- energia total por inferência;
- energia dinâmica por inferência;
- tensão;
- corrente;
- temperatura;
- número de amostras;
- cobertura temporal.

## 21.2 Resultado energético resumido

### Inference-only

| Threads | E total mJ/img | E dinâmica mJ/img |
|---:|---:|---:|
| 1 | 4,8151 | **0,09140** |
| 2 | **2,8323** | 0,09422 |
| 3 | 3,3372 | 0,10820 |
| 4 | 3,4411 | 0,11034 |

### End-to-end

| Threads | E total mJ/img | E dinâmica mJ/img |
|---:|---:|---:|
| 1 | 10,2685 | **0,17624** |
| 2 | 9,4692 | 0,21153 |
| 3 | **9,4538** | 0,22038 |
| 4 | 9,9265 | 0,23053 |

### Saturated

| Threads | E total mJ/img | E dinâmica mJ/img |
|---:|---:|---:|
| 1 | 3,5492 | **0,07123** |
| 2 | 1,9420 | 0,07140 |
| 3 | **1,6341** | 0,07468 |
| 4 | 1,8275 | 0,07769 |

## 21.3 Interpretação energética

A energia **total** por inferência cai quando o throughput aumenta porque o consumo idle da ZCU104 é alto em relação ao incremento de potência causado pela LeNet.

Exemplo saturated:

```text
T1: 3,5492 mJ/img
T3: 1,6341 mJ/img
```

Redução:

```text
≈53,96%
```

Já a energia **dinâmica** não apresenta a mesma redução:

```text
T1: 0,07123 mJ/img
T3: 0,07468 mJ/img
```

Isso indica que o ganho de energia total vem principalmente do melhor amortecimento do consumo estático da plataforma, e não de uma redução da energia incremental da DPU.

---

# 22. Resumo dos melhores pontos

| Objetivo | Configuração | Resultado |
|---|---|---:|
| menor latência inference-only | 1 runner | **0,225585 ms** |
| maior throughput inference-only/application loop | 2 runners | **5291,79 FPS** |
| menor energia total inference-only | 2 runners | **2,8323 mJ/img** |
| menor energia dinâmica inference-only | 1 runner | **0,09140 mJ/img** |
| menor latência end-to-end | 1 runner | **0,650648 ms** |
| maior throughput end-to-end | 3 runners | **1570,13 FPS** |
| melhor compromisso end-to-end | 2 runners | **1565,82 FPS / 1,206 ms** |
| menor energia total end-to-end | 3 runners | **9,4538 mJ/img** |
| menor energia dinâmica end-to-end | 1 runner | **0,17624 mJ/img** |
| menor latência saturated | 1 runner | **0,221690 ms** |
| maior throughput saturated | 3 runners | **9292,01 FPS** |
| menor energia total saturated | 3 runners | **1,6341 mJ/img** |
| menor energia dinâmica saturated | 1 runner | **0,07123 mJ/img** |

---

# 23. Comparação de alto nível com CPU e GPU

Resultados de referência disponíveis para esta LeNet:

## CPU

Intel Core i7-13700:

```text
TensorFlow float32
batch 1
accuracy: 98,98%
latência média: 0,3872 ms
FPS global: 2289,49
potência: 39,782 W
energia total: 17,3548 mJ/inferência
```

## GPU

NVIDIA RTX 3050 OEM:

```text
TensorFlow float32
batch 1
accuracy: 98,98%
latência média: 0,2395 ms
FPS global: 2826,73
potência: 28,2483 W
energia total: 9,9970 mJ/inferência
```

## Vitis AI / ZCU104

Vitis AI:

```text
INT8
batch 1
accuracy: 98,95%
inference-only latency T1: 0,225585 ms
application-loop FPS T2: 5291,79
saturated FPS T3: 9292,01
P ativa T1 inference-only: 14,767 W
E total T1 inference-only: 4,8151 mJ/inferência
```

### Tabela contextual

| Plataforma | Precisão | Acurácia | Latência principal | Throughput principal | Potência medida | Energia total/img |
|---|---|---:|---:|---:|---:|---:|
| Intel i7-13700 | FP32 | 98,98% | 0,3872 ms | 2289,49 FPS | 39,782 W | 17,3548 mJ |
| RTX 3050 | FP32 | 98,98% | 0,2395 ms | 2826,73 FPS | 28,248 W | 9,9970 mJ |
| ZCU104 / Vitis AI T1 | INT8 | 98,95% | **0,2256 ms** | 3066,89 FPS* | **14,767 W** | **4,8151 mJ** |
| ZCU104 / Vitis AI T2 | INT8 | 98,95% | 0,2532 ms | **5291,79 FPS*** | 14,988 W | **2,8323 mJ** |
| ZCU104 saturated T3 | INT8 | — | 0,2818 ms | **9292,01 FPS** | 15,184 W | 1,6341 mJ |

\* O FPS do cenário `inference_only` é o wall time do laço de aplicação e não a inversa direta da latência individual `execute_async + wait`.

### Ressalvas de comparação

Essa tabela deve sempre declarar:

1. CPU/GPU executam FP32; DPU executa INT8.
2. Os limites físicos de potência são diferentes:
   - CPU: RAPL package;
   - GPU: sensor da placa via `nvidia-smi`;
   - ZCU104: rail de entrada ~12 V via INA226.
3. As APIs são diferentes:
   - TensorFlow na CPU/GPU;
   - VART/DPU na ZCU104.
4. As fronteiras são comparáveis em nível de aplicação, mas não fisicamente idênticas.
5. `saturated` é um cenário específico de capacidade e não deve substituir silenciosamente o throughput end-to-end.

---

# 24. Validação de integridade final

Arquivo:

```text
validacao_integridade.json
```

Resultado:

```text
configurations_expected: 12
configurations_completed: 12
all_samples_10000: true
all_latencies_count_10000: true
all_positive_fps: true
all_power_bracketed: true
passed: true
```

Configurações de validação funcional:

```text
inference_only: 1,2,3,4 threads
end_to_end:     1,2,3,4 threads
```

Todas:

```text
correct = 9895
agreement = 10000
divergences = 0
```

Portanto:

```text
8/8 configurações funcionais:
98,95% de acurácia
100,00% de concordância
0 divergências
```

---

# 25. Estrutura de arquivos esperada

Snapshot da placa recomendado:

```text
results/
└── zcu104_physical_20260908/
    └── lenet_mnist_zcu104_vai3_5/
        ├── 00_inspect_dpu.py
        ├── 01_smoke_inference.py
        ├── 02_validate_accuracy.py
        ├── 03_benchmark_vart_robusto.py
        ├── MANIFEST.json
        ├── SHA256SUMS.txt
        ├── arch_zcu104_vai3.5.json
        ├── lenet_mnist_no_softmax.xmodel
        ├── mnist_test_uint8.npz
        ├── quantized_test_outputs.npz
        └── results/
            ├── board_environment.json
            ├── smoke_inference.json
            ├── smoke_inference_pass_20260908.json
            ├── xmodel_accuracy_latency.json
            ├── dryrun_*/
            ├── benchmark_full_20260908T191312Z.console.log
            └── benchmark_full_20260908T191312Z/
                ├── METADATA.json
                ├── FINAL_SUMMARY.csv
                ├── FINAL_SUMMARY.json
                ├── README_RESULTS.txt
                ├── validacao_integridade.json
                ├── SHA256SUMS.txt
                ├── selected_indices_n10000.npy
                ├── run_order_n10000.npy
                ├── inference_only/
                ├── end_to_end/
                └── saturated/
```

Dentro de cada configuração:

```text
threads_1/
threads_2/
threads_3/
threads_4/
```

Arquivos principais:

```text
latencies_ns.npy
thread_id_for_position.npy
blocks.csv
telemetry_idle.csv
telemetry_active.csv
summary.json
SHA256SUMS.txt
```

Nos cenários com validação de imagens:

```text
predictions_dpu.npy
raw_outputs_int8.npy
logits_dequantized.npy
labels.npy
run_order.npy
confusion_matrix.npy
divergent_positions.npy
divergent_dataset_indices.npy
```

---

# 26. Como copiar todo o snapshot da ZCU104 para o PC

No PC:

```bash
cd "$HOME/Downloads/Plano testes TCC/LeNet/Vitis AI"
```

Crie o diretório de snapshot:

```bash
mkdir -p results/zcu104_physical_20260908
```

Copie **todo o diretório utilizado na placa**:

```bash
scp -O -r \
  root@10.77.0.2:/home/root/lenet_mnist_zcu104_vai3_5 \
  results/zcu104_physical_20260908/
```

Assim são preservados:

- XModel;
- dataset usado;
- referência INT8;
- scripts;
- manifests;
- checksums;
- smoke;
- validação;
- dry-run;
- campanha completa;
- telemetria;
- matrizes NPY;
- CSVs;
- JSONs;
- logs.

Depois:

```bash
find results/zcu104_physical_20260908/lenet_mnist_zcu104_vai3_5 \
  -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > results/zcu104_physical_20260908/LOCAL_SNAPSHOT_SHA256SUMS.txt
```

Opcionalmente:

```bash
du -sh results/zcu104_physical_20260908
find results/zcu104_physical_20260908 -type f | wc -l
```

---

# 27. Preservação dos resultados

Não sobrescrever:

```text
results/benchmark_full_20260908T191312Z
```

Essa pasta representa a campanha oficial utilizada para análise.

Se novos testes forem realizados futuramente, usar outro diretório timestampado.

Exemplo:

```text
benchmark_full_YYYYMMDDTHHMMSSZ
```

---

# 28. Decisão sobre testes adicionais

Para o objetivo atual do TCC, esta campanha pode ser considerada finalizada.

Não é obrigatório executar uma nova campanha da LeNet antes de avançar para as outras redes, desde que:

1. os arquivos brutos sejam preservados;
2. a metodologia seja registrada;
3. as diferenças de fronteira sejam explicitadas;
4. o mesmo princípio experimental seja adotado nas demais redes;
5. os resultados instrumentados de potência/energia sejam tratados como parte da metodologia oficial.

Uma campanha adicional sem telemetria poderia ser utilizada como análise complementar do overhead de instrumentação, mas não é necessária para considerar esta LeNet/Vitis AI concluída.

Caso tal experimento complementar seja realizado no futuro, ele deve ser identificado como uma campanha separada e não substituir silenciosamente estes resultados.

---

# 29. Principais conclusões

1. O XModel INT8 foi executado com sucesso no DPU físico da ZCU104.
2. A acurácia física foi **98,95%**.
3. A concordância DPU × referência INT8 foi **100%**.
4. Nenhuma divergência de classe ocorreu em 10.000 imagens.
5. A latência inference-only com um runner foi **0,2256 ms**.
6. Dois runners maximizaram o throughput do laço inference-only em **5291,79 FPS**.
7. Três runners maximizaram o saturated throughput em **9292,01 FPS**.
8. Quatro runners não melhoraram desempenho e aumentaram latência.
9. No end-to-end, 2 e 3 runners apresentaram throughput praticamente equivalente.
10. O custo de preprocessamento/pós-processamento limita a escalabilidade end-to-end.
11. A potência idle da ZCU104 ficou próxima de **14,5 W**.
12. O maior throughput saturated exigiu cerca de **15,18 W** de potência total do rail.
13. O menor valor de energia total observado foi **1,6341 mJ/inferência** em saturated/3.
14. O menor valor de energia dinâmica observado foi **0,07123 mJ/inferência** em saturated/1.
15. O ganho de energia total com concorrência vem principalmente da amortização do consumo estático da plataforma.
16. Os sensores INA226/AMS forneceram cobertura temporal suficiente para a campanha.
17. As 12 configurações passaram pela verificação estrutural final.
18. A LeNet/Vitis AI está pronta para ser usada como referência na comparação com as próximas redes e com o fluxo hls4ml.

---

# 30. Resultado recomendado para citações rápidas no TCC

Se for necessário resumir a LeNet/Vitis AI em poucas linhas:

> A LeNet quantizada em INT8 foi implantada na DPU `DPUCZDX8G_ISA1_B4096` da ZCU104 e validada sobre as 10.000 imagens oficiais de teste do MNIST, atingindo 98,95% de acurácia e 100% de concordância de argmax com a referência INT8 do host. Com batch 1, a latência inference-only média usando um runner foi de 0,2256 ms. O maior throughput do laço inference-only foi 5291,79 FPS com dois runners, enquanto o cenário saturado atingiu 9292,01 FPS com três runners. A potência total da placa foi medida no rail de aproximadamente 12 V via INA226, com baseline idle próximo de 14,5 W. No cenário saturado de maior throughput, foram observados 15,184 W de potência ativa, 1,6341 mJ de energia total por inferência e 0,07468 mJ de energia dinâmica por inferência.

---

# 31. Arquivos centrais para auditoria

Prioridade máxima de preservação:

```text
README_RESULTS_VITIS_AI.md

results/zcu104_physical_20260908/
└── lenet_mnist_zcu104_vai3_5/
    ├── lenet_mnist_no_softmax.xmodel
    ├── mnist_test_uint8.npz
    ├── quantized_test_outputs.npz
    ├── 03_benchmark_vart_robusto.py
    └── results/
        ├── board_environment.json
        ├── xmodel_accuracy_latency.json
        └── benchmark_full_20260908T191312Z/
            ├── METADATA.json
            ├── FINAL_SUMMARY.csv
            ├── FINAL_SUMMARY.json
            ├── validacao_integridade.json
            ├── SHA256SUMS.txt
            └── <dados brutos de cada cenário>
```

---

**Projeto:** TCC — comparação de fluxos de implantação de redes neurais em FPGA  
**Rede:** LeNet  
**Dataset:** MNIST  
**Fluxo:** Vitis AI  
**Placa:** AMD/Xilinx ZCU104  
**DPU:** 2× DPUCZDX8G B4096 @ 300 MHz  
**Precisão:** INT8  
**Campanha física:** 08/09/2026  
**Status:** FINALIZADA
