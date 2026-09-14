# Guia para testes ARM CPU na ZCU104 com PYNQ Linux

## Contexto para outro ChatGPT

Este pacote executa pelo terminal três redes neurais na CPU ARM da ZCU104:

- MLP no conjunto Iris;
- LeNet no MNIST;
- ResNet8 no CIFAR-10.

Cada rede possui modelos TFLite FP32 e INT8. O alvo é o ARM Cortex-A53 de 64
bits (`aarch64`), batch 1, com 1 ou 4 threads. O fluxo não usa VART, DPU,
hls4ml, acelerador FPGA, delegate externo nem carrega overlay. A quantização
INT8 TFLite não corresponde ao `pof2s` do Vitis AI.

Os `.h5` são fontes preservadas no computador. Eles não precisam ser enviados
à placa nem participam da inferência ARM. Não reconverta ou substitua modelos
na ZCU104. Os resultados devem ser JSON, com warm-up fora da medição e sem
remoção de outliers.

Validação de referência feita no host:

| Rede | Modelo | Acurácia | Concordância com Keras |
|---|---:|---:|---:|
| MLP | FP32 | 96,67% | 100,00% |
| MLP | INT8 | 96,67% | 100,00% |
| LeNet | FP32 | 98,98% | 100,00% |
| LeNet | INT8 | 98,99% | 99,97% |
| ResNet8 | FP32 | 74,90% | 100,00% |
| ResNet8 | INT8 | 74,53% | 97,19% |

## Arquivos que devem ser enviados à placa

Envie o diretório completo `deploy_zcu104`. Ele tem aproximadamente 34 MB:

```text
deploy_zcu104/
├── models/tflite/                 seis modelos FP32/INT8
├── data/mlp/                      teste, calibração e scaler Iris
├── data/lenet/                    teste e calibração MNIST
├── data/resnet8/                  teste e calibração CIFAR-10
├── scripts/
│   ├── model_specs.py
│   ├── infer_arm.py
│   ├── benchmark_arm.py
│   └── check_board.py
├── manifests/conversion_manifest.json
├── results/host_validation.json
├── GUIA_CHATGPT_PYNQ_ZCU104.md
├── run_all_arm.sh
├── requirements-board.txt
├── DEPLOY_MANIFEST.json
└── SHA256SUMS.txt
```

Os dados de calibração ficam no pacote para rastreabilidade, embora não sejam
consumidos na inferência. A placa precisa de Python 3, NumPy e
`tflite_runtime`; TensorFlow completo não é necessário.

## 1. Iniciar a placa e abrir o terminal

Use uma imagem oficial PYNQ para ZCU104. As imagens estão em
<https://www.pynq.io/boards.html> e a preparação física da placa em
<https://pynq.readthedocs.io/en/v3.0.0/getting_started/zcu104_setup.html>.

Em Ethernet direta, o IP estático padrão documentado é `192.168.2.99`. Em
rede DHCP, use o endereço atribuído pelo roteador. Para acessar por SSH:

```bash
# Usuário e senha padrão da imagem clássica: xilinx
ssh xilinx@192.168.2.99
```

Também se pode abrir `http://192.168.2.99` e escolher `New -> Terminal` no
Jupyter. Esse terminal costuma ser `root`. A porta serial USB usa 115200 baud,
8 bits, sem paridade, 1 stop bit e sem flow control.

## 2. Diagnosticar antes de instalar pacotes

Execute na placa e guarde toda a saída:

```bash
uname -a
uname -m
cat /etc/os-release
python3 --version
python3 -m pip --version
ldd --version | head -1
getconf GNU_LIBC_VERSION
nproc
lscpu
python3 -c "import sys,platform; print(sys.executable); print(platform.platform()); print(platform.machine())"
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import tflite_runtime.interpreter as tflite; print('tflite_runtime OK')"
```

`uname -m` deve mostrar `aarch64`. Meça Python e glibc em vez de presumi-los,
pois eles determinam qual wheel é compatível. Se o último comando funcionar,
pule a instalação do runtime.

## 3. Instalar o runtime

O guia oficial do LiteRT informa que `tflite_runtime` é o pacote reduzido para
executar `.tflite` e oferece wheels Linux AArch64:
<https://developers.google.com/edge/litert/microcontrollers/python>.

Como usuário `xilinx`, prefira um ambiente virtual que veja os pacotes da
imagem:

```bash
python3 -m venv --system-site-packages /home/xilinx/venvs/arm_cpu
source /home/xilinx/venvs/arm_cpu/bin/activate
python3 -m pip install tflite-runtime
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import tflite_runtime.interpreter as tflite; print('runtime OK', tflite)"
```

No terminal Jupyter como `root`, use `/root/venvs/arm_cpu`. Para manter as
condições constantes, prefira toda a campanha via SSH como `xilinx`. Se
`venv` não existir:

```bash
python3 -m pip install --user tflite-runtime
```

Não reinstale NumPy automaticamente, pois isso pode afetar o PYNQ. Os scripts
aceitam `tensorflow.lite` se TensorFlow já estiver instalado.

### Instalação sem internet ou sem wheel automático

Veja as tags aceitas na placa:

```bash
python3 -m pip debug --verbose
```

No computador com internet, baixe um wheel compatível com o CPython (`cp39`,
por exemplo), `manylinux`/glibc e `aarch64`. Depois:

```bash
scp tflite_runtime-*.whl xilinx@192.168.2.99:/home/xilinx/
ssh xilinx@192.168.2.99
python3 -m pip install --user /home/xilinx/tflite_runtime-*.whl
```

Não escolha pelo exemplo. Confira `python3 --version`, `uname -m`,
`ldd --version` e `pip debug --verbose` da placa.

## 4. Transferir o pacote

No computador, na raiz do projeto:

```bash
cd "/home/miguel/Downloads/Plano testes TCC"
tar -C ARM_CPU -czf /tmp/ARM_CPU_ZCU104.tar.gz deploy_zcu104
scp /tmp/ARM_CPU_ZCU104.tar.gz xilinx@192.168.2.99:/home/xilinx/
```

Na placa:

```bash
mkdir -p /home/xilinx/ARM_CPU
tar -xzf /home/xilinx/ARM_CPU_ZCU104.tar.gz   -C /home/xilinx/ARM_CPU --strip-components=1
cd /home/xilinx/ARM_CPU
```

Alternativa direta no computador:

```bash
scp -r ARM_CPU/deploy_zcu104 xilinx@192.168.2.99:/home/xilinx/ARM_CPU
```

Use apenas um método. Os comandos seguintes presumem que o conteúdo do pacote
está diretamente em `/home/xilinx/ARM_CPU`.

## 5. Conferir integridade e ambiente

Na placa:

```bash
cd /home/xilinx/ARM_CPU
sha256sum -c SHA256SUMS.txt
mkdir -p results
python3 scripts/check_board.py | tee results/board_environment.json
```

Todos os hashes devem terminar em `OK`. Se houver `FAILED`, transfira o
pacote novamente. No diagnóstico, confirme `"machine": "aarch64"` e runtime
`tflite_runtime` ou `tensorflow.lite`.

## 6. Executar uma inferência em cada modelo

Se criou o ambiente virtual:

```bash
source /home/xilinx/venvs/arm_cpu/bin/activate
cd /home/xilinx/ARM_CPU
```

MLP/Iris, com atributos brutos na ordem comprimento/largura da sépala e
comprimento/largura da pétala:

```bash
python3 scripts/infer_arm.py --network mlp --precision fp32 --threads 1 --features 5.1 3.5 1.4 0.2
python3 scripts/infer_arm.py --network mlp --precision int8 --threads 1 --features 5.1 3.5 1.4 0.2
```

LeNet/MNIST:

```bash
python3 scripts/infer_arm.py --network lenet --precision fp32 --threads 1 --index 0
python3 scripts/infer_arm.py --network lenet --precision int8 --threads 1 --index 0
```

ResNet8/CIFAR-10:

```bash
python3 scripts/infer_arm.py --network resnet8 --precision fp32 --threads 1 --index 0
python3 scripts/infer_arm.py --network resnet8 --precision int8 --threads 1 --index 0
```

Resultados esperados: MLP classe 0 `setosa`; LeNet índice 0 classe 7; ResNet8
índice 0 classe 3 `cat`. O JSON informa classe esperada, predição, acerto,
saída desquantizada e, em INT8, a saída inteira.

## 7. Fazer benchmarks individuais

Teste curto:

```bash
mkdir -p results/board_smoke
taskset -c 0 python3 scripts/benchmark_arm.py   --network mlp --precision fp32 --threads 1   --warmup 10 --inferences 30   --output results/board_smoke/mlp_fp32_t1.json
```

Uma thread no CPU 0:

```bash
taskset -c 0 python3 scripts/benchmark_arm.py   --network lenet --precision fp32 --threads 1   --warmup 200 --inferences 10000   --output results/lenet_fp32_t1_run1.json
```

Quatro threads nos CPUs 0 a 3:

```bash
taskset -c 0-3 python3 scripts/benchmark_arm.py   --network lenet --precision int8 --threads 4   --warmup 200 --inferences 10000   --output results/lenet_int8_t4_run1.json
```

Cada JSON registra latência de `invoke()` e da aplicação completa, média,
mediana, desvio padrão, CV, p90, p95, p99, mínimo, máximo, throughput, acurácia,
afinidade, versões e hashes. Nenhum outlier é removido.

## 8. Executar automaticamente toda a matriz

O script percorre 3 redes × 2 precisões × 2 configurações de threads. Comece
com uma campanha curta:

```bash
cd /home/xilinx/ARM_CPU
chmod +x run_all_arm.sh
./run_all_arm.sh smoke
```

Uma campanha completa com 10.000 inferências por configuração:

```bash
REPETITIONS=1 INFERENCES=10000 WARMUP=200 ./run_all_arm.sh full
```

Cinco repetições independentes:

```bash
REPETITIONS=5 INFERENCES=10000 WARMUP=200 ./run_all_arm.sh full
```

Os JSONs e logs ficam em `results/zcu104_<data UTC>/`. Para MLP também podem
ser feitas campanhas de 30.000 e 100.000. Em LeNet e ResNet8, 100 ciclos
completos do teste são 1.000.000 de inferências; estime antes com 10.000.

## 9. Registrar condições da placa

Antes e depois de cada campanha:

```bash
date -u
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true
cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true
uptime
```

Veja os governors disponíveis:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
```

Use a mesma política em todas as comparações. Um overlay carregado pode afetar
a potência total da placa mesmo sem ser chamado pelos scripts; registre esse
estado ao medir energia.

## 10. Copiar os resultados de volta

No computador, na raiz do projeto:

```bash
mkdir -p ARM_CPU/results/from_zcu104
scp -r xilinx@192.168.2.99:/home/xilinx/ARM_CPU/results/* ARM_CPU/results/from_zcu104/
```

Não execute `make_deploy.py` antes de recuperar os resultados da placa.

## 11. Erros comuns

Se aparecer `No matching distribution found for tflite-runtime`, envie ao
ChatGPT:

```bash
uname -m
python3 --version
ldd --version | head -1
python3 -m pip --version
python3 -m pip debug --verbose
```

Se houver erro ao carregar um modelo ou operador:

```bash
python3 -c "import importlib.metadata as m; print(m.version('tflite-runtime'))"
for model in models/tflite/*.tflite; do sha256sum "$model"; done
```

Não instale wheels `x86_64`, `armv7l` ou de outro CPython. Não reconverta o
modelo na placa.

Se `taskset` falhar:

```bash
nproc
taskset -pc $$
python3 scripts/check_board.py
```

Se o processo for encerrado:

```bash
free -h
df -h
dmesg | tail -50
```

## 12. Mensagem pronta para outro ChatGPT

Envie este documento junto com a mensagem e as saídas reais:

```text
Estou executando os testes descritos neste guia em uma ZCU104 com Linux PYNQ,
somente pelo terminal. A inferência deve ocorrer exclusivamente nos quatro ARM
Cortex-A53, sem VART, DPU, delegate FPGA ou overlay. Não reconverta nem altere
os seis modelos ou datasets. Ajude-me passo a passo a resolver o erro ou
interpretar os resultados usando os scripts deste pacote. Antes de indicar um
wheel, confira Python, ABI, glibc e arquitetura nas saídas abaixo. Preserve
FP32/INT8, 1/4 threads, batch 1, warm-up fora da medição e JSON sem remoção de
outliers.

[cole o comando executado]
[cole toda a saída e o erro]
[cole a saída de scripts/check_board.py e dos diagnósticos da seção 2]
```

Para analisar uma campanha, envie todos os JSONs, a versão exata da imagem
PYNQ, governor, temperatura inicial/final e estado dos overlays.
