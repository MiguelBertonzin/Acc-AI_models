# MLP Iris — hls4ml/Vitis HLS

Configuração inicial do acelerador:

- placa/dispositivo: ZCU104, `xczu7ev-ffvc1156-2-e`;
- frequência alvo: 100 MHz (`ClockPeriod = 10 ns`);
- backend: Vitis;
- precisão padrão: `ap_fixed<16,6>` (`fixed<16,6>` na configuração hls4ml);
- sem QKeras;
- `ReuseFactor = 1`;
- estratégia `Latency`;
- interface `io_parallel`;
- softmax mantida no hardware, como no modelo Keras original.

A metodologia adotada na coleta física está descrita em [`../METODOLOGIA_BENCHMARK_MLP_IRIS.md`](../METODOLOGIA_BENCHMARK_MLP_IRIS.md).

O hls4ml usa tipos auxiliares `ap_fixed<18,8>` nas tabelas da softmax. Pesos,
biases, entradas, acumuladores e resultados seguem a precisão padrão
`ap_fixed<16,6>`.

## Documentação técnica

O procedimento completo de integração, validação e coleta física está descrito em [`README_MLP_IRIS_ZCU104_HLS4ML.md`](README_MLP_IRIS_ZCU104_HLS4ML.md).

## Gerar e exportar o IP

Na raiz do projeto:

```bash
bash hls4ml/scripts/run_mlp_iris_hls4ml_ip.sh
```

O script realiza conversão, compilação C++, validação das 30 amostras do
holdout, C simulation, síntese HLS e exportação para o catálogo de IP.

Para validar a conversão sem executar a síntese:

```bash
bash hls4ml/scripts/run_mlp_iris_hls4ml_ip.sh --skip-build
```

Para também executar C/RTL co-simulation:

```bash
bash hls4ml/scripts/run_mlp_iris_hls4ml_ip.sh --cosim
```

O projeto gerado fica em `hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/`.

O wrapper usa uma área temporária em `/tmp` durante o Vitis HLS porque a
ferramenta rejeita caminhos com espaços, e depois copia todos os artefatos de
volta para o projeto.

## Organização

Todos os itens específicos deste fluxo permanecem nesta pasta:

- `scripts/`: geração, validação e exportação do IP;
- `docs/`: especificação técnica do acelerador;
- `logs/`: logs e informações auxiliares das ferramentas;
- `mlp_iris_apfixed16_6_rf1_100mhz/`: projeto gerado e resultados;
- arquivos `.md` desta pasta: relatórios consolidados;
- `vivado_ooc_post_route.tcl`: implementação out-of-context no Vivado.

## Pacote golden e integridade

O diretório [`golden/`](golden/) contém os 30 vetores extraídos diretamente das transações aprovadas no RTL co-sim. Ele inclui a palavra de entrada de 64 bits já quantizada, as três saídas signed-16, classes e rótulos. Antes do benchmark na placa, é obrigatório obter 30/30 de concordância com essas classes e 29/30 de acurácia.

```bash
python3 hls4ml/scripts/generate_golden_board_package.py
(cd hls4ml/golden && sha256sum -c SHA256SUMS.txt)
python3 hls4ml/scripts/generate_flow_manifest.py
(cd hls4ml && sha256sum -c manifests/SHA256SUMS.txt)
```

O inventário [`manifests/flow_manifest.json`](manifests/flow_manifest.json) registra os arquivos do fluxo e seus hashes SHA-256. Os arquivos Tcl, bitstream, HWH, notebooks e resultados da placa permanecem nesta pasta.

## Estado da implementação

A conversão, a simulação C, a síntese HLS, a co-simulação RTL, a exportação do IP, a implementação no Vivado, a geração do bitstream e as campanhas físicas na ZCU104 foram concluídas. Os dados brutos estão em [`resultados_zcu104/`](resultados_zcu104/) e os valores consolidados constam em [`../../README_RESULTADOS_VITIS_AI_HLS4ML.md`](../../README_RESULTADOS_VITIS_AI_HLS4ML.md).
