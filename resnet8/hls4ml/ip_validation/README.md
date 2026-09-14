# Validação dos IPs

Os resultados de catálogo e síntese Out-of-Context ficam separados por versão do IP.

Para `resnet8_resource_fifo_opt_v1_1` foram preservados:

- `standalone_vivado_synth.rpt`: síntese do RTL final antes do empacotamento;
- `resnet8_fifo_opt_v1_1.xci`: instância criada pelo teste de catálogo;
- `ooc_synthesis.log`: log da síntese do próprio IP;
- `ooc_utilization.rpt`: utilização da síntese OOC.

Resultado: catálogo, `validate_ip`, geração de targets e síntese OOC aprovados.
