# Critérios de curadoria

Este repositório é uma cópia organizada do material relevante para preservar e reproduzir os experimentos do TCC. O diretório de trabalho original não foi modificado.

## Conteúdo incluído

- código-fonte, scripts shell, Python, Tcl, C/C++, VHDL e Verilog relevantes;
- modelos treinados, scalers e modelos transformados/compilados;
- dados coletados nos benchmarks, inclusive latências, predições, consumo e telemetria;
- configurações, manifests, hashes e metadados de ambiente;
- relatórios de HLS/Vivado e validações numéricas;
- IPs exportados, bitstreams, handoffs `.hwh` e checkpoints pós-síntese úteis;
- documentação e instruções de reprodução existentes.

## Conteúdo não copiado

Somente da **cópia destinada ao GitHub**, foram omitidos:

- `__pycache__`, arquivos `.pyc`, `.Xil` e bibliotecas `.so` locais;
- bancos internos `.autopilot`, caches, executáveis e ondas de simulação recriáveis;
- árvores completas de projetos `*_prj` geradas automaticamente quando o IP exportado, firmware, configurações e relatórios já foram preservados;
- caches de datasets públicos (CIFAR-10/MNIST) e o array CIFAR-10 completo de 122 MB;
- grandes snapshots/tarballs que apenas duplicavam árvores já representadas no repositório;
- cópias repetidas do dataset CIFAR-10 presentes em vários pacotes de deploy.

Nada dessas categorias foi apagado do diretório original. Elas continuam disponíveis no computador local.

## Decisão sobre `.bit`, `.hwh` e modelos

- `.bit`: incluído; permite reprogramar exatamente o FPGA usado no experimento.
- `.hwh`: incluído; necessário para o PYNQ reconhecer registradores, endereços e hierarquia do overlay.
- `.h5`/`.keras`: incluídos; preservam modelos de referência e intermediários essenciais.
- `.xmodel`: incluído; é a entrega compilada do fluxo Vitis AI para a DPU-alvo.
- IP `.zip`: incluído; permite importar o acelerador no Vivado sem repetir toda a síntese HLS.
- `.dcp`: incluído quando representa um checkpoint pós-síntese importante para auditoria/reabertura do projeto.

## Limites técnicos

GitHub rejeita blobs acima de 100 MB sem Git LFS. Como Git LFS não estava instalado no ambiente no momento da curadoria, nenhum arquivo acima desse limite foi adicionado. O maior arquivo incluído possui 98.976.658 bytes.

