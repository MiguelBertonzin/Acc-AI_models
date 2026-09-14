LeNet/MNIST - Vitis AI / ZCU104
Campanha: 12 configuracoes x 10.000 inferencias

Cenarios:
  inference_only: 1,2,3,4 threads
  end_to_end:     1,2,3,4 threads
  saturated:      1,2,3,4 threads

Consultar FINAL_SUMMARY.csv para a tabela principal.
Cada configuracao contem summary.json, blocks.csv, latencies_ns.npy,
telemetria idle/ativa e SHA256SUMS.txt.

Acuracia e concordancia sao calculadas somente para as 10.000 imagens unicas
nos cenarios inference_only e end_to_end; saturated caracteriza desempenho.

Potencia/energia: INA226 no rail de aproximadamente 12 V exposto via sysfs.
Energia ativa: integral trapezoidal da potencia dentro do intervalo medido.
Energia dinamica: max(E_total - P_idle * tempo, 0).

Nenhum outlier e removido.
