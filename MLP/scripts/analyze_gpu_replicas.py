#!/usr/bin/env python3
"""Agrega as cinco campanhas GPU usando a campanha, e não a inferência, como unidade."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'GPU/replicas'
def load30(p):
 with (p/'metricas_desempenho.csv').open() as f:r=list(csv.DictReader(f))[-1]
 v=json.loads((p/'validacao_estatistica.json').read_text())
 return {'inference_only_mean_ms':float(r['latency_mean_ms']),'application_e2e_effective_ms':1000/float(r['throughput_effective_fps']),'throughput_effective_inf_s':float(r['throughput_effective_fps']),'total_energy_mj_per_inference':1000*float(r['energy_total_per_inference_j']),'correct':int(v['correct']),'accuracy':float(v['accuracy'])}
def load100(p):
 r=json.loads((p/'metricas_resumo.json').read_text());v=json.loads((p/'validacao_execucao.json').read_text())
 return {'inference_only_mean_ms':float(r['inference_only']['mean_ms']),'application_e2e_effective_ms':float(r['end_to_end_effective_mean_ms']),'throughput_effective_inf_s':float(r['throughput_effective_fps']),'total_energy_mj_per_inference':1000*float(r['energy_total_per_inference_j']),'correct':int(v['correct']),'accuracy':float(v['accuracy'])}
def stat(vals):
 a=np.asarray(vals,float);se=np.std(a,ddof=1)/math.sqrt(len(a));c=t.ppf(.975,len(a)-1)
 return {'n_campaigns':len(a),'mean':float(np.mean(a)),'std_sample':float(np.std(a,ddof=1)),'cv_percent':float(100*np.std(a,ddof=1)/np.mean(a)),'minimum':float(np.min(a)),'maximum':float(np.max(a)),'ci95_lower':float(np.mean(a)-c*se),'ci95_upper':float(np.mean(a)+c*se)}
def main():
 dirs30=[ROOT/'GPU/resultados',*[OUT/'30000'/f'rep{i:02d}' for i in range(2,6)]];dirs100=[ROOT/'GPU/resultados_100000',*[OUT/'100000'/f'rep{i:02d}' for i in range(2,6)]]
 rows=[]
 for size,dirs,loader in [(30000,dirs30,load30),(100000,dirs100,load100)]:
  for i,p in enumerate(dirs,1):rows.append({'inferences':size,'replicate':i,'directory':str(p.relative_to(ROOT)),**loader(p)})
 keys=['inference_only_mean_ms','application_e2e_effective_ms','throughput_effective_inf_s','total_energy_mj_per_inference']; summary={str(size):{k:stat([r[k] for r in rows if r['inferences']==size]) for k in keys} for size in (30000,100000)}
 summary['validation']={'all_campaigns_correct_29_of_30':all(r['correct']==29 for r in rows),'campaigns':len(rows),'note':'Acurácia usa apenas o holdout único de 30; repetições não são amostras adicionais.'}
 with (OUT/'campanhas_gpu.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (OUT/'resumo_5_campanhas.json').write_text(json.dumps(summary,indent=2)+'\n')
 def line(size,key,unit):
  x=summary[str(size)][key];return f"| {size:,} | {key} | {x['mean']:.6f} {unit} | {x['std_sample']:.6f} | [{x['ci95_lower']:.6f}, {x['ci95_upper']:.6f}] | {x['cv_percent']:.2f}% |"
 report='''# GPU — resumo de cinco campanhas independentes\n\nA unidade experimental é a campanha (`n=5` por tamanho). Cada processo manteve batch 1, 200 warmups, baseline de 5 s, telemetria a 100 ms, semente 20260831 e nenhuma remoção de outliers. A primeira campanha é o diretório histórico e `rep02`–`rep05` foram executadas em 2026-09-03.\n\n| Inferências | Métrica | Média entre campanhas | DP | IC95% t | CV |\n|---:|---|---:|---:|---:|---:|\n'''
 for size in (30000,100000):
  report+=line(size,'inference_only_mean_ms','ms')+'\n'+line(size,'application_e2e_effective_ms','ms')+'\n'+line(size,'throughput_effective_inf_s','inf/s')+'\n'+line(size,'total_energy_mj_per_inference','mJ')+'\n'
 report+='''\nTodas as dez campanhas mantiveram 29/30 (96,67%) nas 30 amostras únicas. Os valores energéticos usam `nvidia-smi` e representam potência total da placa GPU, não são diretamente equivalentes a RAPL ou aos trilhos internos da ZCU104. Para a comparação energética principal, use o mesmo wattímetro externo.\n\nArquivos: `campanhas_gpu.csv` contém uma linha por campanha; `resumo_5_campanhas.json` preserva os cálculos completos.\n'''
 (OUT/'RESUMO_5_CAMPANHAS.md').write_text(report)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
