#!/usr/bin/env python3
"""Audita o ponto de entrega do host antes das duas implementações na ZCU104."""
from __future__ import annotations
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def add(rows,name,ok,detail):rows.append({'check':name,'status':'passed' if ok else 'failed','detail':detail})
def verify_sum(path,base):
 bad=[];n=0
 for line in path.read_text().splitlines():
  if not line.strip():continue
  expected,name=line.split(None,1);p=base/name.strip();n+=1
  if not p.is_file():bad.append(f'missing:{name.strip()}')
  elif sha(p)!=expected:bad.append(f'hash:{name.strip()}')
 return n,bad
def main():
 r=[];add(r,'canonical_methodology',(ROOT/'METODOLOGIA_BENCHMARK_MLP_IRIS.md').is_file(),'MLP/Iris protocol present')
 x=ROOT/'Vitis AI/artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel';add(r,'xmodel_sha256',x.is_file() and sha(x)=='47921742c6470b190d448278b8c1ec94d431ec4841bb52d19bb7c537fde6a1ce',sha(x) if x.is_file() else 'missing')
 ip=ROOT/'hls4ml/mlp_iris_apfixed16_6_rf1_100mhz/mlp_iris_prj/solution1/impl/ip/xilinx_com_hls_mlp_iris_1_0.zip';add(r,'hls_ip_sha256',ip.is_file() and sha(ip)=='88a11380ea472636af4b4b72348aeda9e0b3b2421456b71b0b30222b2a774ba3',sha(ip) if ip.is_file() else 'missing')
 for label,path,base in [('vitis_deploy',ROOT/'Vitis AI/artifacts/deploy/iris_mlp_zcu104_vai3_5/SHA256SUMS.txt',ROOT/'Vitis AI/artifacts/deploy/iris_mlp_zcu104_vai3_5'),('vitis_flow',ROOT/'Vitis AI/manifests/SHA256SUMS.txt',ROOT/'Vitis AI'),('hls_golden',ROOT/'hls4ml/golden/SHA256SUMS.txt',ROOT/'hls4ml/golden'),('hls_flow',ROOT/'hls4ml/manifests/SHA256SUMS.txt',ROOT/'hls4ml')]:
  try:n,bad=verify_sum(path,base);add(r,label+'_hashes',not bad,f'{n} files; '+('all hashes valid' if not bad else '; '.join(bad[:10])))
  except Exception as e:add(r,label+'_hashes',False,str(e))
 vg=json.loads((ROOT/'Vitis AI/reports/quantized_validation.json').read_text());add(r,'vitis_host_validation',vg.get('status')=='passed' and vg.get('correct')==29 and vg.get('float_argmax_agreement')==30,'29/30; agreement 30/30')
 hg=json.loads((ROOT/'hls4ml/golden/MANIFEST.json').read_text());add(r,'hls_rtl_golden',hg.get('status')=='passed' and hg['validation']['correct']==29,'29/30; RTL golden')
 for platform in ('CPU','GPU'):
  if platform=='CPU':counts=(1+len(list((ROOT/'CPU/validacao_rapl').glob('30000_rep*'))),1+len(list((ROOT/'CPU/validacao_rapl').glob('100000_rep*'))))
  else:counts=(1+len(list((ROOT/'GPU/replicas/30000').glob('rep*'))),1+len(list((ROOT/'GPU/replicas/100000').glob('rep*'))))
  add(r,platform.lower()+'_five_campaigns',counts==(5,5),f'30k={counts[0]}, 100k={counts[1]}')
 # Links Markdown locais, exceto documentos brutos gerados por campanha.
 broken=[];checked=0
 for md in [ROOT/'README.md',ROOT/'CPU/README.md',ROOT/'GPU/README.md',ROOT/'Vitis AI/README.md',ROOT/'Vitis AI/README_CHATGPT_WEB_ZCU104_XMODEL.md',ROOT/'hls4ml/README.md',ROOT/'hls4ml/README_CHATGPT_WEB_ZCU104_IP.md']:
  for raw in re.findall(r'\]\(([^)]+)\)',md.read_text(encoding='utf-8')):
   target=raw.strip('<>').split('#',1)[0]
   if not target or '://' in target or target.startswith('mailto:'):continue
   checked+=1
   if not (md.parent/target).exists():broken.append(f'{md.relative_to(ROOT)} -> {target}')
 add(r,'documentation_links',not broken,f'{checked} checked'+('' if not broken else '; '+ '; '.join(broken[:10])))
 host=all(x['status']=='passed' for x in r)
 report={'generated_utc':datetime.now(timezone.utc).isoformat(),'host_ready_for_board_stage':host,'board_measurements_complete':False,'expected_pending':['Vitis AI: compatible DPU image, physical validation and 5x30k + 5x100k','hls4ml: AXI wrapper, block design, full post-route timing, bit/hwh, physical validation and 5x30k + 5x100k','calibrated ZCU104/external power measurement'], 'checks':r}
 (ROOT/'HOST_READINESS_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 lines=['# Relatório de prontidão para a ZCU104','',f"**Host pronto para iniciar a placa:** {'SIM' if host else 'NÃO'}",'', '| Verificação | Estado | Detalhe |','|---|---:|---|']+[f"| {x['check']} | {x['status']} | {x['detail'].replace('|','/')} |" for x in r]+['','## Pendências que exigem a placa','','- Vitis AI: compatibilidade da imagem/DPU, validação física e 5 campanhas de 30k + 5 de 100k.','- HLS4ML: wrapper AXI, block design, timing pós-route completo, bit/HWH, validação física e 5 + 5 campanhas.','- Energia: descoberta e calibração de sensores ou wattímetro externo com escopo documentado.','', 'Este relatório não declara a implementação na placa concluída; apenas confirma que os insumos de host estão consistentes.']
 (ROOT/'HOST_READINESS_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(report,indent=2,ensure_ascii=False));raise SystemExit(0 if host else 2)
if __name__=='__main__':main()
