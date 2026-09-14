#!/usr/bin/env python3
"""Inventaria e fixa por SHA-256 todos os artefatos do fluxo hls4ml."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'manifests'
EXCLUDE={'__pycache__','.pytest_cache','manifests'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def role(p):
 s=p.as_posix()
 if '/golden/' in '/'+s: return 'board_golden'
 if s.endswith('.zip') and '/impl/ip/' in s: return 'deployable_ip'
 if s.endswith(('.rpt','.xml')): return 'report'
 if s.endswith(('.tcl','.py','.sh')): return 'reproduction'
 if s.endswith(('.md','.txt')): return 'documentation'
 return 'generated_or_source'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or any(part in EXCLUDE for part in p.relative_to(ROOT).parts): continue
  files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'role':role(p)})
 m={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'flow':'hls4ml MLP Iris ap_fixed<16,6> RF=1 io_parallel Latency 100 MHz','board':'ZCU104','status':'host_ready_board_pending','file_count':len(files),'files':files}
 mp=OUT/'flow_manifest.json'; mp.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 (OUT/'SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in files),encoding='utf-8')
 print(json.dumps({'manifest':str(mp),'file_count':len(files),'bytes':sum(x['bytes'] for x in files)},indent=2))
if __name__=='__main__': main()
