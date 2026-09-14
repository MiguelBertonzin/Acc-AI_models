#!/usr/bin/env python3
"""Gera inventário global do estado host-ready, excluindo snapshots e o próprio manifesto."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'manifests'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 files=[]
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or rel.parts[0] in {'snapshots','manifests','.git'} or '__pycache__' in rel.parts:continue
  files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 obj={'schema_version':1,'state':'host_ready_for_zcu104_board_stage','generated_utc':datetime.now(timezone.utc).isoformat(),'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),'files':files}
 (OUT/'host_ready_manifest.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
 (OUT/'HOST_READY_SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in files))
 print(json.dumps({k:obj[k] for k in ('state','file_count','total_bytes')},indent=2))
if __name__=='__main__':main()
