#!/usr/bin/env python3
"""Inventaria hwmon/IIO e não atribui unidades sem confirmação."""
from __future__ import annotations
import json,time
from pathlib import Path

def read(p):
 try:return p.read_text().strip()
 except Exception:return None
def main():
 rows=[]
 for root in [Path('/sys/class/hwmon'),Path('/sys/bus/iio/devices')]:
  if not root.exists():continue
  for p in sorted(root.glob('**/*')):
   if p.is_file() and any(k in p.name for k in ('power','curr','in','temp','energy','name')):
    v=read(p)
    if v is not None and len(v)<200: rows.append({'path':str(p),'value':v})
 out={'captured_unix_s':time.time(),'warning':'Valores são brutos. Confirme datasheet/driver, unidade, escala, trilho e escopo antes de energia.','sensors':rows}
 print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
