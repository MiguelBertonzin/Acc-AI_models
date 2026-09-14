#!/usr/bin/env python3
"""Funções compartilhadas e verificações estritas para a MLP Iris via VART."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

EXPECTED_INPUT=(1,4); EXPECTED_OUTPUT=(1,3); EXPECTED_INPUT_FIX=5; EXPECTED_OUTPUT_FIX=3

def only_dpu_subgraph(graph):
 children=graph.get_root_subgraph().toposort_child_subgraph()
 dpu=[s for s in children if s.has_attr('device') and str(s.get_attr('device')).upper()=='DPU']
 if len(dpu)!=1: raise RuntimeError(f'Esperado um subgrafo DPU; encontrados {len(dpu)}')
 return dpu[0]
def fix(t):
 if not t.has_attr('fix_point'): raise RuntimeError(f'{t.name} sem fix_point')
 return int(t.get_attr('fix_point'))
def dims(t): return tuple(int(v) for v in t.dims)
def softmax(x):
 x=np.asarray(x,dtype=np.float32); e=np.exp(x-np.max(x)); return e/np.sum(e)
class IrisDpu:
 def __init__(self,xmodel:Path):
  import xir, vart
  self.graph=xir.Graph.deserialize(str(xmodel)); self.runner=vart.Runner.create_runner(only_dpu_subgraph(self.graph),'run')
  self.it=self.runner.get_input_tensors()[0]; self.ot=self.runner.get_output_tensors()[0]
  self.input_fix=fix(self.it); self.output_fix=fix(self.ot)
  if dims(self.it)!=EXPECTED_INPUT or dims(self.ot)!=EXPECTED_OUTPUT: raise RuntimeError(f'Shapes incompatíveis: {dims(self.it)} -> {dims(self.ot)}')
  if (self.input_fix,self.output_fix)!=(EXPECTED_INPUT_FIX,EXPECTED_OUTPUT_FIX): raise RuntimeError(f'fix_point incompatível: {self.input_fix}, {self.output_fix}')
  self.ib=np.empty(dims(self.it),dtype=np.int8); self.ob=np.empty(dims(self.ot),dtype=np.int8)
 def quantize(self,normalized):
  return np.clip(np.rint(np.asarray(normalized,dtype=np.float32)*2**self.input_fix),-128,127).astype(np.int8).reshape(dims(self.it))
 def execute_quantized(self,q):
  self.ib[...] = q; jid=self.runner.execute_async([self.ib],[self.ob]); self.runner.wait(jid); return self.ob.copy()
 def decode(self,qout): return np.asarray(qout,dtype=np.float32).reshape(-1)/(2**self.output_fix)
def load_preprocessing(path):
 d=json.loads(Path(path).read_text(encoding='utf-8')); return d,np.asarray(d['mean'],np.float32),np.asarray(d['scale'],np.float32)
