#!/usr/bin/env python3
"""Valida as 30 amostras canônicas no XModel antes de qualquer benchmark."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from vart_common import IrisDpu,load_preprocessing,softmax
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--xmodel',type=Path,default=HERE/'iris_mlp.xmodel'); ap.add_argument('--preprocessing',type=Path,default=HERE/'preprocessing.json'); ap.add_argument('--test-data',type=Path,default=HERE/'iris_test.npz'); ap.add_argument('--reference',type=Path,default=HERE/'quantized_test_outputs.npz'); ap.add_argument('--output-dir',type=Path,default=HERE/'validation_output'); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 data=np.load(a.test_data); ref=np.load(a.reference); normalized=data['features']; labels=data['labels']; raw=data['raw_features']; refpred=ref['predictions']
 dpu=IrisDpu(a.xmodel); rows=[]; pred=[]
 for i,x in enumerate(normalized):
  qi=dpu.quantize(x); qo=dpu.execute_quantized(qi); logits=dpu.decode(qo); p=int(np.argmax(logits)); pred.append(p); rows.append([i,*raw[i].tolist(),*x.tolist(),*qi.reshape(-1).tolist(),*qo.reshape(-1).tolist(),*logits.tolist(),*softmax(logits).tolist(),p,int(refpred[i]),int(labels[i])])
 pred=np.asarray(pred); correct=int(np.sum(pred==labels)); agree=int(np.sum(pred==refpred)); passed=correct==29 and agree==30
 cols=['sample',*[f'raw_x{i}' for i in range(4)],*[f'norm_x{i}' for i in range(4)],*[f'input_int8_{i}' for i in range(4)],*[f'output_int8_{i}' for i in range(3)],*[f'logit_{i}' for i in range(3)],*[f'prob_{i}' for i in range(3)],'prediction','reference_prediction','label']
 with (a.output_dir/'predictions.csv').open('w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(cols); w.writerows(rows)
 result={'status':'passed' if passed else 'failed','samples':30,'correct':correct,'accuracy':correct/30,'reference_agreement':agree,'reference_agreement_percent':100*agree/30,'input_shape':[1,4],'output_shape':[1,3],'input_fix_point':dpu.input_fix,'output_fix_point':dpu.output_fix,'xmodel_sha256':sha(a.xmodel),'test_data_sha256':sha(a.test_data),'reference_sha256':sha(a.reference)}
 (a.output_dir/'validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(0 if passed else 2)
if __name__=='__main__': main()
