#!/usr/bin/env python3
"""Benchmark canônico batch-1 da MLP Iris na DPU da ZCU104."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,threading,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from vart_common import IrisDpu,load_preprocessing
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def summary(v,pass_means):
 a=np.asarray(v,float); pm=np.asarray(pass_means,float); n=len(pm); df=max(n-1,1); z=1.959963984540054
 # Expansão de Cornish-Fisher para t(0,975), suficiente para n>=30.
 tc=z+(z**3+z)/(4*df)+(5*z**5+16*z**3+3*z)/(96*df**2)
 se=float(np.std(pm,ddof=1)/math.sqrt(n)) if n>1 else float('nan'); m=float(np.mean(a));
 return {'count':len(a),'mean_ms':m,'median_ms':float(np.median(a)),'std_sample_ms':float(np.std(a,ddof=1)),'cv_percent':float(100*np.std(a,ddof=1)/m),'minimum_ms':float(np.min(a)),'maximum_ms':float(np.max(a)),'p90_ms':float(np.percentile(a,90)),'p95_ms':float(np.percentile(a,95)),'p99_ms':float(np.percentile(a,99)),'mean_ci95_lower_ms':float(np.mean(pm)-tc*se),'mean_ci95_upper_ms':float(np.mean(pm)+tc*se),'ci_unit':'means of complete 30-sample passages','ci_passages':n}
def parse_sensor(s):
 try:name,rest=s.split('=',1); path,mult=rest.rsplit(':',1); return name,Path(path),float(mult)
 except Exception as e:raise argparse.ArgumentTypeError('use NAME=/sys/path:multiplier_to_watts') from e
class Sampler:
 def __init__(self,sensors,interval):self.sensors=sensors;self.interval=interval;self.phase='setup';self.rows=[];self.stop_event=threading.Event();self.thread=None
 def start(self):self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def run(self):
  while not self.stop_event.is_set():
   row={'perf_counter_ns':time.perf_counter_ns(),'unix_s':time.time(),'phase':self.phase}
   for n,p,m in self.sensors:
    try:row[n]=float(p.read_text().strip())*m
    except Exception:row[n]=float('nan')
   self.rows.append(row);self.stop_event.wait(self.interval)
 def stop(self):self.stop_event.set();self.thread.join(timeout=2)
def integrate(rows,name,phase):
 r=[x for x in rows if x['phase']==phase and math.isfinite(x.get(name,float('nan')))]
 if len(r)<2:return None
 t=np.asarray([x['perf_counter_ns'] for x in r],float)/1e9;p=np.asarray([x[name] for x in r],float)
 return {'samples':len(r),'duration_s':float(t[-1]-t[0]),'mean_w':float(np.mean(p)),'median_w':float(np.median(p)),'std_sample_w':float(np.std(p,ddof=1)),'p95_w':float(np.percentile(p,95)),'maximum_w':float(np.max(p)),'energy_j':float(np.trapz(p,t))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--xmodel',type=Path,default=HERE/'iris_mlp.xmodel');ap.add_argument('--preprocessing',type=Path,default=HERE/'preprocessing.json');ap.add_argument('--test-data',type=Path,default=HERE/'iris_test.npz');ap.add_argument('--reference',type=Path,default=HERE/'quantized_test_outputs.npz');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--inferences',type=int,choices=[30000,100000],required=True);ap.add_argument('--warmup',type=int,default=200);ap.add_argument('--baseline-seconds',type=float,default=5);ap.add_argument('--telemetry-ms',type=int,default=100);ap.add_argument('--seed',type=int,default=20260831);ap.add_argument('--sensor',action='append',default=[],type=parse_sensor);ap.add_argument('--primary-power-sensor');a=ap.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise SystemExit(f'Recuso sobrescrever {a.output_dir}')
 a.output_dir.mkdir(parents=True);data=np.load(a.test_data);raw=data['raw_features'];normalized=data['features'];labels=data['labels'];ref=np.load(a.reference)['predictions'];prep,mean,scale=load_preprocessing(a.preprocessing);dpu=IrisDpu(a.xmodel)
 # Preflight funcional obrigatório.
 vp=[]
 for x in normalized:vp.append(int(np.argmax(dpu.decode(dpu.execute_quantized(dpu.quantize(x))))))
 vp=np.asarray(vp);validation={'correct':int(np.sum(vp==labels)),'accuracy':float(np.mean(vp==labels)),'reference_agreement':int(np.sum(vp==ref)),'passed':bool(np.sum(vp==labels)==29 and np.sum(vp==ref)==30)}
 (a.output_dir/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 if not validation['passed']:raise SystemExit('Validação falhou; benchmark bloqueado')
 sampler=Sampler(a.sensor,a.telemetry_ms/1000);sampler.start();sampler.phase='baseline';time.sleep(a.baseline_seconds);sampler.phase='warmup'
 rng=np.random.default_rng(a.seed)
 for i in range(a.warmup):x=normalized[rng.integers(0,30)];dpu.execute_quantized(dpu.quantize(x))
 acc=[];device=[];e2e=[];passrows=[];done=0;sampler.phase='benchmark';bench0=time.perf_counter_ns()
 while done<a.inferences:
  count=min(30,a.inferences-done);order=rng.permutation(30)[:count];p0=time.perf_counter_ns();aa=[];dd=[];ee=[]
  for idx in order:
   t0=time.perf_counter_ns();norm=(raw[idx]-mean)/scale;t1=time.perf_counter_ns();qin=dpu.quantize(norm);t2=time.perf_counter_ns();qout=dpu.execute_quantized(qin);t3=time.perf_counter_ns();logits=dpu.decode(qout);t4=time.perf_counter_ns();_pred=int(np.argmax(logits));t5=time.perf_counter_ns();aa.append((t3-t2)/1e6);dd.append((t4-t1)/1e6);ee.append((t5-t0)/1e6)
  p1=time.perf_counter_ns();acc.extend(aa);device.extend(dd);e2e.extend(ee);passrows.append({'passage':len(passrows),'samples':count,'accelerator_mean_ms':float(np.mean(aa)),'device_call_mean_ms':float(np.mean(dd)),'application_e2e_mean_ms':float(np.mean(ee)),'passage_duration_ms':(p1-p0)/1e6});done+=count
 bench1=time.perf_counter_ns();sampler.phase='post';sampler.stop()
 full=[r for r in passrows if r['samples']==30]; metrics={'inferences':a.inferences,'full_passages':len(full),'final_passage_samples':passrows[-1]['samples'],'benchmark_wall_s':(bench1-bench0)/1e9,'throughput_effective_inf_s':a.inferences/((bench1-bench0)/1e9),'accelerator':summary(acc,[r['accelerator_mean_ms'] for r in full]),'device_call':summary(device,[r['device_call_mean_ms'] for r in full]),'application_end_to_end':summary(e2e,[r['application_e2e_mean_ms'] for r in full])}
 np.save(a.output_dir/'latencies_accelerator_ms.npy',np.asarray(acc));np.save(a.output_dir/'latencies_device_call_ms.npy',np.asarray(device));np.save(a.output_dir/'latencies_application_e2e_ms.npy',np.asarray(e2e))
 with (a.output_dir/'passages.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=passrows[0]);w.writeheader();w.writerows(passrows)
 fields=['perf_counter_ns','unix_s','phase',*[s[0] for s in a.sensor]]
 with (a.output_dir/'telemetry.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sampler.rows)
 power={'status':'not_measured','reason':'No calibrated primary sensor supplied'}
 if a.primary_power_sensor:
  base=integrate(sampler.rows,a.primary_power_sensor,'baseline');bench=integrate(sampler.rows,a.primary_power_sensor,'benchmark')
  if base and bench:
   dyn=max(0.,bench['energy_j']-base['mean_w']*bench['duration_s']);power={'status':'measured','scope':a.primary_power_sensor,'baseline':base,'benchmark':bench,'dynamic_energy_j':dyn,'total_mj_per_inference':1000*bench['energy_j']/a.inferences,'dynamic_mj_per_inference':1000*dyn/a.inferences}
 metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'protocol':'METODOLOGIA_BENCHMARK_MLP_IRIS.md','batch':1,'serial_synchronous':True,'warmup':a.warmup,'baseline_seconds':a.baseline_seconds,'telemetry_ms':a.telemetry_ms,'seed':a.seed,'samples_per_passage':30,'outliers_removed':0,'xmodel_sha256':sha(a.xmodel),'test_data_sha256':sha(a.test_data),'reference_sha256':sha(a.reference),'timing_boundaries':{'accelerator':'immediately before execute_async through completed wait','device_call':'before input quantization through output dequantization','application_end_to_end':'before StandardScaler transform through argmax'}}
 for name,obj in [('metrics_summary.json',metrics),('power_summary.json',power),('metadata.json',metadata)]:(a.output_dir/name).write_text(json.dumps(obj,indent=2)+'\n')
 generated=sorted(p for p in a.output_dir.iterdir() if p.name!='SHA256SUMS.txt');(a.output_dir/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in generated));print(json.dumps({'validation':validation,'metrics':metrics,'power':power},indent=2))
if __name__=='__main__':main()
