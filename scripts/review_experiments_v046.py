"""Targeted revision experiments, with study-level nested model selection."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, ctypes, csv, json, sys, subprocess, time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from aquabiosim.experiment_support import (prepared, coefficients, candidate_bounds, val,
    fit_heads, head_predict, ANCHORS, secondary, proxy_residuals)
OUT=ROOT/'artifacts/reviewer-revision-v0.46'
CANDIDATES=('gain_core','mass_core','process_scale','learned_maintenance')
COUNTS={'gain_core':9,'mass_core':9,'process_scale':12,'learned_maintenance':15}
LIBS={}

def read(path): return list(csv.DictReader((ROOT/path).open(encoding='utf-8-sig')))
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def write(name,records):
 if not records:return
 path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in records for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(records)
def load_data(): return read('data/processed/multisource_v0.18/treatments.csv'),read('data/processed/multisource_v0.18/process_targets.csv'),read('data/processed/multisource_v0.18/secondary_targets.csv')
def target(r):return val(r,'wf_g') if r['wf_g'] else val(r,'w0_g')*val(r,'observed_growth_ratio')
def core(q,candidate):
 x=np.zeros(14);x[:min(14,COUNTS[candidate]) if candidate=='learned_maintenance' else 9]=q[:14] if candidate=='learned_maintenance' else q[:9]
 return x

def setup():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'runtime').mkdir(exist_ok=True)
 src=(ROOT/'src/aquabiosim/experiment_fast.c').read_text()
 path=OUT/'runtime/revision_engine.c';path.write_text(src)
 subprocess.run(['cc','-O3','-std=c99','-shared','-fPIC',str(path),'-o',str(OUT/'runtime/revision_engine.dylib'),'-lm'],check=True)

def sim(x,d,learned_log=None):
 path=str(OUT/'runtime/revision_engine.dylib')
 if path not in LIBS:
  lib=ctypes.CDLL(path);p=ctypes.POINTER(ctypes.c_double)
  lib.aquasim.argtypes=[ctypes.c_int,p,p,p,p,p,p,p,ctypes.c_int,ctypes.c_double,p];lib.aquasim.restype=ctypes.c_int;LIBS[path]=lib
 cap,eta=coefficients(np.asarray(x)[:14],d)
 pars=np.asarray(x) if learned_log is None else np.r_[x,learned_log]
 arrays=[d['inputs'],cap,eta,d['adc'],d['urine'],d['rho'],pars]
 arrays=[np.ascontiguousarray(a,dtype=float) for a in arrays];out=np.empty((len(d['rows']),7));p=ctypes.POINTER(ctypes.c_double)
 status=LIBS[path].aquasim(len(out),*[a.ctypes.data_as(p) for a in arrays],len(pars),1.,out.ctypes.data_as(p))
 if status or np.any(out[:,0]<=0) or np.any(~np.isfinite(out)):raise ValueError('Integration domain')
 return {'weight_g':out[:,0],'feed_g':out[:,1],'energy_balance_error_kj':out[:,2]-out[:,3]-out[:,4]-out[:,5]-out[:,6]-d['rho']*(out[:,0]-d['inputs'][:,0]),'maintenance_kj':out[:,6]}

def apply_process(d,base,scales):
 d.update(adc=np.clip(expit(base[:,0]+scales[0]),.35,.98),urine=np.clip(expit(base[:,1]+scales[1]),.001,.25),rho=np.clip(np.exp(base[:,2]+scales[2]),2.,12.))
def process_base(rows,heads):
 hp=head_predict(rows,heads)
 return np.column_stack([logit(np.clip(hp.get('adc_energy_pct',np.full(len(rows),80.))/100,1e-10,1-1e-10)),logit(np.clip(hp.get('urinary_fraction_pct',np.full(len(rows),7.))/100,1e-10,1-1e-10)),np.log(hp.get('body_energy_kj_g',np.full(len(rows),4.8)))])
def forward(fit,rows,b=None):
 candidate=fit['candidate'];q=np.array(fit['parameters']);d=prepared(rows,fit['process_heads'],'nutrition_compact')
 if candidate=='process_scale':apply_process(d,process_base(rows,fit['process_heads']),q[9:12])
 if b is not None:d.update(capacity_log_offset=b[0],eta_offset=b[1])
 return sim(core(q,candidate),d,q[14] if candidate=='learned_maintenance' else None)

def fit_global(candidate,excluded,rows,targets,diag):
 excluded=tuple(sorted(excluded));key=candidate+'__'+('__'.join(excluded) or 'all');path=OUT/'fits'/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 train=[r for r in rows if r['study_id'] not in excluded];studies=sorted({r['study_id'] for r in train});heads=fit_heads(train,targets)
 assert all(not(set(h['studies'])&set(excluded)) for h in heads.values())
 d=prepared(train,heads,'nutrition_compact');ix=np.array([studies.index(r['study_id']) for r in train]);px=secondary(train,diag);nc=COUNTS[candidate]
 lower,upper=candidate_bounds('nutrition_compact')
 if candidate=='learned_maintenance':lo=np.r_[lower[:14],-12.];hi=np.r_[upper[:14],-12.+np.log(1e5)]
 elif candidate=='process_scale':lo=np.r_[lower[:9],-2.,-2.,-np.log(2.)];hi=np.r_[upper[:9],2.,2.,np.log(2.)]
 else:lo=lower[:9];hi=upper[:9]
 lo=np.r_[lo,[-4.]*(2*len(studies))];hi=np.r_[hi,[4.]*(2*len(studies))]
 groups=[]
 for s in studies:
  ii=np.array([i for i,r in enumerate(train) if r['study_id']==s]);w0=np.array([val(train[i],'w0_g') for i in ii]);yy=np.array([target(train[i]) for i in ii]);jj=np.array([i for i in ii if train[i]['observed_feed_g']],int)
  groups.append((ii,w0,yy,jj,np.array([val(train[i],'observed_feed_g') for i in jj])))
 base=process_base(train,heads);process=[]
 if candidate=='process_scale':
  ids={r['treatment_id']:i for i,r in enumerate(train)}
  for j,metric in enumerate(('adc_energy_pct','urinary_fraction_pct','body_energy_kj_g')):
   tt=[t for t in targets if t['treatment_id'] in ids and t['metric']==metric]
   for s in sorted({t['study_id'] for t in tt}):
    rr=[t for t in tt if t['study_id']==s];ii=np.array([ids[t['treatment_id']] for t in rr]);yy=np.array([val(t,'value') for t in rr]);yy=logit(np.clip(yy/100,1e-10,1-1e-10)) if j<2 else np.log(yy)
    ww=np.sqrt(.25*np.array([.5 if t['kind'].startswith('component-calculated') else 1. for t in rr])/len(rr));process.append((j,ii,yy,ww))
 def residual(q):
  d.update(capacity_log_offset=q[nc:][::2][ix],eta_offset=q[nc:][1::2][ix])
  if candidate=='process_scale':apply_process(d,base,q[9:12])
  ss=sim(core(q,candidate),d,q[14] if candidate=='learned_maintenance' else None);out=[]
  for ii,w0,yy,jj,fi in groups:
   prediction=ss['weight_g'][ii]
   out.extend(np.log(np.maximum(prediction-w0,1e-7)/(yy-w0))/np.sqrt(len(ii)) if candidate=='gain_core' else np.log(prediction/yy)/np.sqrt(len(ii)))
   if len(jj):out.extend(2*np.log(ss['feed_g'][jj]/fi)/np.sqrt(len(jj)))
  out.extend(proxy_residuals(ss,train,px))
  if candidate=='learned_maintenance':extra=q[8:15].copy();extra[0]-=1;extra[-1]=(extra[-1]+12)/np.log(1e5)
  elif candidate=='process_scale':extra=np.r_[q[8]-1,q[9:12]]
  else:extra=np.array([q[8]-1])
  out.extend(np.sqrt(.1)*extra);out.extend(.5*q[nc:])
  for j,ii,yy,ww in process:out.extend(ww*(base[ii,j]+q[9+j]-yy))
  return np.asarray(out)
 seeds=[]
 for j,theta in enumerate((-.3,-1.,.3)):
  s=np.r_[[-3.6,0.,-.15,theta,-.3,-.8,-.4,-.15,1.],np.zeros(nc-9+2*len(studies))]
  if candidate=='learned_maintenance':s[14]=-12.+[0,.5,.8][j]*np.log(1e5)
  seeds.append(np.clip(s,lo+1e-7,hi-1e-7))
 if len(excluded)<=1 and candidate in ('gain_core','mass_core'):
  folder='gain-objective-optimization-v0.39' if candidate=='gain_core' else 'core-optimization-v0.30'
  recs=json.loads((ROOT/f'models/initializations/{folder}.json').read_text())
  cc='gain_only' if candidate=='gain_core' else 'core9_chemistry_only'
  ref=next((r for r in recs if r['candidate']==cc and r['held']==(excluded[0] if excluded else 'all')),None)
  if ref:
   pars=np.array(ref['parameters']);seeds[0]=np.clip(pars,lo+1e-7,hi-1e-7)
 length=len(residual(seeds[0]));attempts=[];best=None
 def safe(q):
  try:
   with np.errstate(over='raise',invalid='raise',divide='raise'):return residual(q)
  except (ValueError,FloatingPointError):return np.full(length,1e3)
 for seed in seeds:
  f=least_squares(safe,seed,bounds=(lo,hi),max_nfev=400,ftol=1e-7,xtol=1e-7,gtol=1e-7);valid=np.max(abs(safe(f.x)))<100
  attempts.append({'success':bool(f.success),'valid':bool(valid),'nfev':int(f.nfev),'objective':float(2*f.cost)})
  if valid and (best is None or f.cost<best.cost):best=f
 if best is None:raise ValueError('No fit '+key)
 if not best.success:
  retry=least_squares(safe,best.x,bounds=(lo,hi),max_nfev=800,ftol=1e-7,xtol=1e-7,gtol=1e-7)
  attempts.append({'success':bool(retry.success),'valid':bool(np.max(abs(safe(retry.x)))<100),'nfev':int(retry.nfev),'objective':float(2*retry.cost)})
  if retry.cost<=best.cost:best=retry
 out={'candidate':candidate,'excluded_studies':list(excluded),'parameters':best.x.tolist(),'global_count':nc,'study_effect_studies':studies,'train_ids':[r['treatment_id'] for r in train],'process_heads':heads,'objective':float(2*best.cost),'success':bool(best.success),'attempts':attempts,'maintenance': 'estimated_in_training' if candidate=='learned_maintenance' else 'zero_fixed'}
 save(path,out);return out

def control_residual(ss,rows,ai,diag):
 r=rows[ai];out=[np.log(ss['weight_g'][ai]/target(r))]
 if r['observed_feed_g']:out.append(2*np.log(ss['feed_g'][ai]/val(r,'observed_feed_g')))
 pp=secondary([r],diag);out.extend(proxy_residuals({'weight_g':ss['weight_g'][ai:ai+1],'feed_g':ss['feed_g'][ai:ai+1]},[r],pp))
 return np.asarray(out)
def calibrate(fit,rows,diag,penalty=None,growth_only=False,seeds=None):
 ai=next(i for i,r in enumerate(rows) if r['treatment_id']==ANCHORS[rows[0]['study_id']]);r=rows[ai]
 penalty=(0. if r['observed_feed_g'] or secondary([r],diag) else .25) if penalty is None else penalty
 def residual(b):
  try:
   with np.errstate(over='raise',invalid='raise',divide='raise'):
    sh=[0.,b[0]] if growth_only else b
    ss=forward(fit,rows,sh);rr=[np.log(ss['weight_g'][ai]/target(r))] if growth_only else control_residual(ss,rows,ai,diag)
    return np.r_[rr,np.sqrt(penalty)*b] if not growth_only else np.array(rr)
  except (ValueError,FloatingPointError):return np.full(1 if growth_only else 3+int(bool(r['observed_feed_g']) or bool(secondary([r],diag))),1e3)
 n=1 if growth_only else 2;seeds=seeds or ([0.],[-1.],[1.]) if growth_only else seeds or ([0.,0.],[-1.,0.],[1.,-1.])
 best=None;attempts=[]
 for seed in seeds:
  f=least_squares(residual,seed,bounds=([-4.]*n,[4.]*n),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  attempts.append({'shifts':f.x.tolist(),'objective':float(2*f.cost),'success':bool(f.success)})
  if np.max(abs(residual(f.x)))<100 and (best is None or f.cost<best.cost):best=f
 if best is None:raise ValueError('No valid reference calibration '+r['treatment_id'])
 b=np.r_[0.,best.x[0]] if growth_only else best.x
 return b,{'anchor_id':r['treatment_id'],'shifts':b.tolist(),'penalty':float(penalty),'objective':float(2*best.cost),'attempts':attempts,'used_outcome_ids':[r['treatment_id']]}

def prediction_rows(label,fit,rows,ss,stage='joint'):
 out=[]
 for i,r in enumerate(rows):
  if r['treatment_id']==ANCHORS[r['study_id']]:continue
  w0=val(r,'w0_g');w=ss['weight_g'][i];fi=ss['feed_g'][i]
  out.append({'model':label,'stage':stage,'study_id':r['study_id'],'treatment_id':r['treatment_id'],'observed_weight_g':target(r),'predicted_weight_g':float(w),'initial_weight_g':w0,'observed_feed_g':r['observed_feed_g'],'predicted_feed_g':float(fi),'W_APE_pct':float(100*abs(w/target(r)-1)),'gain_APE_pct':float(100*abs((w-w0)/(target(r)-w0)-1)),'FI_APE_pct':float(100*abs(fi/val(r,'observed_feed_g')-1)) if r['observed_feed_g'] else ''})
 return out

def summaries(pp):
 mm=[];summary=[]
 for label in sorted({r['model'] for r in pp}):
  qq=[r for r in pp if r['model']==label]
  for s in sorted({r['study_id'] for r in qq}):
   rr=[r for r in qq if r['study_id']==s];fi=[r for r in rr if r['FI_APE_pct']!=''];m={'model':label,'study_id':s,'n':len(rr),'W_MAPE_pct':float(np.mean([r['W_APE_pct'] for r in rr])),'gain_MAPE_pct':float(np.mean([r['gain_APE_pct'] for r in rr])),'FI_n':len(fi),'FI_MAPE_pct':float(np.mean([r['FI_APE_pct'] for r in fi])) if fi else ''};mm.append(m)
  bb=[r for r in mm if r['model']==label];ff=[r for r in bb if r['FI_n']];summary.append({'model':label,'studies':len(bb),'treatments':len(qq),'W_MAPE_pct':float(np.mean([r['W_MAPE_pct'] for r in bb])),'gain_MAPE_pct':float(np.mean([r['gain_MAPE_pct'] for r in bb])),'FI_studies':len(ff),'FI_n':sum(r['FI_n'] for r in ff),'FI_MAPE_pct':float(np.mean([r['FI_MAPE_pct'] for r in ff])) if ff else ''})
 return mm,summary

def job_fit(job):
 candidate,excluded,rows,targets,diag=job;t=time.time();f=fit_global(candidate,excluded,rows,targets,diag)
 return {'candidate':candidate,'excluded':excluded,'success':f['success'],'seconds':round(time.time()-t,2)}

def nested():
 rows,targets,diag=load_data();studies=sorted(ANCHORS);fi_studies={r['study_id'] for r in rows if r['observed_feed_g']};splits={};jobs={}
 for outer in studies:
  available=[s for s in studies if s!=outer];blocks=[[] for _ in range(3)]
  for j,s in enumerate(sorted(fi_studies-{outer})):blocks[j].append(s)
  for s in [s for s in available if s not in fi_studies]:min(blocks,key=len).append(s)
  splits[outer]=blocks
  for c in CANDIDATES:
   for excluded in [(outer,)]+[tuple(sorted([outer]+b)) for b in blocks]:jobs[(c,excluded)]=(c,excluded,rows,targets,diag)
 jobs[('gain_core',())]=('gain_core',(),rows,targets,diag)
 save(OUT/'nested_protocol.json',{'outer':studies,'inner_blocks':splits,'candidates':list(CANDIDATES),'selection_score':'0.5 * macro_gain_MAPE + 0.5 * macro_FI_MAPE, on inner source calibrated predictions','tie_break':'smaller global coefficient count, then candidate name','seeds':'three bounded starts; inner starts generic; outer core warm starts use matching original omitted-study fits','maintenance':'zero fixed in core/process configurations; maintenance model fits an absolute log rate only to its training studies','targets':'one completed control per source; remaining treatments only scored','fits_requested':len(jobs),'max_nfev':400,'retry_max_nfev':800,'inner_fi_assignment':'spread intake studies across blocks before balancing source counts'})
 done=0;t=time.time()
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(job_fit,j):key for key,j in jobs.items()}
  for future in as_completed(futures):
   rr=future.result();done+=1;save(OUT/'nested_status.json',{'completed':done,'total':len(jobs),'seconds':round(time.time()-t,1),'last':rr});print(json.dumps({'done':done,'total':len(jobs),**rr}),flush=True)
 results=[];selection=[];inner_scores=[];outer_pp=[]
 for outer in studies:
  per_candidate={}
  for c in CANDIDATES:
   inner=[]
   for block in splits[outer]:
    fit=fit_global(c,tuple(sorted([outer]+block)),rows,targets,diag)
    for s in block:
     rr=[r for r in rows if r['study_id']==s];b,info=calibrate(fit,rr,diag);inner.extend(prediction_rows(c,fit,rr,forward(fit,rr,b)))
   mm,summary=summaries(inner);z=summary[0];score=.5*(z['gain_MAPE_pct']+z['FI_MAPE_pct']);per_candidate[c]=score;inner_scores.append({'outer_study':outer,'candidate':c,'score':score,**{k:v for k,v in z.items() if k!='model'}})
  chosen=min(CANDIDATES,key=lambda c:(per_candidate[c],COUNTS[c],c));selection.append({'study_id':outer,'selected':chosen,'selection_score':per_candidate[chosen],**{'score_'+c:per_candidate[c] for c in CANDIDATES}})
  for c in CANDIDATES:
   fit=fit_global(c,(outer,),rows,targets,diag);rr=[r for r in rows if r['study_id']==outer];b,info=calibrate(fit,rr,diag);save(OUT/'calibrations'/f'{c}__{outer}.json',info);p=prediction_rows(c,fit,rr,forward(fit,rr,b));outer_pp.extend(p)
   if c==chosen:results.extend([dict(x,model='nested_selected') for x in p])
 outer_pp.extend(results);mm,summary=summaries(outer_pp);write('nested_predictions.csv',outer_pp);write('nested_study_metrics.csv',mm);write('nested_summary.csv',summary);write('inner_selection_scores.csv',inner_scores);write('selection_by_study.csv',selection)
 save(OUT/'nested_status.json',{'status':'complete','fits':len(jobs),'seconds':round(time.time()-t,1),'summary':summary});print(json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['nested']);args=p.parse_args();setup();nested()
