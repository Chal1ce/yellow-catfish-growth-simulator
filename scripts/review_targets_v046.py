"""Fair reference baselines, source calibration diagnostics and paired summaries."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from review_experiments_v046 import *

def get_fit(study):
 return json.loads((OUT/f'fits/gain_core__{study}.json').read_text())

def tgc_forward(fit,rows,k,shift):
 d=prepared(rows,fit['process_heads'],'nutrition_compact');a=d['inputs'];x=core(np.array(fit['parameters']),'gain_core');cap,_=coefficients(x,d);cap*=np.exp(shift);p=x[8];feed=np.zeros(len(rows));anchor=a[:,0].copy()
 def weights(t):return (a[:,0]**(1/3)+k*a[:,1]*np.minimum(t,a[:,2]))**3
 def rate(t):
  w=weights(t);capacity=cap*(w/10)**p*10;offer=capacity.copy()
  for i,v in enumerate(a):
   mode=int(v[4]);ci=int(v[9])
   if mode==1:offer[i]=v[8]*cap[ci]*(w[ci]/10)**p*10
   elif mode in (2,3):offer[i]=v[5]*anchor[i]
   elif mode==4:offer[i]=v[8]*anchor[ci]
  return np.minimum(capacity,offer)
 for day in range(int(max(a[:,2]))):
  w=weights(day)
  for i,v in enumerate(a):
   if day%int(v[6])==0:
    mode=int(v[4])
    if mode==2:anchor[i]=w[i]
    elif mode==3 and day>0:anchor[i]*=1+v[7]*v[5]*v[6]
    elif mode==4:anchor[i]=cap[i]*(w[i]/10)**p*10
  increment=(rate(day)+4*rate(day+.5)+rate(day+1))/6
  feed+=increment*(day<a[:,2])
 return {'weight_g':weights(a[:,2]),'feed_g':feed}

def baselines():
 rows,targets,diag=load_data();pp=[];records=[];prefix=''
 for s in sorted(ANCHORS):
  rr=[r for r in rows if r['study_id']==s];fit=get_fit(s);ai=next(i for i,r in enumerate(rr) if r['treatment_id']==ANCHORS[s]);ar=rr[ai]
  k=(target(ar)**(1/3)-val(ar,'w0_g')**(1/3))/(val(ar,'temperature_c')*val(ar,'days'))
  b,rec=calibrate(fit,rr,diag,growth_only=True);pp.extend(prediction_rows('coupled_growth_only',fit,rr,forward(fit,rr,b),'growth'));records.append({'study_id':s,'model':'coupled_growth_only',**rec})
  ss=tgc_forward(fit,rr,k,0);p=prediction_rows('TGC_growth_only',fit,rr,ss,'growth')
  for r in p:r['FI_APE_pct']=''
  pp.extend(p)
  has_channel=bool(ar['observed_feed_g']) or bool(secondary([ar],diag))
  if has_channel:
   def fun(q):return control_residual(tgc_forward(fit,rr,k,float(q[0])),rr,ai,diag)[1:]
   best=min((least_squares(fun,[seed],bounds=([-4.],[4.]),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10) for seed in [0.,-1.,1.]),key=lambda r:r.cost);shift=float(best.x[0]);obj=float(2*best.cost)
  else:shift=0.;obj=0.
  pp.extend(prediction_rows('TGC_plus_intake',fit,rr,tgc_forward(fit,rr,k,shift)));records.append({'study_id':s,'model':'TGC_plus_intake','anchor_id':ar['treatment_id'],'tgc_coefficient':float(k),'capacity_log_shift':shift,'objective':obj,'channel':'FI' if ar['observed_feed_g'] else secondary([ar],diag).get(ar['treatment_id'],('growth_only',0))[0]})
  b=np.array(json.loads((OUT/f'calibrations/gain_core__{s}.json').read_text())['shifts'])
  pp.extend(prediction_rows('coupled_joint',fit,rr,forward(fit,rr,b)))
  pp.extend(prediction_rows('coupled_uncalibrated_matched',fit,rr,forward(fit,rr,[0.,0.]),'uncalibrated'))
  gc=np.log(target(ar)/val(ar,'w0_g'))/val(ar,'days');sg={'weight_g':np.array([val(r,'w0_g')*np.exp(gc*val(r,'days')) for r in rr]),'feed_g':np.zeros(len(rr))};sp=prediction_rows('SGR_growth_only',fit,rr,sg,'growth')
  for r in sp:r['FI_APE_pct']=''
  pp.extend(sp)
 mm,summary=summaries(pp);write(prefix+'baseline_predictions.csv',pp);write(prefix+'baseline_study_metrics.csv',mm);write(prefix+'baseline_summary.csv',summary);save(OUT/(prefix+'baseline_calibrations.json'),records);print(json.dumps(summary),flush=True)

def diagnostics():
 rows,targets,diag=load_data();records=[];sensitivity=[];prefix=''
 for s in sorted(ANCHORS):
  rr=[r for r in rows if r['study_id']==s];fit=get_fit(s);ai=next(i for i,r in enumerate(rr) if r['treatment_id']==ANCHORS[s]);ar=rr[ai]
  anc=json.loads((OUT/f'calibrations/gain_core__{s}.json').read_text());b=np.array(anc['shifts']);h=1e-4
  def fun(v):return control_residual(forward(fit,rr,v),rr,ai,diag)
  jac=np.column_stack([(fun(b+np.eye(2)[j]*h)-fun(b-np.eye(2)[j]*h))/(2*h) for j in range(2)]);sv=np.linalg.svd(jac,compute_uv=False);rank=int(np.sum(sv>max(1e-8,sv[0]*1e-6)));condition=float(sv[0]/sv[-1]) if rank==2 and len(sv)==2 else None
  proxy=secondary([ar],diag).get(ar['treatment_id']);channel='mass + FI' if ar['observed_feed_g'] else 'mass + '+proxy[0] if proxy else 'mass'
  records.append({'study_id':s,'anchor_id':ar['treatment_id'],'channels':channel,'data_rank':rank,'condition_number':condition,'singular_values':sv.tolist(),'capacity_log_shift':float(b[0]),'conversion_logit_shift':float(b[1]),'penalty':anc['penalty'],'at_bound':bool(np.any(abs(b)>4-1e-5)),'control_residual':fun(b).tolist()})
  if s=='Zhang2021':
   for penalty in (0.,.025,.1,.25,1.):
    bb,rec=calibrate(fit,rr,diag,penalty=penalty,seeds=[b.tolist(),[0.,0.],[-1.,0.],[1.,-1.]])
    preds=prediction_rows('Zhang2021',fit,rr,forward(fit,rr,bb));mm,ss=summaries(preds);sensitivity.append({'penalty':penalty,'capacity_log_shift':float(bb[0]),'conversion_logit_shift':float(bb[1]),'control_data_loss':float(np.sum(fun(bb)**2)),'penalized_objective':rec['objective'],**ss[0],'attempts':rec['attempts']})
 save(OUT/(prefix+'identification_diagnostics.json'),records);save(OUT/(prefix+'shrinkage_sensitivity.json'),sensitivity);write(prefix+'control_channels.csv',[{k:v for k,v in r.items() if k not in ['singular_values','control_residual']} for r in records]);print(json.dumps({'rank_counts':{str(i):sum(r['data_rank']==i for r in records) for i in (1,2)},'shrinkage':[{k:v for k,v in r.items() if k in ['penalty','W_MAPE_pct','gain_MAPE_pct']} for r in sensitivity]}),flush=True)

def weighted_quantile(values,weights,q):
 order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order];return float(np.interp(q,np.cumsum(w)/sum(w),v))
def stats():
 pp=read('artifacts/reviewer-revision-v0.46/baseline_predictions.csv');pp+=read('artifacts/reviewer-revision-v0.46/nested_predictions.csv');pp+=read('artifacts/reviewer-revision-v0.46/tabular_predictions.csv');mm,summary=summaries([{**r,**{k:float(r[k]) if r[k]!='' else '' for k in ('W_APE_pct','gain_APE_pct','FI_APE_pct')}} for r in pp]);write('all_study_metrics.csv',mm);write('all_summary.csv',summary)
 rng=np.random.default_rng(20261008);studies=sorted(ANCHORS);index=rng.integers(0,16,(20000,16));intervals=[];differences=[];distributions=[]
 lookup={(r['model'],r['study_id']):r for r in mm}
 for label in sorted({r['model'] for r in pp}):
  for metric in ('W','gain'):
   values=np.array([lookup[(label,s)][metric+'_MAPE_pct'] for s in studies]);boot=values[index].mean(1);low,high=np.percentile(boot,[2.5,97.5]);intervals.append({'model':label,'metric':metric,'mean':float(values.mean()),'lower':float(low),'upper':float(high),'resamples':20000,'unit':'study','n':16})
   rr=[r for r in pp if r['model']==label];vals=[float(r[metric+'_APE_pct']) for r in rr];weights=[1/(16*sum(p['study_id']==r['study_id'] for p in rr)) for r in rr];distributions.append({'model':label,'metric':metric,'weighted_median':weighted_quantile(vals,weights,.5),'weighted_P90':weighted_quantile(vals,weights,.9),'max':float(max(vals)),'treatments':len(rr)})
 for a,b in [('coupled_joint','TGC_plus_intake'),('coupled_growth_only','TGC_growth_only'),('coupled_joint','coupled_uncalibrated_matched'),('gain_core','mass_core'),('gain_core','learned_maintenance'),('gain_core','process_scale'),('nested_selected','TGC_plus_intake'),('coupled_joint','ridge'),('coupled_joint','XGBoost'),('coupled_joint','TabICLv2')]:
  for metric in ('W','gain'):
   diff=np.array([lookup[(a,s)][metric+'_MAPE_pct']-lookup[(b,s)][metric+'_MAPE_pct'] for s in studies]);lo,hi=np.percentile(diff[index].mean(1),[2.5,97.5]);differences.append({'model_A':a,'model_B':b,'metric':metric,'difference_pp':float(diff.mean()),'lower':float(lo),'upper':float(hi),'A_lower_error_studies':int(np.sum(diff<0)),'studies':16})
 write('score_intervals.csv',intervals);write('paired_differences.csv',differences);write('error_distribution.csv',distributions)
 write('intake_source_metrics.csv',[r for r in mm if r['FI_n'] and r['model'] in ('coupled_joint','TGC_plus_intake','nested_selected','ridge','XGBoost','TabICLv2')]);print(json.dumps({'intervals':[r for r in intervals if r['model']=='coupled_joint'],'differences':[r for r in differences if r['model_A']=='coupled_joint' and r['model_B']=='TGC_plus_intake']}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('mode',choices=['baselines','diagnostics','stats'])
 args=parser.parse_args()
 if args.mode != 'stats':setup()
 {'baselines':baselines,'diagnostics':diagnostics,'stats':stats}[args.mode]()
