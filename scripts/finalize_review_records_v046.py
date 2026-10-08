"""Export development and uncalibrated endpoint records from saved revision fits."""
import json
from review_experiments_v046 import *

def main():
 setup()
 rows,targets,diag=load_data();fit=json.loads((OUT/'fits/gain_core__all.json').read_text());pp=[]
 for s in sorted(ANCHORS):
  rr=[r for r in rows if r['study_id']==s];f=json.loads((OUT/f'fits/gain_core__{s}.json').read_text());ss=forward(f,rr,[0.,0.])
  # Include completed controls in the input-only evaluation.
  for i,r in enumerate(rr):
   w0=val(r,'w0_g');w=float(ss['weight_g'][i]);fi=float(ss['feed_g'][i]);pp.append({'model':'uncalibrated_all','study_id':s,'treatment_id':r['treatment_id'],'observed_weight_g':target(r),'predicted_weight_g':w,'initial_weight_g':w0,'observed_feed_g':r['observed_feed_g'],'predicted_feed_g':fi,'W_APE_pct':100*abs(w/target(r)-1),'gain_APE_pct':100*abs((w-w0)/(target(r)-w0)-1),'FI_APE_pct':100*abs(fi/val(r,'observed_feed_g')-1) if r['observed_feed_g'] else ''})
 mm,ss=summaries(pp);write('uncalibrated_all_predictions.csv',pp);write('uncalibrated_all_study_metrics.csv',mm);write('uncalibrated_all_summary.csv',ss)
 ff=prepared(rows,fit['process_heads'],'nutrition_compact');q=np.array(fit['parameters']);ix=np.array([fit['study_effect_studies'].index(r['study_id']) for r in rows]);ff.update(capacity_log_offset=q[9:][::2][ix],eta_offset=q[9:][1::2][ix]);result=sim(core(q,'gain_core'),ff)
 full=[]
 for i,r in enumerate(rows):
  w0=val(r,'w0_g');w=float(result['weight_g'][i]);fi=float(result['feed_g'][i]);full.append({'model':'full_development','study_id':r['study_id'],'treatment_id':r['treatment_id'],'observed_weight_g':target(r),'predicted_weight_g':w,'initial_weight_g':w0,'observed_feed_g':r['observed_feed_g'],'predicted_feed_g':fi,'W_APE_pct':100*abs(w/target(r)-1),'gain_APE_pct':100*abs((w-w0)/(target(r)-w0)-1),'FI_APE_pct':100*abs(fi/val(r,'observed_feed_g')-1) if r['observed_feed_g'] else ''})
 mm,fs=summaries(full);write('full_development_predictions.csv',full);write('full_development_study_metrics.csv',mm);write('full_development_summary.csv',fs)

if __name__=='__main__':main()
