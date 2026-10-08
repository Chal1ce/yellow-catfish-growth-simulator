"""Input preparation, chemistry heads, bounds and control targets for v0.46 experiments.

Functions are taken from multisource.py and run_multisource_adaptation_v018.py.
The thermal bounds are the original joint_v051/asymmetric_eta bounds.
"""
import numpy as np
from scipy.special import expit, logit

def thermal_bounds(candidate):
    if candidate != 'asymmetric_eta':
        raise ValueError('The revision experiment uses asymmetric_eta bounds')
    return (np.array([-5., -1.5, -2., -5., -3., -5., -1.5, -5.]),
            np.array([0., 1.5, 0., 4., 3., 0., 0., 0.]))

def val(r,k,default=0.):
    return float(r[k]) if r.get(k,'')!='' else default

def head_features(rows):
    return np.array([[1.,(val(r,'temperature_c')-27)/6,np.log(val(r,'w0_g')/10),(val(r,'days')-56)/56,
       np.log((val(r,'protein_pct')/val(r,'ge_kj_g'))/(39.88/16.7)),(val(r,'lipid_pct')-4.49)/5,
       (val(r,'met_pct',.8)-.8)/.3,val(r,'plant_replacement'),val(r,'insect_replacement'),max(0.,6-val(r,'do_mg_l'))/3.5] for r in rows])

def fit_heads(rows,targets):
    """Fit every head only on training papers, with a fixed ridge penalty .1."""
    byid={r['treatment_id']:r for r in rows};out={}
    for metric in sorted(set(a['metric'] for a in targets)):
        aa=[a for a in targets if a['metric']==metric and a['treatment_id'] in byid]
        if not aa:continue
        xx=head_features([byid[a['treatment_id']] for a in aa]);yy=np.array([val(a,'value') for a in aa])
        transform='logit' if metric.startswith('adc_') or metric=='urinary_fraction_pct' else 'log'
        yy=logit(np.clip(yy/100,1e-5,1-1e-5)) if transform=='logit' else np.log(yy)
        weights=np.array([1/sum(b['study_id']==a['study_id'] for b in aa) for a in aa])
        # Component-derived energy/rho proxies get half weight within their source.
        weights*=np.array([.5 if a['kind'].startswith('component-calculated') else 1. for a in aa])
        pen=np.diag([0.]+[.1]*(xx.shape[1]-1));gram=xx.T@(weights[:,None]*xx)
        coef=np.linalg.solve(gram+pen,xx.T@(weights*yy))
        out[metric]=dict(coefficients=coef.tolist(),transform=transform,n=len(aa),
                         studies=sorted(set(a['study_id'] for a in aa)),effective_df=float(np.trace(np.linalg.solve(gram+pen,gram))))
    return out

def head_predict(rows,heads):
    xx=head_features(rows);out={}
    for name,h in heads.items():
        z=xx@np.asarray(h['coefficients']);out[name]=100*expit(z) if h['transform']=='logit' else np.exp(z)
    return out

def prepared(rows,heads,candidate):
    modes={'satiation':0,'relative_control':1,'mass_updated':2,'design_schedule':3,'block_relative_control':4,
           'absolute_feed':5}
    controls={r['study_id']:i for i,r in enumerate(rows) if r['feeding'] in ['relative_control','block_relative_control'] and val(r,'relative_ration')==1}
    inp=np.array([[val(r,'w0_g'),val(r,'temperature_c'),val(r,'days'),val(r,'ge_kj_g'),modes[r['feeding']],val(r,'offer_fraction'),val(r,'update_days'),
                   val(r,'design_fce'),val(r,'relative_ration'),controls.get(r['study_id'],i)] for i,r in enumerate(rows)],float)
    pred=head_predict(rows,heads)
    use=candidate!='all_source_base'
    adc=np.clip(pred.get('adc_energy_pct',np.full(len(rows),80.))/100,.35,.98) if use else np.full(len(rows),.8)
    rho=np.clip(pred.get('body_energy_kj_g',np.full(len(rows),4.8)),2.,12.) if use else np.full(len(rows),4.8)
    u=np.clip(pred.get('urinary_fraction_pct',np.full(len(rows),7.))/100,.001,.25) if use else np.full(len(rows),.07)
    dp=pred.get('adc_protein_pct',np.full(len(rows),90.))/100 if use else np.full(len(rows),.9)
    feats=[]
    for i,r in enumerate(rows):
        met=(val(r,'met_pct',.8)-.8)/.3;arg=(val(r,'arg_pct',2.8)-2.8)/.5;ph=val(r,'available_p_pct',.8)-.8
        feats.append([np.log((val(r,'protein_pct')*dp[i]/(val(r,'ge_kj_g')*adc[i]))/(39.88*.9/(16.7*.8))),met,met*met,
           max(0.,6-val(r,'do_mg_l'))/3.5,arg,arg*arg,val(r,'plant_replacement'),val(r,'insect_replacement'),val(r,'feather_replacement'),
           ph,ph*ph,np.log1p(val(r,'rutin_mg_kg')/100),val(r,'simple_sugar'),val(r,'corn_starch'),val(r,'selenium_inorganic'),
           val(r,'selenium_yeast'),val(r,'selenium_spirulina'),val(r,'hydrolysate_fraction')])
    return dict(inputs=inp,adc=adc,rho=rho,urine=u,features=np.array(feats),rows=rows)

def coefficients(x,data):
    a=data['inputs'];z=(a[:,1]-27)/6
    cap=np.exp(x[0]+x[1]*z+x[2]*z*z)
    eta=x[3]+x[4]*z+np.where(z>0,x[7],x[5])*z*z
    if len(x)>8:
        cap*=np.exp(x[9]*np.log(16.7/a[:,3]));eta+=data['features'][:,:len(x)-10]@x[10:]
    cap*=np.exp(data.get('capacity_log_offset',0.));eta+=data.get('eta_offset',0.)
    return cap,eta

def candidate_bounds(candidate):
    lo,hi=thermal_bounds('asymmetric_eta')
    if candidate in ['nutrition_compact','nutrition_full']:
        n=6 if candidate=='nutrition_compact' else 20
        lo=np.r_[lo,.5,[-2.]*(n-1)];hi=np.r_[hi,1.1,[2.]*(n-1)]
        # Hypoxia and quadratic deficiency response are monotone/nonpositive priors.
        for idx in [12,13]+([15,20] if candidate=='nutrition_full' else []):hi[idx]=0.
    return lo,hi

ANCHORS={'Zhang2011':'Z11_R100','Zhang2017':'Z17_T27_M','Dong2017':'Dong2017_Satiation','Chen2023Se':'Chen2023Se_Control',
         'Zhao2020':'Zhao2020_CS','Chen2014':'Chen2014_D0.55','Chen2016':'Chen2016_A2.44','Hu2017':'Hu2017_Control',
         'Han2022':'Han2022_Control','Liu2023':'Liu2023_R0','Wang2025':'Wang2025_N26','Tang2012':'Tang2012_P0.33',
         'Jiang2004':'Jiang2004_D1','Zhang2021':'Zhang2021_F1','Zheng2023':'Zheng2023_CON','Zheng2025':'Zheng2025_FM'}

def secondary(rows,diag):
    out={}
    for r in rows:
        if r['observed_feed_g']!='':continue  # Do not duplicate directly available FI.
        dd={a['metric']:val(a,'value') for a in diag if a['treatment_id']==r['treatment_id']}
        if 'feeding_rate_pct_bw_day' in dd:out[r['treatment_id']]=('FR',dd['feeding_rate_pct_bw_day'])
        elif 'fcr_g_g' in dd:out[r['treatment_id']]=('FCR',dd['fcr_g_g'])
    # v0.15 CSV uses feeding_rate_pct_bw_day too; one endpoint per group.
    return out

def proxy_residuals(sim,rows,proxy):
    out=[]
    for study in dict.fromkeys(r['study_id'] for r in rows):
        ii=[i for i,r in enumerate(rows) if r['study_id']==study and r['treatment_id'] in proxy]
        for i in ii:
            r=rows[i];kind,value=proxy[r['treatment_id']];w0=val(r,'w0_g');wf=sim['weight_g'][i];fi=sim['feed_g'][i]
            pred=100*fi/(val(r,'days')*(w0+wf)/2) if kind=='FR' else fi/max(wf-w0,1e-7)
            # Composite endpoint loss, quarter of direct-FI weight; not independent raw-feed observations.
            out.append(.5*np.log(max(pred,1e-12)/value)/np.sqrt(len(ii)))
    return out
