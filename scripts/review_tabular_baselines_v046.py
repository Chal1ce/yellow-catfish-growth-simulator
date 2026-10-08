"""Input-only regression baselines, with matched completed-control calibration."""
import argparse, json, sys, time, importlib.metadata
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from review_experiments_v046 import *

NUMERIC = ['temperature_c','w0_g','days','protein_pct','lipid_pct','ge_kj_g',
 'offer_fraction','update_days','design_fce','relative_ration','do_mg_l','met_pct',
 'arg_pct','available_p_pct','plant_replacement','insect_replacement','feather_replacement',
 'rutin_mg_kg','simple_sugar','corn_starch','selenium_inorganic','selenium_yeast',
 'selenium_spirulina','hydrolysate_fraction']
FEEDING = ['satiation','relative_control','mass_updated','design_schedule','block_relative_control']

def features(rows):
 return np.array([[float(r[k]) if r.get(k,'')!='' else np.nan for k in NUMERIC]+
                 [float(r['feeding']==v) for v in FEEDING] for r in rows])

def observed_feed(r,diag):
 if r['observed_feed_g']:return val(r,'observed_feed_g'),'FI'
 px=secondary([r],diag).get(r['treatment_id'])
 if px is None:return None,'none'
 kind,y=px
 if kind=='FR':return y/100*val(r,'days')*(val(r,'w0_g')+target(r))/2,'FR'
 if kind=='FCR':return y*(target(r)-val(r,'w0_g')),'FCR'
 raise ValueError(kind)

def factory(name,settings=None):
 settings=settings or {}
 if name=='ridge':return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=settings.get('alpha',1.)))
 if name=='XGBoost':
  from xgboost import XGBRegressor
  return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),XGBRegressor(n_estimators=200,max_depth=settings.get('depth',2),learning_rate=.03,min_child_weight=1,reg_lambda=settings.get('lambda',1.),subsample=1.,colsample_bytree=1.,random_state=20261008,n_jobs=2,objective='reg:squarederror'))
 from tabicl import TabICLRegressor
 return TabICLRegressor(n_estimators=8,device='cpu',n_jobs=2,random_state=20261008,verbose=False)

def fit_predict(name,train,test,y,settings=None):
 model=factory(name,settings);xx=features(train);tt=features(test)
 if name=='TabICLv2':model.fit(xx,y)
 else:
  counts={s:sum(r['study_id']==s for r in train) for s in {r['study_id'] for r in train}}
  weights=np.array([1/counts[r['study_id']] for r in train]);weights*=len(weights)/weights.sum()
  model.fit(xx,y,**{('ridge' if name=='ridge' else 'xgbregressor')+'__sample_weight':weights})
 return np.asarray(model.predict(tt))

def predict_block(name,train,test,diag,settings=None):
 base=np.array([val(r,'w0_g')*val(r,'days') for r in test]);yg=np.log([(target(r)-val(r,'w0_g'))/(val(r,'w0_g')*val(r,'days')) for r in train]);pg=np.exp(fit_predict(name,train,test,yg,settings))*base
 ft=[r for r in train if observed_feed(r,diag)[0] is not None];yf=np.log([observed_feed(r,diag)[0]/(val(r,'w0_g')*val(r,'days')) for r in ft]);pf=np.exp(fit_predict(name,ft,test,yf,settings))*base
 pp=[];records=[]
 for s in sorted({r['study_id'] for r in test}):
  ix=[i for i,r in enumerate(test) if r['study_id']==s];rr=[test[i] for i in ix];ai=next(i for i in ix if test[i]['treatment_id']==ANCHORS[s]);ar=test[ai];fg=(target(ar)-val(ar,'w0_g'))/pg[ai];fc,channel=observed_feed(ar,diag);ff=fc/pf[ai] if fc is not None else 1.
  pp.extend(prediction_rows(name,{},rr,{'weight_g':np.array([val(r,'w0_g') for r in rr])+pg[ix]*fg,'feed_g':pf[ix]*ff}));records.append({'model':name,'study_id':s,'anchor_id':ar['treatment_id'],'gain_multiplier':float(fg),'intake_multiplier':float(ff),'control_intake_channel':channel,'training_growth_n':len(train),'training_feed_n':len(ft),'settings':settings or 'default'})
 return pp,records

def tune_classical():
 rows,targets,diag=load_data();splits=json.loads((OUT/'nested_protocol.json').read_text())['inner_blocks'];results=[];settings=[]
 for name in ['ridge','XGBoost']:
  grid=[{'alpha':a} for a in [.1,1.,10.,100.]] if name=='ridge' else [{'depth':d,'lambda':l} for d in [1,2,3] for l in [1.,10.]]
  for outer in sorted(ANCHORS):
   cache=OUT/'tabular_tuned_cache'/f'{name}__{outer}.json'
   if cache.exists():r=json.loads(cache.read_text());results.extend(r['predictions']);settings.append(r['selection']);continue
   scores=[]
   for cfg in grid:
    inner=[]
    for block in splits[outer]:
     train=[r for r in rows if r['study_id']!=outer and r['study_id'] not in block];test=[r for r in rows if r['study_id'] in block];pp,_=predict_block(name,train,test,diag,cfg);inner.extend(pp)
    z=summaries(inner)[1][0];scores.append({'settings':cfg,'score':.5*(z['gain_MAPE_pct']+z['FI_MAPE_pct'])})
   chosen=min(scores,key=lambda r:r['score']);pp,cal=predict_block(name,[r for r in rows if r['study_id']!=outer],[r for r in rows if r['study_id']==outer],diag,chosen['settings']);sel={'model':name,'outer':outer,'selected':chosen,'inner_scores':scores,'calibration':cal};save(cache,{'predictions':pp,'selection':sel});results.extend(pp);settings.append(sel)
   print(json.dumps({'tuned':name,'study':outer,'settings':chosen}),flush=True)
 # TabICLv2 uses its pretrained default; classical hyperparameters are selected in inner study blocks.
 results.extend(r for r in read('artifacts/reviewer-revision-v0.46/tabular_predictions.csv') if r['model']=='TabICLv2')
 for r in results:
  for k in ['W_APE_pct','gain_APE_pct','FI_APE_pct']:
   if r[k]!='':r[k]=float(r[k])
 mm,ss=summaries(results)
 for name in ['tabular_predictions.csv','tabular_study_metrics.csv','tabular_summary.csv']:
  p=OUT/name;p.rename(p.with_name('fixed_'+name))
 write('tabular_predictions.csv',results);write('tabular_study_metrics.csv',mm);write('tabular_summary.csv',ss);save(OUT/'tabular_inner_selection.json',settings);protocol=json.loads((OUT/'tabular_baseline_protocol.json').read_text());protocol['classical_tuning']='3 inner study blocks matching the mechanistic nested protocol; gain and FI macro errors weighted 0.5 each; ridge alpha in [0.1,1,10,100]; XGBoost depth in [1,2,3], lambda in [1,10]; outer outcomes excluded from selection';save(OUT/'tabular_baseline_protocol.json',protocol);print(json.dumps(ss),flush=True)

def main():
 rows,targets,diag=load_data();names=['TabICLv2','ridge','XGBoost'];results=[];records=[];t=time.time()
 protocol={'models':names,'features':NUMERIC+['feeding='+v for v in FEEDING],
  'target':'log increment per initial mass per day; log cumulative feed per initial mass per day',
  'training_FI':'reported FI preferred; otherwise one FR or FCR converted using training outcomes only',
  'calibration':'multiplicative endpoint gain adjustment and FI adjustment, each from the single designated completed control; no FI adjustment without a control intake/proxy',
  'split':'16 leave-one-study-out folds; all imputation/scaling/regression fitted on other studies; no study IDs or target final outcomes in features',
  'hyperparameters':'fixed before evaluation; ridge alpha=1; XGBoost depth=2, 200 trees, rate=0.03; TabICLv2 default 8 ensemble members, CPU, checkpoint tabicl-regressor-v2-20260212.ckpt',
  'source_weighting':'equal study weights for ridge and XGBoost; native unweighted in-context training for TabICLv2',
  'seed':20261008,'versions':{p:importlib.metadata.version(p) for p in ['scikit-learn','xgboost','tabicl','torch']}}
 save(OUT/'tabular_baseline_protocol.json',protocol)
 for name in names:
  for study in sorted(ANCHORS):
   cache=OUT/'tabular_cache'/f'{name}__{study}.json'
   if cache.exists():record=json.loads(cache.read_text());results.extend(record['predictions']);records.append(record['calibration']);continue
   train=[r for r in rows if r['study_id']!=study];test=[r for r in rows if r['study_id']==study];ai=next(i for i,r in enumerate(test) if r['treatment_id']==ANCHORS[study]);ar=test[ai]
   base=np.array([val(r,'w0_g')*val(r,'days') for r in test])
   yg=np.log([(target(r)-val(r,'w0_g'))/(val(r,'w0_g')*val(r,'days')) for r in train]);pg=np.exp(fit_predict(name,train,test,yg))*base
   fg=(target(ar)-val(ar,'w0_g'))/pg[ai];pg*=fg
   ft=[r for r in train if observed_feed(r,diag)[0] is not None];yf=np.log([observed_feed(r,diag)[0]/(val(r,'w0_g')*val(r,'days')) for r in ft]);pf=np.exp(fit_predict(name,ft,test,yf))*base
   fc,channel=observed_feed(ar,diag);ff=fc/pf[ai] if fc is not None else 1.;pf*=ff
   ss={'weight_g':np.array([val(r,'w0_g') for r in test])+pg,'feed_g':pf};pred=prediction_rows(name,{},test,ss)
   rec={'model':name,'study_id':study,'anchor_id':ar['treatment_id'],'gain_multiplier':float(fg),'intake_multiplier':float(ff),'control_intake_channel':channel,'training_growth_n':len(train),'training_feed_n':len(ft)}
   save(cache,{'predictions':pred,'calibration':rec});results.extend(pred);records.append(rec);print(json.dumps({'model':name,'study':study,'seconds':round(time.time()-t,1)}),flush=True)
 mm,summary=summaries(results);write('tabular_predictions.csv',results);write('tabular_study_metrics.csv',mm);write('tabular_summary.csv',summary);save(OUT/'tabular_calibrations.json',records);print(json.dumps(summary),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--tune',action='store_true',help='Select ridge and XGBoost settings in the saved inner study blocks.')
 args=parser.parse_args()
 tune_classical() if args.tune else main()
