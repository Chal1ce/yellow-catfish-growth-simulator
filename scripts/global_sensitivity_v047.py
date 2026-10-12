"""Sobol sensitivity of frozen v0.46; no refitting or observed-outcome use."""
from pathlib import Path
import sys,json,csv,itertools,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1]
PUB=ROOT if (ROOT/'models/yellow_catfish.json').exists() else ROOT/'release/public_repo'
sys.path.insert(0,str(PUB/'src'))
from aquabiosim.model import prepared,simulate,compile_engine
from aquabiosim.experiment_support import candidate_bounds
OUT=ROOT/'artifacts/global-sensitivity-v0.47'
D=9
GROUPS={'capacity':[0,1,2,8],'conversion':[3,4,5,6,7],'intercept_pair':[0,3]}
PAIRS={f'pair_{i}_{j}':[i,j] for i,j in itertools.combinations(range(D),2) if [i,j]!=[0,3]}
def indices(a,b,c):
 v=np.var(np.concatenate([a,b]),axis=0,ddof=0)
 scale=np.maximum(np.mean(np.concatenate([a,b])**2,axis=0),1.)
 valid=v>1e-18*scale
 den=np.where(valid,2*v,np.nan)
 return 1-np.mean((b-c)**2,axis=0)/den,np.mean((a-c)**2,axis=0)/den
def make_data(model):
 base=next(r for r in csv.DictReader((ROOT/'data/processed/multisource_v0.18/treatments.csv').open(encoding='utf-8-sig')) if r['treatment_id']=='Z17_T27_M')
 rows=[];meta=[]
 for T,W,feed in itertools.product([21.,27.,33.],[6.,16.,35.],['satiation','mass_updated']):
  r=dict(base);r.update(study_id=f'scenario_{T}_{W}_{feed}',treatment_id=f'scenario_{T}_{W}_{feed}',temperature_c=T,w0_g=W,days=42,feeding=feed,offer_fraction=.02,update_days=14)
  rows.append(r)
 d=prepared(rows,model['process_heads'],'nutrition_compact')
 # Chemistry is calculated once for each complete 42-day scenario, then kept fixed at all checkpoints.
 for i,r in enumerate(rows):
  for day in [7,21,42]:meta.append(dict(scenario=i,temperature_c=r['temperature_c'],w0_g=r['w0_g'],feeding=r['feeding'],day=day,adc=float(d['adc'][i]),urine=float(d['urine'][i]),rho=float(d['rho'][i])))
 dd={k:(np.repeat(v,3,axis=0) if isinstance(v,np.ndarray) else [r for r in rows for _ in range(3)]) for k,v in d.items()}
 dd['inputs'][:,2]=np.tile([7,21,42],len(rows));dd['inputs'][:,9]=np.arange(len(meta))
 return dd,meta

def evaluate(job):
 halfwidth,seed,power=job;N=2**power
 model=json.loads((PUB/'models/yellow_catfish.json').read_text());core=np.array(model['core_parameters']);data,meta=make_data(model)
 lo,hi=candidate_bounds('nutrition_compact');lower=np.maximum(lo[:D],core[:D]-halfwidth);upper=np.minimum(hi[:D],core[:D]+halfwidth)
 design=qmc.Sobol(2*D,scramble=True,seed=seed).random_base2(power);a=lower+design[:,:D]*(upper-lower);b=lower+design[:,D:]*(upper-lower)
 mats=[a,b];labels=[]
 for j in range(D):
  c=a.copy();c[:,j]=b[:,j];mats.append(c);labels.append((model['core_parameter_names'][j],False,[j]))
 for label,ii in {**GROUPS,**PAIRS}.items():
  c=a.copy();c[:,ii]=b[:,ii];mats.append(c);labels.append((label,True,ii))
 lib=OUT/'runtime/multisource_fast.dylib';ys=[];maximum_closure=0.;begin=time.time()
 for block,mat in enumerate(mats):
  out=np.empty((N,len(meta),2))
  for i,q in enumerate(mat):
   x=core.copy();x[:D]=q
   sim=simulate(x,data,lib);out[i,:,0]=sim['weight_g']-data['inputs'][:,0];out[i,:,1]=sim['feed_g'];maximum_closure=max(maximum_closure,float(np.max(np.abs(sim['energy_balance_error_kj']))))
  ys.append(out)
  print(f'width={halfwidth} seed={seed} block={block+1}/{len(mats)} elapsed={time.time()-begin:.1f}s',flush=True)
 ys=np.array(ys);np.savez_compressed(OUT/f'samples_w{halfwidth}_seed{seed}.npz',responses=ys,A=a,B=b)
 records=[]
 for prefix in ([512,1024,N] if N>=2048 else [N]):
  for k,(name,group,ii) in enumerate(labels):
   s1,st=indices(ys[0,:prefix],ys[1,:prefix],ys[k+2,:prefix])
   for m,metadata in enumerate(meta):
    for o,metric in enumerate(['gain_g','cumulative_intake_g']):
     records.append(dict(halfwidth=halfwidth,seed=seed,N=prefix,parameter=name,group=group,metric=metric,S1=float(s1[m,o]),ST=float(st[m,o]),**metadata))
 # Reference prediction exactly uses same engine and scenario setup.
 sim=simulate(core,data,lib);nominal=[dict(**r,gain_g=float(sim['weight_g'][i]-data['inputs'][i,0]),cumulative_intake_g=float(sim['feed_g'][i])) for i,r in enumerate(meta)]
 return records,dict(halfwidth=halfwidth,seed=seed,N=N,model_calls=N*len(mats),scenario_trajectories=N*len(mats)*len(meta),failed_calls=0,max_energy_closure=maximum_closure,bounds=[dict(parameter=model['core_parameter_names'][j],lower=float(lower[j]),upper=float(upper[j]),frozen=float(core[j])) for j in range(D)]),nominal

def writecsv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_estimator():
 x=qmc.Sobol(6,scramble=True,seed=91).random_base2(15)*2*np.pi-np.pi;a,b=x[:,:3],x[:,3:]
 def f(z):return np.sin(z[:,0])+7*np.sin(z[:,1])**2+.1*z[:,2]**4*np.sin(z[:,0])
 fa,fb=f(a),f(b);ss=[];tt=[]
 for j in range(3):
  c=a.copy();c[:,j]=b[:,j];s,t=indices(fa,fb,f(c));ss.append(float(s));tt.append(float(t))
 assert np.max(abs(np.array(ss)-[.3139,.4424,0]))<.015
 assert np.max(abs(np.array(tt)-[.5576,.4424,.2437]))<.015
 return dict(function='Ishigami',N=32768,S1=ss,ST=tt,pass_=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True);compile_engine(OUT/'runtime')
 protocol=dict(model='v0.46 gain_core frozen',method='Sobol pick-freeze; Jansen difference estimators for first and total indices',primary_halfwidth=.25,primary_N=8192,independent_scrambles=2,seeds=[20261012,20261013],range_checks=[.1,.5],range_check_N=8192,distribution='Independent uniform intervals centered on frozen coefficients, intersected with original fitting bounds. Not a fitted posterior.',outputs=['biomass gain','cumulative intake'],T=[21,27,33],W0=[6,16,35],checkpoints=[7,21,42],feeding=['satiation','2% predicted body mass per day, mass updated every 14 days'],chemistry='Frozen heads predict each full 42-day scenario once; ADCe/u/rho held fixed at every checkpoint and throughout coefficient perturbations.',source_context='No source offsets applied. bs equals alpha0 increment; cs equals theta0 increment. Do not vary duplicate intercepts simultaneously.',groups=GROUPS,coordinate_pairs=PAIRS,degenerate_variance_threshold='variance <= 1e-18 * max(E[Y^2], 1): index undefined, retained as NaN',claim='Variance attribution conditional on specified ranges and scenarios; not proof of optimal calibration or parameter identifiability.',method_source='https://doi.org/10.1016/j.cpc.2009.09.018',estimator_check=check_estimator(),model_sha256=hashlib.sha256((PUB/'models/yellow_catfish.json').read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 jobs=[(.25,20261012,13),(.25,20261013,13),(.1,20261012,13),(.5,20261012,13)]
 rows=[];runs=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for rr,run,nominal in pool.map(evaluate,jobs):rows+=rr;runs.append(run)
 writecsv(OUT/'sobol_indices.csv',rows);writecsv(OUT/'nominal_scenarios.csv',nominal)
 (OUT/'run_summary.json').write_text(json.dumps(runs,indent=2));print('Completed',sum(r['model_calls'] for r in runs),'model calls',flush=True)
if __name__=='__main__':main()
