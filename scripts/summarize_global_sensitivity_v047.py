import csv,math,json
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'artifacts/global-sensitivity-v0.47'
rr=list(csv.DictReader((O/'sobol_indices.csv').open()));r=[x for x in rr if x['N']=='8192']
def aggregate(rows,keys):
 d=defaultdict(list)
 for x in rows:d[tuple(x[k] for k in keys)].append(x)
 out=[]
 for key,vs in d.items():
  y=dict(zip(keys,key));y['n_defined']=sum(math.isfinite(float(v['ST'])) for v in vs);y['n_total']=len(vs)
  for k in ['S1','ST']:
   a=[float(v[k]) for v in vs if math.isfinite(float(v[k]))];y[k]=float(np.mean(a)) if a else float('nan')
  out.append(y)
 return out
def write(name,rows):
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
primary=[x for x in r if x['halfwidth']=='0.25']
a=aggregate(r,['halfwidth','parameter','feeding','metric']);write('aggregate_indices.csv',a)
stage=aggregate([x for x in primary if x['parameter'] in ['capacity','conversion','intercept_pair']],['parameter','feeding','metric','temperature_c','day']);write('temperature_stage_indices.csv',stage)
pairs=[x for x in a if x['parameter'].startswith('pair_') or x['parameter']=='intercept_pair'];rank=[]
for key in sorted(set((x['halfwidth'],x['feeding'],x['metric']) for x in pairs)):
 subset=sorted([x for x in pairs if (x['halfwidth'],x['feeding'],x['metric'])==key],key=lambda x:x['S1'],reverse=True)
 for i,x in enumerate(subset):rank.append(dict(rank=i+1,**x))
write('coordinate_pair_rankings.csv',rank)

z=aggregate([x for x in primary if x['group']=='False'],['parameter','feeding','metric','temperature_c','day']);write('coefficient_temperature_stage_indices.csv',z)
print("Wrote global sensitivity summaries")
