"""Frozen yellow-catfish intake, energy and growth model."""
from pathlib import Path
import csv
import ctypes
import json
import subprocess
import numpy as np
from scipy.special import expit

_LIB = None

def val(r,k,default=0.):
    return float(r[k]) if r.get(k,'')!='' else default

def head_features(rows):
    return np.array([[1.,(val(r,'temperature_c')-27)/6,np.log(val(r,'w0_g')/10),(val(r,'days')-56)/56,
       np.log((val(r,'protein_pct')/val(r,'ge_kj_g'))/(39.88/16.7)),(val(r,'lipid_pct')-4.49)/5,
       (val(r,'met_pct',.8)-.8)/.3,val(r,'plant_replacement'),val(r,'insect_replacement'),max(0.,6-val(r,'do_mg_l'))/3.5] for r in rows])

def head_predict(rows,heads):
    xx=head_features(rows);out={}
    for name,h in heads.items():
        z=xx@np.asarray(h['coefficients']);out[name]=100*expit(z) if h['transform']=='logit' else np.exp(z)
    return out

def compile_engine(directory,source='multisource_fast.c',names=None):
    """Compile one integrator. names are (source,target) pairs to build together."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pairs=names or [(source,Path(source).with_suffix('.dylib').name)]
    built=[]
    for src,out in pairs:
        source_path=Path(__file__).with_name(src);target=directory/out
        subprocess.run(['cc','-O3','-std=c99','-shared','-fPIC',str(source_path),'-o',str(target),'-lm'],check=True,capture_output=True)
        built.append(target)
    return built[0] if len(built)==1 else built

def library(path):
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(path));p=ctypes.POINTER(ctypes.c_double)
        _LIB.aquasim.argtypes=[ctypes.c_int,p,p,p,p,p,p,p,ctypes.c_int,ctypes.c_double,p];_LIB.aquasim.restype=ctypes.c_int
    return _LIB

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

def simulate(x,data,libpath,step=1.):
    if step<=0 or abs(1/step-round(1/step))>1e-9:
        raise ValueError('Step must divide one day')
    if np.any(data['inputs'][:,0]<=0) or np.any(data['inputs'][:,2]<0) or np.any(data['inputs'][:,6]<1):
        raise ValueError('Invalid mass/horizon/update inputs')
    x=np.ascontiguousarray(x,dtype=float);cap,eta=coefficients(x,data)
    aa=[data['inputs'],cap,eta,data['adc'],data['urine'],data['rho'],x]
    aa=[np.ascontiguousarray(a,dtype=float) for a in aa];out=np.empty((len(data['rows']),7))
    p=ctypes.POINTER(ctypes.c_double)
    result=library(libpath).aquasim(len(out),*[a.ctypes.data_as(p) for a in aa],len(x),step,out.ctypes.data_as(p))
    if result or np.any(out[:,0]<=0) or np.any(~np.isfinite(out)):raise ValueError('ODE mass/energy domain exhausted')
    closure=out[:,2]-out[:,3]-out[:,4]-out[:,5]-out[:,6]-data['rho']*(out[:,0]-data['inputs'][:,0])
    return dict(weight_g=out[:,0],feed_g=out[:,1],energy=out[:,2:],energy_balance_error_kj=closure)


def load_inputs(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('The input CSV is empty')
    required = ['study_id', 'treatment_id', 'w0_g', 'days', 'temperature_c',
                'ge_kj_g', 'protein_pct', 'lipid_pct', 'feeding']
    modes = {'satiation', 'relative_control', 'mass_updated',
             'design_schedule', 'block_relative_control'}
    defaults = dict(offer_fraction=0., update_days=14., design_fce=0.,
                    relative_ration=1., do_mg_l=6.5)
    identifiers = set()
    for row in rows:
        for key in required:
            if not row.get(key, '').strip():
                raise ValueError('Missing input ' + key)
        key = (row['study_id'], row['treatment_id'])
        if key in identifiers:
            raise ValueError('Duplicate study/treatment ID: ' + str(key))
        identifiers.add(key)
        if row['feeding'] not in modes:
            raise ValueError('Unknown feeding mode: ' + row['feeding'])
        for key in required[2:8]:
            if not np.isfinite(float(row[key])):
                raise ValueError('Nonfinite input ' + key)
        if min(val(row, 'w0_g'), val(row, 'ge_kj_g'), val(row, 'protein_pct')) <= 0:
            raise ValueError('Initial mass, gross energy and protein must be positive')
        days = val(row, 'days')
        if days < 1 or days != int(days):
            raise ValueError('days must be a positive integer')
        if row['feeding'] in {'mass_updated', 'design_schedule'} and not row.get('offer_fraction'):
            raise ValueError('This feeding mode requires offer_fraction')
        if row['feeding'] == 'design_schedule' and not row.get('design_fce'):
            raise ValueError('design_schedule requires design_fce')
        for key, value in defaults.items():
            if row.get(key, '') == '':
                row[key] = value
            if not np.isfinite(float(row[key])):
                raise ValueError('Nonfinite input ' + key)
        update = val(row, 'update_days')
        if update < 1 or update != int(update):
            raise ValueError('update_days must be a positive integer')
        if min(val(row, 'offer_fraction'), val(row, 'relative_ration'), val(row, 'design_fce')) < 0:
            raise ValueError('Feeding amounts and conversion coefficients must be nonnegative')
    for study in {r['study_id'] for r in rows}:
        group = [r for r in rows if r['study_id'] == study]
        if any(r['feeding'] in {'relative_control', 'block_relative_control'} for r in group):
            controls = [r for r in group if r['feeding'] in {'relative_control', 'block_relative_control'}
                        and val(r, 'relative_ration') == 1.]
            if len(controls) != 1:
                raise ValueError('Relative feeding requires exactly one control per study')
    return rows


def load_model(path):
    model = json.loads(Path(path).read_text())
    core = np.asarray(model['core_parameters'], dtype=float)
    if core.shape != (14,) or not np.all(np.isfinite(core)):
        raise ValueError('Expected 14 finite canonical core coefficients')
    return model, core


def apply_source_shifts(data, shifts):
    missing = {r['study_id'] for r in data['rows']} - set(shifts)
    if missing:
        raise ValueError('Source shifts missing for: ' + ', '.join(sorted(missing)))
    data['capacity_log_offset'] = np.array([shifts[r['study_id']]['capacity_log_offset'] for r in data['rows']])
    data['eta_offset'] = np.array([shifts[r['study_id']]['eta_offset'] for r in data['rows']])


def write_predictions(path, rows, result, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(study_id=r['study_id'], treatment_id=r['treatment_id'],
                    predicted_weight_g=float(result['weight_g'][i]),
                    predicted_feed_g=float(result['feed_g'][i]),
                    effective_ADC=float(data['adc'][i]),
                    effective_body_energy_kj_g=float(data['rho'][i]),
                    energy_balance_abs_kj=float(abs(result['energy_balance_error_kj'][i])))
               for i, r in enumerate(rows)]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
