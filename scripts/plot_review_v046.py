"""Updated primary plots and paired-score intervals from saved revision fits."""
from pathlib import Path
import csv,json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_experiments_v046 import *
import experiment_plot_support as dynamic
FIG=ROOT/'outputs/figures_v0.46'
plt.rcParams.update({'font.sans-serif':['DejaVu Sans'],'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def save(fig,name,multi=False):
 fig.canvas.draw()
 for suffix in ['pdf','svg']:fig.savefig(FIG/(name+'.'+suffix),facecolor='white')
 fig.savefig(FIG/(name+'.png'),dpi=600,facecolor='white')
 fig.savefig(FIG/(name+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
 plt.close(fig)

dynamic.save=save

def trajectories():
 rows,targets,diag=load_data();obs=read('data/figure_source_data/trajectory_observation_source_data.csv');records=[]
 for r in obs:
  for k in ['days','initial_weight_g','final_weight_g','initial_weight_se_g','final_weight_se_g','cumulative_feed_target_g','temperature_c']:
   if r[k]!='':r[k]=float(r[k])
  for k in ['displayed','is_calibration_control']:r[k]=r[k]=='True'
 for study in ['Chen2014','Zhang2017']:
  rr=[r for r in rows if r['study_id']==study];fit=json.loads((OUT/f'fits/gain_core__{study}.json').read_text());b=json.loads((OUT/f'calibrations/gain_core__{study}.json').read_text())['shifts'];horizon=int(max(val(r,'days') for r in rr))
  for stage,shifts in [('loso',[0.,0.]),('information_matched_anchor',b)]:
   data=prepared(rr,fit['process_heads'],'nutrition_compact');data.update(capacity_log_offset=shifts[0],eta_offset=shifts[1]);ww=[];ff=[]
   for day in range(horizon+1):
    data['inputs'][:,2]=np.minimum(day,[val(r,'days') for r in rr]);ss=sim(core(np.array(fit['parameters']),'gain_core'),data);ww.append(ss['weight_g']);ff.append(ss['feed_g'])
   ww=np.asarray(ww);ff=np.asarray(ff);daily=np.diff(ff,axis=0)
   for i,r in enumerate(rr):
    for day in range(horizon+1):records.append({'study_id':study,'treatment_id':r['treatment_id'],'stage':stage,'displayed':study=='Chen2014' or r['treatment_id'].endswith('_M'),'is_calibration_control':r['treatment_id']==ANCHORS[study],'day':day,'weight_g':float(ww[day,i]),'cumulative_feed_g':float(ff[day,i]),'daily_intake_g':float(daily[day,i]) if day<horizon else '', 'daily_intake_interval_start':day if day<horizon else '', 'daily_intake_interval_end':day+1 if day<horizon else '', 'capacity_log_offset':shifts[0],'conversion_logit_offset':shifts[1]})
 for name,rr in [('trajectory_source_data.csv',records),('trajectory_observation_source_data.csv',obs)]:
  with (FIG/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
 dynamic.OUT=FIG;dynamic.dynamic(records,obs)

def scores():
 rr=read('artifacts/reviewer-revision-v0.46/all_study_metrics.csv');lookup={(r['model'],r['study_id']):r for r in rr};order=sorted(ANCHORS,key=lambda s:float(lookup['coupled_joint',s]['W_MAPE_pct']));models=[('coupled_uncalibrated_matched','Before calibration','#9BA6AE'),('coupled_joint','After calibration','#235C8E')]
 if any(float(lookup[m,s][k])<=0 for m,l,c in models for s in order for k in ['W_MAPE_pct','gain_MAPE_pct']):raise ValueError('Logarithmic error axes require positive scores')
 fig,axes=plt.subplots(2,1,figsize=(3.45,5.85),sharey=True);fig.subplots_adjust(left=.28,right=.985,bottom=.13,top=.945,hspace=.37)
 for letter,ax,key,title in zip('ab',axes,['W_MAPE_pct','gain_MAPE_pct'],['Growth endpoint','Individual mass gain']):
  for i,s in enumerate(order):ax.plot([float(lookup[m,s][key]) for m,l,c in models],[i,i],color='#D5DCE1',lw=.8,zorder=1)
  for m,label,c in models:ax.scatter([float(lookup[m,s][key]) for s in order],np.arange(16),s=15,color=c,label=label,zorder=3)
  ax.set_xscale('log');ax.set_xlim(.7,700);ax.set_xticks([1,10,100]);ax.set_xticklabels(['1','10','100']);ax.set_yticks(np.arange(16));ax.set_yticklabels(order);ax.grid(axis='x',color='#E3E8EB',lw=.5);ax.set_title(title,fontsize=8,pad=8);ax.set_xlabel('Study MAPE (%) · log scale',fontsize=7.2);ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-15,8),textcoords='offset points',weight='bold',fontsize=9,annotation_clip=False)
 axes[0].invert_yaxis();handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.53,.013),columnspacing=.8,handletextpad=.35,fontsize=6.8);dynamic.save(fig,'fig2_transfer',True)
 ss=['Chen2014','Zhang2017','Zhao2020'];fig,ax=plt.subplots(figsize=(3.45,2.7));fig.subplots_adjust(left=.19,right=.98,bottom=.25,top=.92)
 for offset,(m,label,c) in zip([-.5,.5],models):
  vals=[float(lookup[m,s]['FI_MAPE_pct']) for s in ss];bars=ax.bar(np.arange(3)+offset*.3,vals,.3,color=c,label=label)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.9,f'{v:.1f}',ha='center',fontsize=8)
 ax.set_xticks(range(3));ax.set_xticklabels(ss);ax.set_ylabel('Cumulative intake MAPE (%)');ax.set_ylim(0,50);fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(.53,.02),fontsize=6.8,columnspacing=.8,handletextpad=.4);dynamic.save(fig,'fig3_intake')
 intervals=read('artifacts/reviewer-revision-v0.46/score_intervals.csv');look={(r['model'],r['metric']):r for r in intervals};names=['SGR_growth_only','TGC_plus_intake','ridge','XGBoost','TabICLv2','coupled_joint','nested_selected'];labels=['SGR','TGC + intake','Ridge','XGBoost','TabICLv2','Coupled core','Nested selection']
 fig,axes=plt.subplots(1,2,figsize=(7.0866,3.6),sharey=True);fig.subplots_adjust(left=.18,right=.98,bottom=.19,top=.87,wspace=.27)
 for ax,metric,letter,title in zip(axes,['W','gain'],'ab',['Growth endpoint','Individual mass gain']):
  for i,m in enumerate(names):
   r=look[m,metric];v,lo,hi=[float(r[k]) for k in ['mean','lower','upper']];ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,capsize=2,color='#235C8E' if m in ['coupled_joint','nested_selected'] else '#788792',lw=1)
  ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels);ax.set_xlim(left=0);ax.set_xlabel('Mean study MAPE (%)');ax.set_title(title,pad=12);ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-15,12),textcoords='offset points',weight='bold',fontsize=9,annotation_clip=False);ax.grid(axis='x',color='#E3E8EB',lw=.5)
 axes[0].invert_yaxis();fig.text(.53,.03,'95% intervals: 20,000 paired study bootstrap samples',ha='center',fontsize=8);dynamic.save(fig,'figS2_baseline_intervals',True)
 with (FIG/'matched_study_source_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
  records=[dict(lookup[m,s],stage='loso' if m.startswith('coupled_uncalibrated') else 'information_matched_anchor') for m,label,c in models for s in sorted(ANCHORS)];w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)

def main():
 setup();FIG.mkdir(parents=True,exist_ok=True)
 trajectories();scores()

if __name__=='__main__':main()
