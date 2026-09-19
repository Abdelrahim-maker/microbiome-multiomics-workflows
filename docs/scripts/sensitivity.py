"""Recorded-design and inferred-plot sensitivity, plus mutually adjusted chemistry."""
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.stats.multitest import multipletests
OUT=Path(__file__).resolve().parents[1];RNG=np.random.default_rng(20260911);NPERM=4999
# Reuse the exact tested numerical functions without executing the main pipeline.
tree=ast.parse((OUT/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'analysis_helpers','exec'))
rows=[];effects=[];qc=[]
md={s:rd(f'{s}_metadata.tsv') for s in ['GC','OSS']}
for s,d in md.items():d['shrub']=pd.Categorical(d.shrub,categories=['noShrub','Shrub'])
base={'GC':['C(shrub)','C(watering)','C(organic_matter)','C(phase)','C(block)'],'OSS':['C(shrub)','C(fertilizer)','C(context)','C(block)']}
d=md['OSS'];plotqc=d.groupby('candidate_plot').agg(n=('shrub','size'),contexts=('context','nunique'),shrub_levels=('shrub','nunique'),fertilizer_levels=('fertilizer','nunique'),block_levels=('block','nunique'))
save(plotqc.reset_index(),'OSS_candidate_plot_audit.tsv')
complete=plotqc.index[(plotqc.contexts==4)&(plotqc.shrub_levels==1)&(plotqc.fertilizer_levels==1)&(plotqc.block_levels==1)]
for level in ['gene','pathway','MAG']:
 x=rd(f'OSS_{"MAG_abundance" if level=="MAG" else "direct_"+level+"_TPM"}.tsv').loc[:,d.index]
 x=x.loc[(x.gt(0).mean(axis=1)>=.1)&(x.std(axis=1)>0)]
 if level=='MAG':x=x.div(x.sum(0))*1e6
 H=hellinger(x);z=d.copy();z['block']=z.candidate_plot
 r=test_mv(H,z,['C(candidate_plot)','C(context)'],'C(context)');rows.append(dict(study='OSS',level=level,analysis='within_candidate_plot_context',term='context',**r))
 # Average duplicate observations within context before equal-weight averaging contexts.
 yy=pd.DataFrame(transform(x),index=d.index,columns=x.index).join(d[['candidate_plot','context']])
 yy=yy.groupby(['candidate_plot','context'],observed=True).mean().groupby('candidate_plot').mean().loc[complete]
 p=d.groupby('candidate_plot',observed=True).first().loc[complete];p['shrub']=pd.Categorical(p.shrub,categories=['noShrub','Shrub'])
 terms=['C(shrub)','C(fertilizer)','C(block)']
 rr,qq=regress(yy.to_numpy(),p,terms,'OSS',level,'complete_candidate_plot_mean',yy.columns);effects+=rr;qc.append(qq)
 # Euclidean multivariate model on mean log relative abundances, one unit per plot.
 for term in terms[:2]:rows.append(dict(study='OSS',level=level,analysis='complete_candidate_plot_mean_logTPM',term=term,**test_mv(yy.to_numpy(),p,terms,term)))
 # Soil-only repeated season test: plot fixed effects, permutations within plot.
 soil=d[d.sample_type=='Soil'].copy();soil['block']=soil.candidate_plot
 rows.append(dict(study='OSS',level=level,analysis='soil_season_within_candidate_plot',term='season',**test_mv(hellinger(x.loc[:,soil.index]),soil,['C(candidate_plot)','C(season)'],'C(season)')))

for study,d in md.items():
 # Test relative allocation within the target panel, separating panel-wide shifts
 # against the full MAG reference from changes among the selected genes.
 for level in ['gene','pathway']:
  x=rd(f'{study}_direct_{level}_TPM.tsv').loc[:,d.index];x=x.loc[(x.gt(0).mean(axis=1)>=.1)&(x.std(axis=1)>0)]
  yy=transform(x);yy=yy-yy.mean(axis=1,keepdims=True)
  rr,qq=regress(yy,d,base[study],study,level,'within_panel_CLR',x.index);effects+=rr;qc.append(qq)
 chem=['Al','Ca','Cu','Fe','K','Mg','Mn','Na','P','Zn'] if study=='GC' else ['pH','perC','perN']
 z=d.dropna(subset=chem).copy();A=z[chem];A=(A-A.mean())/A.std();u,s,v=np.linalg.svd(A,full_matrices=False)
 z['chemPC1']=u[:,0]*s[0];z['chemPC2']=u[:,1]*s[1]
 for c in ['chemPC1','chemPC2']:z[c]=(z[c]-z[c].mean())/z[c].std()
 terms=[t for t in base[study] if z[t[2:-1]].nunique()>1]+['chemPC1','chemPC2']
 for level in ['gene','pathway']:
  x=rd(f'{study}_direct_{level}_TPM.tsv').loc[:,z.index];x=x.loc[(x.gt(0).mean(axis=1)>=.1)&(x.std(axis=1)>0)]
  rr,qq=regress(transform(x),z,terms,study,level,'design_plus_joint_chemistry_PCs',x.index);effects+=rr;qc.append(qq)
  if study=='OSS':
   rr,qq=regress(transform(x),z,terms,study,level,'design_plus_joint_chemistry_PCs_candidate_plot',x.index,True);effects+=rr;qc.append(qq)
r=pd.DataFrame(rows);r['q']=r.groupby(['study','analysis']).p.transform(bh);save(r,'experimental_unit_sensitivity_community.tsv')
e=pd.DataFrame(effects);e['q']=e.groupby(['study','level','analysis']).p.transform(bh);save(e,'joint_chemistry_and_plot_sensitivity_features.tsv');save(pd.DataFrame(qc),'sensitivity_design_QC.tsv')
print('Sensitivity complete',flush=True)
