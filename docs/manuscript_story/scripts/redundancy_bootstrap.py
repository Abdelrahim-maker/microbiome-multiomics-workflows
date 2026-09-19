"""Paired conditional bootstrap of cross-level effect-size differences."""
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
OUT=Path(__file__).resolve().parents[1];RNG=np.random.default_rng(20260913);NBOOT=999
rows=[]
for study in ['GC','OSS']:
 md=pd.read_csv(OUT/'inputs'/f'{study}_metadata.tsv',sep='\t',index_col=0);md['Shrub']=np.where(md.shrub=='Shrub',.5,-.5)
 if study=='GC':md['OM']=np.where(md.organic_matter=='OM',.5,-.5);md['OM_x_Shrub']=md.OM*md.Shrub
 formula='1 + OM + Shrub + OM_x_Shrub + C(watering) + C(phase) + C(block)' if study=='GC' else '1 + Shrub + C(fertilizer) + C(context) + C(block)'
 X=patsy.dmatrix(formula,md,return_type='dataframe');termsets=['OM','Shrub','OM_x_Shrub'] if study=='GC' else ['Shrub','C(fertilizer)','C(context)']
 Y={}
 filtering=pd.read_csv(OUT/'tables/feature_filtering.tsv',sep='\t')
 for level in ['MAG','carbon_gene','carbon_pathway']:
  x=pd.read_csv(OUT/'inputs'/f'{study}_{level}.tsv',sep='\t',index_col=0).loc[:,md.index];keep=filtering[(filtering.study==study)&(filtering.level==level)&filtering.retained].feature;x=x.loc[keep]
  a=x.T.to_numpy();a=np.sqrt(a/a.sum(1,keepdims=True));u,s,v=np.linalg.svd(a-a.mean(0),full_matrices=False);Y[level]=u*s
 group=md.block.astype(str)+'__'+md.phase if study=='GC' else md.block.astype(str)
 strata=[np.flatnonzero(group.to_numpy()==g) for g in group.unique()]
 samples=[np.arange(len(md))]+[np.concatenate([RNG.choice(ix,len(ix),replace=True) for ix in strata]) for _ in range(NBOOT)]
 stats={t:{lev:[] for lev in Y} for t in termsets};valid=0
 for ix in samples:
  A=X.to_numpy()[ix];rank=np.linalg.matrix_rank(A)
  if rank<A.shape[1]:continue
  for term in termsets:
   sl=X.design_info.term_name_slices[term];A0=np.delete(A,np.arange(sl.start,sl.stop),axis=1)
   inv=np.linalg.pinv(A);inv0=np.linalg.pinv(A0)
   for lev,yy in Y.items():
    z=yy[ix];s1=np.square(z-A@inv@z).sum();s0=np.square(z-A0@inv0@z).sum();stats[term][lev].append((s0-s1)/s0)
  valid+=1
 for term,vals in stats.items():
  for lev in ['carbon_gene','carbon_pathway']:
   delta=np.array(vals['MAG'])-np.array(vals[lev]);low,high=np.quantile(delta[1:],[.025,.975])
   rows.append(dict(study=study,term=term,comparison='MAG minus '+lev,observed_partial_R2_difference=delta[0],bootstrap_CI_low=low,bootstrap_CI_high=high,valid_bootstraps=valid-1,interpretation='conditional row bootstrap within block/phase; not a formal redundancy mechanism test'))
 print(study,'bootstrap complete',flush=True)
pd.DataFrame(rows).to_csv(OUT/'tables/paired_cross_level_R2_bootstrap.tsv',sep='\t',index=False)
