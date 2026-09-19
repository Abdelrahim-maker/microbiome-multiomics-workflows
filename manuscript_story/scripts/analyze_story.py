"""Factorial, stratified, dispersion, differential and redundancy analyses."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.stats.multitest import multipletests
OUT=Path(__file__).resolve().parents[1];BASE=OUT.parent;ROOT=BASE.parent
RNG=np.random.default_rng(20260911);NPERM=4999
tree=ast.parse((BASE/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'analysis_helpers','exec'))
LEVELS=['MAG','carbon_gene','carbon_pathway','carbon_category','broad_KO_inferred','broad_pathway_inferred']
md={s:rd(f'{s}_metadata.tsv') for s in ['GC','OSS']}
for s,d in md.items():
 d['Shrub']=np.where(d.shrub=='Shrub',.5,-.5)
 if s=='GC':d['OM']=np.where(d.organic_matter=='OM',.5,-.5);d['OM_x_Shrub']=d.OM*d.Shrub
GC=['OM','Shrub','OM_x_Shrub','C(watering)','C(phase)','C(block)'];OSS=['Shrub','C(fertilizer)','C(context)','C(block)']
def scope(level):return 'supplementary_inferred' if 'inferred' in level else 'primary_direct_and_MAG'
def compact(Y):
 # All nonzero singular dimensions retained: distances/statistics are unchanged.
 Y=Y-Y.mean(0);u,s,v=np.linalg.svd(Y,full_matrices=False)
 return u*s, s,v
def covariance_model(Y,d,terms,contrasts,study,level,analysis,features,cluster=False):
 X=design(d,terms);a=X.to_numpy();n,k=a.shape;rank=np.linalg.matrix_rank(a)
 if rank<k or n-rank<8:return [],dict(study=study,level=level,analysis=analysis,n=n,rank=rank,columns=k,status='skipped_rank_or_df')
 inv=np.linalg.pinv(a.T@a);B=inv@a.T@Y;E=Y-a@B;leverage=np.einsum('ij,jk,ik->i',a,inv,a);dof=n-rank
 result=[]
 for label,weights in contrasts.items():
  c=np.array([weights.get(col,0) for col in X.columns],float)
  if not np.any(c):continue
  beta=c@B
  if cluster:
   gs=d.candidate_plot.unique();G=len(gs)
   if G<=k:continue
   variance=np.zeros(Y.shape[1])
   for g in gs:
    ix=np.flatnonzero(d.candidate_plot.to_numpy()==g);score=c@inv@a[ix].T@E[ix];variance+=score*score
   variance*=G/(G-1)*(n-1)/(n-k);dof=G-1
  else:
   w=c@inv@a.T;variance=(w*w)@((E/np.maximum(1-leverage[:,None],1e-8))**2)
  se=np.sqrt(np.maximum(variance,0));p=2*stats.t.sf(np.divide(abs(beta),se,out=np.zeros_like(beta),where=se>0),dof);crit=stats.t.ppf(.975,dof)
  for j,f in enumerate(features):result.append(dict(study=study,level=level,analysis=analysis,feature=f,contrast=label,beta=beta[j],SE=se[j],CI_low=beta[j]-crit*se[j],CI_high=beta[j]+crit*se[j],p=p[j],n=n,df_resid=dof,method='candidate_plot_CR1' if cluster else 'HC3'))
 return result,dict(study=study,level=level,analysis=analysis,n=n,rank=rank,columns=k,status='fitted',condition_number=np.linalg.cond(a))
GC_CON={'OM_average':{'OM':1},'Shrub_average':{'Shrub':1},'OM_x_Shrub':{'OM_x_Shrub':1},'OM_with_shrub':{'OM':1,'OM_x_Shrub':.5},'OM_without_shrub':{'OM':1,'OM_x_Shrub':-.5},'Shrub_with_OM':{'Shrub':1,'OM_x_Shrub':.5},'Shrub_without_OM':{'Shrub':1,'OM_x_Shrub':-.5}}
community=[];dispersion=[];effects=[];qc=[];filters=[];spectra=[];matrix={}
for study,d in md.items():
 for level in LEVELS:
  print(study,level,flush=True)
  x=rd(f'{study}_{level}.tsv').loc[:,d.index]
  if level=='MAG':x=x.div(x.sum(0))*1e6
  keep=(x.gt(0).sum(1)>=max(5,int(np.ceil(.1*len(d)))))&(x.std(1)>1e-10)
  filters.extend([dict(study=study,level=level,feature=f,prevalence=x.loc[f].gt(0).mean(),retained=bool(keep[f])) for f in x.index]);x=x.loc[keep];matrix[(study,level)]=x
  Y=hellinger(x);S,s,v=compact(Y)
  scores=pd.DataFrame(S[:,:2],index=d.index,columns=['PC1','PC2']).join(d);scores.to_csv(OUT/'tables'/f'{study}_{level}_PCA_scores.tsv',sep='\t',index_label='sample_id')
  save(pd.DataFrame({'feature':x.index,'PC1_loading':v[0],'PC2_loading':v[1]}),f'{study}_{level}_PCA_loadings.tsv')
  spectra.append(dict(study=study,level=level,n=len(d),features=len(x),PC1_percent=s[0]**2/s.dot(s)*100,PC2_percent=s[1]**2/s.dot(s)*100))
  terms=GC if study=='GC' else OSS
  for term in terms:
   if term=='C(block)':continue
   res=test_mv(S,d,terms,term);community.append(dict(study=study,level=level,analysis='factorial' if study=='GC' else 'main',scope=scope(level),term=term,**res))
  if study=='GC':
   contrasts=GC_CON
  else:
   columns=design(d,terms).columns;contrasts={c:{c:1} for c in columns if c!='Intercept' and c!='C(block)' and not c.startswith('C(block)')}
  r,q=covariance_model(transform(x),d,terms,contrasts,study,level,'factorial' if study=='GC' else 'main',x.index);effects+=r;qc.append(q)
  if study=='OSS':
   r,q=covariance_model(transform(x),d,terms,contrasts,study,level,'candidate_plot_sensitivity',x.index,True);effects+=r;qc.append(q)
  if study=='GC' and 'inferred' not in level:
   for pc in [.1,1.0]:
    r,q=covariance_model(transform(x,pc),d,terms,contrasts,study,level,f'pseudocount_{pc}',x.index);effects+=r;qc.append(q)
  # Marginal dispersion of groups plus four-cell factorial heterogeneity.
  groupdefs={'OM':'organic_matter','Shrub':'shrub','four_groups':'four_groups'} if study=='GC' else {'context':'context','shrub':'shrub','fertilizer':'fertilizer'}
  dd=d.copy()
  if study=='GC':dd['four_groups']=dd.shrub+'__'+dd.organic_matter
  for label,group in groupdefs.items():
   dist=np.zeros(len(dd))
   for value in dd[group].unique():
    ix=np.flatnonzero(dd[group].to_numpy()==value);dist[ix]=np.linalg.norm(S[ix]-spatial_median(S[ix]),axis=1)*np.sqrt(len(ix)/(len(ix)-1))
   ts=(['C(four_groups)','C(watering)','C(phase)','C(block)'] if label=='four_groups' else (['OM','Shrub','C(watering)','C(phase)','C(block)'] if study=='GC' else OSS))
   term='C(four_groups)' if label=='four_groups' else (label if study=='GC' else ('Shrub' if label=='shrub' else f'C({label})'))
   dispersion.append(dict(study=study,level=level,analysis='group_dispersion',scope=scope(level),term=label,**test_mv(dist[:,None],dd,ts,term)))
  if 'inferred' not in level:
   subsets=({'OM_only':d.organic_matter=='OM','noOM_only':d.organic_matter=='noOM','Shrub_only':d.shrub=='Shrub','noShrub_only':d.shrub=='noShrub'} if study=='GC' else {'rainy_only':d.season=='Rainy','soil_only':d.sample_type=='Soil'})
   for label,mask in subsets.items():
    z=d.loc[mask];xx=x.loc[:,z.index];SS,ss,vv=compact(hellinger(xx))
    if study=='GC':
     term='Shrub' if label in ['OM_only','noOM_only'] else 'OM';ts=[term,'C(watering)','C(phase)','C(block)'];factor='shrub' if term=='Shrub' else 'organic_matter';testterms=[term]
    else:
     ts=['Shrub','C(fertilizer)','C(sample_type)' if label=='rainy_only' else 'C(season)','C(block)'];testterms=ts[:-1];factor='sample_type' if label=='rainy_only' else 'season'
    sc=pd.DataFrame(SS[:,:2],index=z.index,columns=['PC1','PC2']).join(z);sc.to_csv(OUT/'tables'/f'{study}_{level}_{label}_PCA_scores.tsv',sep='\t',index_label='sample_id')
    spectra.append(dict(study=study,level=level+'__'+label,n=len(z),features=len(xx),PC1_percent=ss[0]**2/ss.dot(ss)*100,PC2_percent=ss[1]**2/ss.dot(ss)*100))
    for t in testterms:community.append(dict(study=study,level=level,analysis=label,scope=scope(level),term=t,**test_mv(SS,z,ts,t)))
    dist=np.zeros(len(z))
    for value in z[factor].unique():
     ix=np.flatnonzero(z[factor].to_numpy()==value);dist[ix]=np.linalg.norm(SS[ix]-spatial_median(SS[ix]),axis=1)*np.sqrt(len(ix)/(len(ix)-1))
    t=term if study=='GC' else ('C(sample_type)' if label=='rainy_only' else 'C(season)')
    dispersion.append(dict(study=study,level=level,analysis=label,scope=scope(level),term=t,**test_mv(dist[:,None],z,ts,t)))
  if study=='OSS':
   z=d.copy();z['block']=z.candidate_plot
   community.append(dict(study=study,level=level,analysis='within_candidate_plot',scope=scope(level),term='C(context)',**test_mv(S,z,['C(candidate_plot)','C(context)'],'C(context)')))
for data,name in [(community,'community_PERMANOVA.tsv'),(dispersion,'PERMDISP.tsv')]:
 a=pd.DataFrame(data);a['q']=a.groupby(['study','scope','analysis']).p.transform(bh);save(a,name)
e=pd.DataFrame(effects);e['q']=e.groupby(['study','level','analysis']).p.transform(bh);save(e,'differential_features.tsv')
save(pd.DataFrame(qc),'model_QC.tsv');save(pd.DataFrame(filters),'feature_filtering.tsv');save(pd.DataFrame(spectra),'ordination_variance.tsv')

# Functional redundancy is summarized directly from carrier distributions.
cp=pd.read_csv(BASE/'inputs/MAG_gene_copy_number.tsv',sep='\t',index_col=0)
hits=pd.read_csv(BASE/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
gene_static=hits.groupby('image_gene').agg(carrier_MAGs=('MAG','nunique'),carrier_phyla=('phylum','nunique'),annotated_loci=('gene_id','nunique')).reset_index();save(gene_static,'gene_carrier_redundancy.tsv')
red=[];redeffects=[];redqc=[]
for study,d in md.items():
 m=rd(f'{study}_MAG.tsv');cp0=cp.reindex(m.index,fill_value=0)
 neff=pd.DataFrame(index=d.index,columns=cp0.columns,dtype=float)
 for gene in cp0:
  a=cp0[gene].to_numpy()[:,None]*m.loc[:,d.index].to_numpy();den=a.sum(0);p=np.divide(a,den,out=np.zeros_like(a),where=den>0)
  effective=np.divide(1,np.square(p).sum(0),out=np.full(len(d),np.nan),where=den>0);neff[gene]=effective
  for j,sample in enumerate(d.index):red.append(dict(study=study,sample_id=sample,gene=gene,effective_carriers=effective[j],detected_carriers=int((a[:,j]>0).sum()),largest_carrier_fraction=p[:,j].max(),inferred_total=den[j]))
 # Geometric mean effective carrier count across available labels is descriptive.
 # Identical locus sets are collapsed so aliases do not receive extra weight.
 sets={}
 for g,z in hits.groupby('image_gene'):sets.setdefault(tuple(sorted(z.gene_id.unique())),g)
 distinct=[g for g in sets.values() if g in neff]
 summary=np.exp(np.log(neff[distinct]).mean(axis=1));summary.to_frame('geometric_mean_effective_carriers').to_csv(OUT/'inputs'/f'{study}_redundancy_summary.tsv',sep='\t')
 contrasts=GC_CON if study=='GC' else {'Shrub':{'Shrub':1}}
 r,q=covariance_model(np.log2(summary.to_numpy())[:,None],d,GC if study=='GC' else OSS,contrasts,study,'redundancy_summary','factorial' if study=='GC' else 'main',['effective_carriers']);redeffects+=r;redqc.append(q)
pd.DataFrame(red).to_csv(OUT/'tables/sample_gene_redundancy.tsv',sep='\t',index=False)
r=pd.DataFrame(redeffects);r['q']=r.groupby('study').p.transform(bh);save(r,'redundancy_treatment_models.tsv')
# Crosswalk to actual MAG treatment coefficients and taxonomy, not a network inference.
mag_eff=e[(e.level=='MAG')&(e.analysis.isin(['factorial','main','candidate_plot_sensitivity']))]
carrier=hits[['image_gene','MAG','phylum','genus','classification']].drop_duplicates().merge(mag_eff,left_on='MAG',right_on='feature')
save(carrier,'carbon_gene_MAG_taxonomy_treatment_response.tsv')
print('Story statistics complete',flush=True)
