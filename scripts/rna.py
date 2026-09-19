"""Expanded image-target DNA/RNA association using existing shared mappings."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
hits=pd.read_csv(OUT/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
cw=pd.read_csv(ROOT/'protein_cluster_pipeline/results/mmseqs_shared_mapping/crosswalks/MAG.tsv',sep='\t')
links=hits.merge(cw,left_on='catalog_gene_id',right_on='gene_id')
links[['image_gene','pathway','mechanism','catalog_gene_id','shared_cluster_id']].drop_duplicates().to_csv(OUT/'tables/image_shared_cluster_crosswalk.tsv',sep='\t',index=False)
keep=set(links.shared_cluster_id);qc=[];data={}
cached_qc=OUT/'tables/expanded_DNA_RNA_mapping_QC.tsv'
cached=pd.read_csv(cached_qc,sep='\t') if cached_qc.exists() else pd.DataFrame()
gc=pd.read_excel(ROOT/'GC_metadata_omics_updated_afaf.xlsx');rna_id=dict(zip(gc.file_name,gc.sample_ID))
for source in ['GC_MetaG','GC_MetaT']:
 cache=OUT/'inputs'/f'{source}_expanded_shared_target_counts.tsv'
 if cache.exists() and len(cached) and (cached.source==source).sum()==48:
  data[source]=pd.read_csv(cache,sep='\t',index_col=0);qc+=cached[cached.source==source].to_dict('records');continue
 mat={};files=sorted((ROOT/f'protein_cluster_pipeline/results/mmseqs_shared_mapping/{source}/counts').glob('*.shared_cluster_counts.tsv'))
 for i,f in enumerate(files):
  sample=f.name.split('__')[1].split('.')[0].split('_S')[0];sample='sample'+sample if source=='GC_MetaG' else rna_id.get(sample,sample)
  total=0;parts=[]
  for chunk in pd.read_csv(f,sep='\t',usecols=['shared_cluster_id','mapped_reads'],chunksize=400000):
   total+=chunk.mapped_reads.sum();parts.append(chunk[chunk.shared_cluster_id.isin(keep)])
  d=pd.concat(parts).set_index('shared_cluster_id').mapped_reads;mat[sample]=d
  qc.append(dict(source=source,sample_id=sample,total_shared_mapped_reads=total,target_shared_reads=d.sum(),detected_target_clusters=int(d.gt(0).sum())))
  if i%8==0:print(source,i+1,'/',len(files),flush=True)
 x=pd.DataFrame(mat).reindex(sorted(keep)).fillna(0);x.to_csv(OUT/'inputs'/f'{source}_expanded_shared_target_counts.tsv',sep='\t');data[source]=x
q=pd.DataFrame(qc);q.to_csv(OUT/'tables/expanded_DNA_RNA_mapping_QC.tsv',sep='\t',index=False)
md=pd.read_csv(OUT/'inputs/GC_metadata.tsv',sep='\t',index_col=0);md['shrub']=pd.Categorical(md.shrub,categories=['noShrub','Shrub'])
effects=[];filtering=[];sensitivity=[]
for level,col in [('gene','image_gene'),('pathway','pathway')]:
 # A shared cluster assigned to several image features cannot identify one feature.
 z=links[['shared_cluster_id',col]].drop_duplicates();unique=z.groupby('shared_cluster_id')[col].nunique();z=z[z.shared_cluster_id.isin(unique[unique==1].index)]
 mats={}
 for source,x in data.items():
  a=z.merge(x,left_on='shared_cluster_id',right_index=True).drop(columns='shared_cluster_id').groupby(col).sum()
  totals=q[q.source==source].set_index('sample_id').total_shared_mapped_reads
  mats[source]=a.div(totals.reindex(a.columns).replace(0,np.nan),axis=1)*1e6
  mats[source].to_csv(OUT/'inputs'/f'{source}_unambiguous_{level}_CPM.tsv',sep='\t')
 dna,rna=mats['GC_MetaG'],mats['GC_MetaT'];samples=md.index.intersection(dna.columns).intersection(rna.columns)
 for gene in dna.index.intersection(rna.index):
  d=md.loc[samples].copy();d['rna']=rna.loc[gene,samples];d['dna']=dna.loc[gene,samples];d=d.dropna(subset=['rna','dna'])
  present=int(d.rna.gt(0).sum());status='eligible' if len(d)>=20 and present>=10 and d.dna.gt(0).sum()>=10 else 'insufficient_RNA_or_DNA_detection'
  filtering.append(dict(level=level,feature=gene,n=len(d),RNA_detected=present,DNA_detected=int(d.dna.gt(0).sum()),status=status))
  if status!='eligible':continue
  d['log_rna']=np.log2(d.rna+.5);d['log_dna']=np.log2(d.dna+.5)
  fit=smf.ols('log_rna ~ log_dna + C(shrub) + C(watering) + C(organic_matter) + C(phase) + C(block)',d).fit(cov_type='HC3',use_t=True)
  if np.linalg.matrix_rank(fit.model.exog)<fit.model.exog.shape[1]:continue
  ci=fit.conf_int()
  for term in fit.params.index:
   if term=='Intercept' or term.startswith('C(block)'):continue
   effects.append(dict(level=level,feature=gene,coefficient=term,beta=fit.params[term],SE=fit.bse[term],CI_low=ci.loc[term,0],CI_high=ci.loc[term,1],p=fit.pvalues[term],n=len(d),RNA_detected=present,interpretation='relative_shared_cluster_transcription_adjusted_for_DNA; exploratory_sparse_RNA'))
  for pc in [.1,1.0]:
   d['log_rna']=np.log2(d.rna+pc);d['log_dna']=np.log2(d.dna+pc)
   f=smf.ols('log_rna ~ log_dna + C(shrub) + C(watering) + C(organic_matter) + C(phase) + C(block)',d).fit(cov_type='HC3',use_t=True)
   for term in f.params.index:
    if term=='Intercept' or term.startswith('C(block)'):continue
    sensitivity.append(dict(level=level,feature=gene,coefficient=term,pseudocount=pc,beta=f.params[term],p=f.pvalues[term],n=len(d)))
e=pd.DataFrame(effects)
if len(e):e['q']=e.groupby('level').p.transform(lambda p:multipletests(p,method='fdr_bh')[1])
e.to_csv(OUT/'tables/DNA_adjusted_RNA_models.tsv',sep='\t',index=False);pd.DataFrame(filtering).to_csv(OUT/'tables/RNA_feature_filtering.tsv',sep='\t',index=False)
ss=pd.DataFrame(sensitivity)
if len(ss):ss['q']=ss.groupby(['level','pseudocount']).p.transform(lambda p:multipletests(p,method='fdr_bh')[1])
ss.to_csv(OUT/'tables/RNA_pseudocount_sensitivity.tsv',sep='\t',index=False)
print('RNA complete; eligible models',len(e),flush=True)
