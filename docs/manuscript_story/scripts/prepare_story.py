"""Prepare documented direct carbon categories and broader inferred functions."""
from pathlib import Path
import re,shutil
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parents[1];BASE=OUT.parent;ROOT=BASE.parent
def write(d,n,index=True):d.to_csv(OUT/'inputs'/n,sep='\t',index=index)
def read(p):return pd.read_csv(p,sep='\t',index_col=0)
for study in ['GC','OSS']:
 for source,dest in [('MAG_abundance','MAG'),('direct_gene_TPM','carbon_gene'),('direct_pathway_TPM','carbon_pathway')]:
  write(read(BASE/'inputs'/f'{study}_{source}.tsv'),f'{study}_{dest}.tsv')
 shutil.copy2(BASE/'inputs'/f'{study}_metadata.tsv',OUT/'inputs'/f'{study}_metadata.tsv')
h=pd.read_csv(BASE/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
category={
 'EPS and export markers':h.image_gene[(h.mechanism=='EPS production')&(h.pathway!='Broad glycosyltransferases')].unique().tolist(),
 'Broad glycosyltransferases':['GT2','GT4'],
 'Biomass and necromass formation':h.image_gene[h.mechanism=='Biomass and necromass'].unique().tolist(),
 'PHA storage synthesis':['phaA','phaB','phaC'],
 'Glycogen storage synthesis':['glgC','glgA','glgB'],
 'Trehalose stress persistence':['otsA','otsB','treY','treZ','treS'],
 'Necromass recycling':['nagZ','ampD','anmK','amiA','amiB','amiC'],
 'Plant substrate decomposition':h.image_gene[h.mechanism=='Plant organic matter decomposition'].unique().tolist(),
 'PHA degradation':['phaZ'],'Glycogen degradation':['glgP','glgX'],'Trehalose degradation':['treA','treF']}
links=pd.concat([h[h.image_gene.isin(genes)][['catalog_gene_id','gene_id','MAG']].drop_duplicates().assign(category=cat) for cat,genes in category.items()]).drop_duplicates(['catalog_gene_id','gene_id','category'])
links.to_csv(OUT/'tables/category_locus_membership.tsv',sep='\t',index=False)
pd.DataFrame([{'category':k,'image_labels':';'.join(v),'interpretation':'marker potential; not measured carbon flux'} for k,v in category.items()]).to_csv(OUT/'tables/category_definitions.tsv',sep='\t',index=False)
copy=links.groupby(['MAG','category']).gene_id.nunique().unstack(fill_value=0);write(copy,'MAG_category_copy_number.tsv')
mat={s:{} for s in ['GC','OSS']};keep=set(links.catalog_gene_id.dropna());match=links[['catalog_gene_id','category']].dropna().drop_duplicates()
files=sorted((ROOT/'protein_cluster_pipeline/results/diamond_mcl_mapping/MAG/counts').glob('*.gene_counts.tsv'))
for i,f in enumerate(files):
 s='GC' if f.name.startswith('GC_') else 'OSS';d=pd.read_csv(f,sep='\t',usecols=['sample','gene_id','length','mapped_reads']);sample=str(d['sample'].iloc[0]);sample='sample'+sample.split('_S')[0] if s=='GC' else sample
 rpk=d.mapped_reads/d.length*1000;total=rpk.sum();d['rpk']=rpk
 a=d[d.gene_id.isin(keep)].merge(match,left_on='gene_id',right_on='catalog_gene_id')
 mat[s][sample]=a.groupby('category').rpk.sum()/total*1e6
 if i%25==0:print('Category read quantification',i+1,'/',len(files),flush=True)
for s in mat:write(pd.DataFrame(mat[s]).fillna(0),f'{s}_carbon_category.tsv')
# Broad supplementary potential: all annotated KOs, inferred from copy number.
ko=read(ROOT/'functional_analysis_results/WGCNA/all_MAG_KO_copy_number.tsv')
mapping=pd.read_csv(ROOT/'functional_analysis_results/WGCNA/KO_to_KEGG_pathway_2026-08-12.tsv',sep='\t',header=None,names=['KO','pathway'])
mapping=mapping[mapping.pathway.str.startswith('path:map')].copy();mapping.KO=mapping.KO.str.replace('ko:','',regex=False);mapping.pathway=mapping.pathway.str.replace('path:','',regex=False)
allowed={};section='';subsection=''
for line in (ROOT/'functional_analysis_results/WGCNA/KEGG_pathway_hierarchy_br08901_2026-08-12.txt').read_text().splitlines():
 if line.startswith('A'):section=line[1:].strip()
 elif line.startswith('B'):subsection=line[1:].strip()
 elif line.startswith('C') and section=='Metabolism' and subsection!='Global and overview maps':
  m=re.match(r'C\s+(\d+)\s+(.+)',line)
  if m:allowed['map'+m.group(1)]=m.group(2)
mapping=mapping[mapping.pathway.isin(allowed)&mapping.KO.isin(ko.columns)].drop_duplicates()
mapping.to_csv(OUT/'tables/broad_KO_pathway_mapping.tsv',sep='\t',index=False)
pd.DataFrame(list(allowed.items()),columns=['pathway','name']).to_csv(OUT/'tables/broad_pathway_names.tsv',sep='\t',index=False)
for s in ['GC','OSS']:
 m=read(OUT/'inputs'/f'{s}_MAG.tsv');shared=m.index.intersection(ko.index);pot=ko.loc[shared].T.dot(m.loc[shared]);pot=pot.div(pot.sum(axis=0))*1e6
 write(pot,f'{s}_broad_KO_inferred.tsv')
 p=mapping.merge(pot,left_on='KO',right_index=True).drop(columns='KO').groupby('pathway').sum();write(p,f'{s}_broad_pathway_inferred.tsv')
print('Story inputs complete',flush=True)
