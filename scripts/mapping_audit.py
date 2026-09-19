"""Reference concentration QC and read-supported taxonomic contributions."""
from pathlib import Path
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
h=pd.read_csv(OUT/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
links=h[['catalog_gene_id','image_gene','MAG']].dropna().drop_duplicates();keep=set(links.catalog_gene_id)
meta={s:pd.read_csv(OUT/'inputs'/f'{s}_metadata.tsv',sep='\t',index_col=0) for s in ['GC','OSS']}
contrib={};ns={};qc=[]
files=sorted((ROOT/'protein_cluster_pipeline/results/diamond_mcl_mapping/MAG/counts').glob('*.gene_counts.tsv'))
for i,f in enumerate(files):
 s='GC' if f.name.startswith('GC_') else 'OSS';d=pd.read_csv(f,sep='\t',usecols=['sample','gene_id','length','mapped_reads'])
 raw=str(d['sample'].iloc[0]);sample='sample'+raw.split('_S')[0] if s=='GC' else raw
 d['MAG']=d.gene_id.str.extract(r'^MAG__(.*?)\|',expand=False);d['rpk']=d.mapped_reads/d.length*1000;total=d.rpk.sum()
 bymag=d.groupby('MAG').mapped_reads.sum().sort_values(ascending=False);reads=d.mapped_reads.sum()
 qc.append(dict(study=s,sample_id=sample,dominant_MAG=bymag.index[0],dominant_MAG_read_fraction=bymag.iloc[0]/reads,top10_locus_read_fraction=d.mapped_reads.nlargest(10).sum()/reads,detected_loci=int(d.mapped_reads.gt(0).sum()),reference_genes=len(d)))
 z=d[d.gene_id.isin(keep)][['gene_id','rpk']].merge(links,left_on='gene_id',right_on='catalog_gene_id')
 z=z.groupby(['image_gene','MAG']).rpk.sum()/total*1e6
 key=(s,str(meta[s].loc[sample,'shrub']));contrib[key]=contrib[key].add(z,fill_value=0) if key in contrib else z.copy();ns[key]=ns.get(key,0)+1
 if i%25==0:print('Attribution',i+1,'/',len(files),flush=True)
rows=[]
for (s,shrub),z in contrib.items():
 z=z/ns[(s,shrub)];den=z.groupby(level=0).sum()
 for (gene,mag),value in z.items():rows.append(dict(study=s,shrub=shrub,gene=gene,MAG=mag,mean_direct_TPM=value,fraction_gene_reads=value/den[gene] if den[gene] else np.nan,n_samples=ns[(s,shrub)]))
r=pd.DataFrame(rows);tax=h[['MAG','phylum','genus','classification']].drop_duplicates('MAG');r=r.merge(tax,on='MAG',how='left');r.to_csv(OUT/'tables/direct_read_MAG_gene_contributions.tsv',sep='\t',index=False)
pd.DataFrame(qc).to_csv(OUT/'tables/reference_mapping_concentration_QC.tsv',sep='\t',index=False)
print('Mapping audit complete',flush=True)
