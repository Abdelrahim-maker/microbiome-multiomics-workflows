"""Audit image targets and construct direct-read and MAG-inferred matrices."""
from pathlib import Path
import re, json, hashlib
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]; ROOT=OUT.parent
def save(d,name,folder='tables',index=False): d.to_csv(OUT/folder/name,sep='\t',index=index)
def rd(p): return pd.read_csv(ROOT/p,sep='\t',index_col=0)
defs={}; aliases={}
for line in (OUT/'sources/KEGG_KO_definitions_2026-09-10.tsv').read_text().splitlines():
    ko,desc=line.split('\t',1);defs[ko]=desc
    for gene in desc.split(';')[0].split(', '): aliases.setdefault(gene,[]).append(ko)
groups=[
 ('EPS production','Alginate','algD alg8 alg44 algE algL algG algK algX algA algC'),
 ('EPS production','Psl polysaccharide',' '.join('psl'+c for c in 'ABCDEFGHIJKLMNO')),
 ('EPS production','Pel polysaccharide',' '.join('pel'+c for c in 'ABCDEFG')),
 ('EPS production','PNAG','pgaA pgaB pgaC pgaD'),
 ('EPS production','Colanic acid','wcaA wcaC wcaD wcaE wcaF wcaI wcaJ wcaK wcaL'),
 ('EPS production','Cellulose synthesis','bcsA bcsB bcsC bcsZ bcsQ'),
 ('EPS production','Polysaccharide export','wzx wzy wza wzc wzz'),
 ('EPS production','Broad glycosyltransferases','GT2 GT4'),
 ('Biomass and necromass','Peptidoglycan synthesis','glmS glmM glmU murA murB murC murD murE murF mraY murG murJ pbp'),
 ('Biomass and necromass','LPS and envelope','lpxA lpxB lpxC lpxD lpxH lpxK kdsA kdsB gmhA gmhB gmhC gmhD waa'),
 ('Intracellular storage','Glycogen synthesis','glgC glgA glgB'),
 ('Intracellular storage','Glycogen degradation','glgP glgX'),
 ('Intracellular storage','PHA synthesis','phaA phaB phaC'),
 ('Intracellular storage','PHA degradation','phaZ'),
 ('Intracellular storage','Trehalose synthesis','otsA otsB treY treZ treS'),
 ('Intracellular storage','Trehalose degradation','treA treF'),
 ('Necromass recycling','Peptidoglycan degradation','nagZ ampD anmK amiA amiB amiC muramidases amidases'),
 ('Necromass recycling','General lysis and turnover','lyt cid peptidases'),
 ('Plant organic matter decomposition','Cellulose degradation','GH5 GH6 GH7 GH8 GH9 GH12 GH44 GH45 GH48 CBM1 CBM2 CBM3'),
 ('Plant organic matter decomposition','Hemicellulose degradation','GH10 GH11 GH30 GH43 CE1 CE2 CE4 CE6'),
 ('Plant organic matter decomposition','Starch degradation','GH13 GH15 GH31 GH77 amyA pulA malZ'),
 ('Plant organic matter decomposition','Aromatic degradation','AA1 AA2 AA3 AA4 AA5 AA6 AA7 AA9 catA catB catC pcaG pcaH pcaB pcaC pcaD')]
overrides={'pelA':['K21006'],'pelC':['K21008'],'glmS':['K00820'],
 'phaA':['K00626'],'phaB':['K00023'],'phaC':['K03821'],'treS':['K05343'],
 'catA':['K03381'],'catB':['K01856'],'amiA':['K01448'],'amiB':['K01448'],'amiC':['K01448'],
 'wzx':['K16693','K16695','K18799'],'wzc':['K16692'],'wzz':['K05789','K05790'],
 'pbp':[k for k,v in defs.items() if 'penicillin-binding protein' in v and 'activator' not in v],
 'waa':sorted({k for a,ks in aliases.items() if re.fullmatch(r'waa[A-Z]',a) for k in ks})}
rows=[]
for mechanism,pathway,genes in groups:
 for gene in genes.split():
    family=bool(re.fullmatch(r'(?:GH|GT|CE|AA|CBM)\d+',gene))
    kos=overrides.get(gene,aliases.get(gene,[]))
    note='KO ortholog; homologous function, not biochemical validation'
    if family: note='Broad CAZy family: substrate specificity and stabilization direction unresolved'
    if gene in ['phaA','phaB','algA','wzx','wzy','wza','wzc','wzz','waa','pbp']:note='Shared/broad function; not pathway-specific alone'
    if gene in ['amiA','amiB','amiC','treA','treF']:note='KO does not distinguish these image gene names'
    if not family and not kos:note='No unambiguous KO mapping assigned; not evidence of biological absence'
    rows.append(dict(mechanism=mechanism,pathway=pathway,image_gene=gene,KO=';'.join(kos),
                     annotation_type='CAZy_family' if family else ('KO' if kos else 'unresolved'),
                     definition=' | '.join(defs[k] for k in kos),interpretation=note))
targets=pd.DataFrame(rows);save(targets,'image_target_crosswalk.tsv')
annotation_files=sorted((ROOT/'dram_annotation/working_dir').glob('*/annotations.tsv'))+[ROOT/'dram_annotation_batch2/annotations.tsv']
parts=[]
for f in annotation_files:
    a=pd.read_csv(f,sep='\t',low_memory=False)
    a=a.rename(columns={a.columns[0]:'gene_id','fasta':'MAG'});parts.append(a)
ann=pd.concat(parts,ignore_index=True).drop_duplicates('gene_id')
ann['MAG']=ann.MAG.str.replace('_bin.','.bin.',regex=False)
mag=rd('functional_analysis_results/maaslin2_inputs/MAG_GC_MetaG.tsv')
ann=ann[ann.MAG.isin(mag.index)].copy()
save(pd.DataFrame({'MAG':mag.index,'has_annotation':mag.index.isin(ann.MAG)}),'MAG_annotation_coverage.tsv')
cw=pd.read_csv(ROOT/'protein_cluster_pipeline/results/diamond_mcl_source_clusters/MAG/gene_to_local_cluster.tsv',sep='\t')
cw['dram_id']=cw.gene_id.str.replace('MAG__','',regex=False).str.replace('|','_',regex=False)
cw=cw.rename(columns={'gene_id':'catalog_gene_id'})
ann=ann.merge(cw[['dram_id','catalog_gene_id','local_cluster_id']],left_on='gene_id',right_on='dram_id',how='left')
# Use an existing threshold-validated EPS call set for KOs covered by that scan.
eps=pd.read_csv(ROOT/'functional_analysis_results/EPS_KOfam_validated_hits.tsv',sep='\t')
eps_panel=pd.read_csv(ROOT/'analysis/functional_targets.tsv',sep='\t')
eps_kos=set(';'.join(eps_panel.loc[eps_panel.category=='EPS','target_kos']).split(';'))
hits=[];coverage=[]
for r in targets.itertuples():
 if r.annotation_type=='CAZy_family':
    a=ann[ann.cazy_ids.fillna('').str.contains(r'(?<![A-Za-z0-9])'+r.image_gene+r'(?:_\d+)?(?![A-Za-z0-9])',regex=True)].copy()
    a['evidence']='DRAM CAZy family';a['matched_KO']=''
 elif r.annotation_type=='KO':
    kos=r.KO.split(';');parts=[]
    for ko in kos:
     if ko in eps_kos:
        e=eps[eps.ko_id==ko];a=ann[ann.gene_id.isin(e.gene_id)].copy();a['evidence']='existing adaptive-threshold KOfam EPS validation'
     else:
        a=ann[ann.ko_id==ko].copy();a['evidence']='DRAM KO rank '+a['rank'].fillna('unknown')
     a['matched_KO']=ko;parts.append(a)
    a=pd.concat(parts,ignore_index=True)
 else:a=ann.iloc[:0].copy()
 coverage.append(dict(image_gene=r.image_gene,pathway=r.pathway,mechanism=r.mechanism,
                      annotated_loci=a.gene_id.nunique(),carrier_MAGs=a.MAG.nunique(),
                      status='detected' if len(a) else ('unresolved_mapping' if r.annotation_type=='unresolved' else 'not_detected_in_available_annotations')))
 if len(a):
    a['image_gene']=r.image_gene;a['pathway']=r.pathway;a['mechanism']=r.mechanism;a['annotation_type']=r.annotation_type
    hits.append(a)
hits=pd.concat(hits,ignore_index=True)
cols=['image_gene','pathway','mechanism','annotation_type','gene_id','catalog_gene_id','MAG','local_cluster_id','matched_KO','evidence','rank','kegg_hit','cazy_ids','scaffold','start_position','end_position']
tax=pd.read_csv(ROOT/'gtdbtk_out/gtdbtk_taxonomy.tsv',sep='\t')
hits=hits[cols].merge(tax,on='MAG',how='left');save(hits,'image_gene_locus_cluster_MAG_taxonomy.tsv')
save(hits[hits.catalog_gene_id.isna()],'annotated_loci_without_cluster_mapping.tsv')
save(pd.DataFrame(coverage),'image_target_detection.tsv')
# Count each locus once within a feature/pathway, even when aliases share a KO.
levels={'gene':'image_gene','pathway':'pathway','mechanism':'mechanism'}
links={level:hits[['catalog_gene_id',col]].dropna().drop_duplicates() for level,col in levels.items()}
keep=set(hits.catalog_gene_id.dropna());all_qc=[];matrices={s:{l:{} for l in levels} for s in ['GC','OSS']}
lengths={};clusters={s:{} for s in ['GC','OSS']}
files=sorted((ROOT/'protein_cluster_pipeline/results/diamond_mcl_mapping/MAG/counts').glob('*.gene_counts.tsv'))
for i,f in enumerate(files):
 study='GC' if f.name.startswith('GC_') else 'OSS'
 d=pd.read_csv(f,sep='\t');raw=str(d['sample'].iloc[0]);sample='sample'+raw.split('_S')[0] if study=='GC' else raw
 assert (d.length>0).all() and (d.mapped_reads>=0).all()
 d['rpk']=d.mapped_reads/d.length*1000;denom=d.rpk.sum();sub=d[d.gene_id.isin(keep)].copy()
 lengths.update(dict(zip(sub.gene_id,sub.length)))
 for level,col in levels.items():
    z=sub.merge(links[level],left_on='gene_id',right_on='catalog_gene_id')
    matrices[study][level][sample]=z.groupby(col).rpk.sum()/denom*1e6 if denom else z.groupby(col).rpk.sum()*np.nan
 cc=sub.merge(cw[['catalog_gene_id','local_cluster_id']],left_on='gene_id',right_on='catalog_gene_id')
 clusters[study][sample]=cc.groupby('local_cluster_id').rpk.sum()/denom*1e6
 all_qc.append(dict(study=study,sample_id=sample,reference_mapped_reads=int(d.mapped_reads.sum()),target_locus_reads=int(sub.mapped_reads.sum()),reference_RPK=denom,target_fraction_reads=sub.mapped_reads.sum()/max(d.mapped_reads.sum(),1)))
 if i%15==0:print('Read-count files processed',i+1,'/',len(files),flush=True)
save(pd.DataFrame(all_qc),'mapping_sample_QC.tsv')
for study in matrices:
 for level in levels:save(pd.DataFrame(matrices[study][level]).fillna(0),f'{study}_direct_{level}_TPM.tsv','inputs',True)
 save(pd.DataFrame(clusters[study]).fillna(0),f'{study}_direct_target_cluster_TPM.tsv','inputs',True)
 m=rd(f'functional_analysis_results/maaslin2_inputs/MAG_{study}_MetaG.tsv');save(m,f'{study}_MAG_abundance.tsv','inputs',True)
 for level,col in levels.items():
    copy=hits[['gene_id','MAG',col]].drop_duplicates().groupby(['MAG',col]).size().unstack(fill_value=0).reindex(m.index,fill_value=0)
    inf=copy.T.dot(m);save(inf,f'{study}_inferred_{level}_abundance.tsv','inputs',True)
    save(copy,f'MAG_{level}_copy_number.tsv','inputs',True)
gc=pd.read_excel(ROOT/'GC_metadata_omics_updated_afaf.xlsx').rename(columns={'sample_ID':'sample_id','cropping':'shrub','organicMatter':'organic_matter'})
oss=pd.read_csv(ROOT/'metadata_OSS_PLFA_soils_afaf.csv').rename(columns={'sample_ID':'sample_id','type':'sample_type'})
oss['shrub']=oss.shrub.replace({'shrub':'Shrub'})
oss['context']=np.where(oss.sample_type=='Soil',oss.season+'_Soil','Rainy_'+oss.sample_type)
oss['candidate_plot']=oss.sample_id.str.replace(r'[DERS](?:\d+)?$','',regex=True)
mdqc=[];miss=[]
for study,d in [('GC',gc),('OSS',oss)]:
 assert not d.sample_id.duplicated().any()
 samples=pd.DataFrame(matrices[study]['gene']).columns
 for s in sorted(set(d.sample_id)|set(samples)):mdqc.append(dict(study=study,sample_id=s,in_metadata=s in set(d.sample_id),in_mapping=s in samples))
 d=d.set_index('sample_id').loc[samples];d.index.name='sample_id';save(d,f'{study}_metadata.tsv','inputs',True)
 for c in d:
    miss.append(dict(study=study,variable=c,n=len(d),observed=d[c].notna().sum(),distinct=d[c].nunique(),dtype=str(d[c].dtype)))
 factors=['shrub','watering','organic_matter','phase','block'] if study=='GC' else ['shrub','fertilizer','context','block']
 save(d.groupby(factors,observed=True).size().rename('n').reset_index(),f'{study}_design_cells.tsv')
save(pd.DataFrame(mdqc),'sample_matching_audit.tsv');save(pd.DataFrame(miss),'metadata_missingness.tsv')
save(hits[['image_gene','pathway','mechanism']].drop_duplicates(),'feature_labels.tsv')
print('Targets',len(targets),'detected',sum(x['annotated_loci']>0 for x in coverage),'unique loci',hits.gene_id.nunique(),'MAGs',hits.MAG.nunique(),flush=True)
