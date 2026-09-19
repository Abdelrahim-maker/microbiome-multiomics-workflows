"""Readable deliverables, annotated result tables, figures, workbook and provenance."""
from pathlib import Path
import hashlib,json,html,sys,platform,subprocess,tempfile,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
def tab(n):return pd.read_csv(OUT/'tables'/n,sep='\t')
def save(d,n):d.to_csv(OUT/'tables'/n,sep='\t',index=False)
def mdtable(d):
 d=d.copy().fillna('')
 for c in d.select_dtypes(include='number'):d[c]=d[c].map(lambda x:f'{x:.4g}')
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in d.to_numpy())
def figsave(fig,name):
 fig.savefig(OUT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
hits=tab('image_gene_locus_cluster_MAG_taxonomy.tsv');targets=tab('image_target_crosswalk.tsv');det=tab('image_target_detection.tsv')
labels=hits.groupby('image_gene').agg(pathways=('pathway',lambda s:'; '.join(sorted(set(s)))),mechanisms=('mechanism',lambda s:'; '.join(sorted(set(s))))).reset_index()
eff=tab('adjusted_feature_models.tsv');community=tab('adjusted_community_tests.tsv');sens=tab('joint_chemistry_and_plot_sensitivity_features.tsv');unit=tab('experimental_unit_sensitivity_community.tsv')
main=eff[(eff.analysis=='main')&(eff.level=='gene')].merge(labels,left_on='feature',right_on='image_gene',how='left')
main['contrast']=main.coefficient
contrasts={'C(shrub)[T.Shrub]':'Shrub minus no shrub','C(organic_matter)[T.noOM]':'No OM minus OM','C(watering)[T.watered]':'Watered minus drought','C(phase)[T.droughtStart]':'Drought start minus end',
 'C(fertilizer)[T.1x]':'Fertilizer 1x minus 0x','C(context)[T.Rainy_Endo]':'Rainy endosphere minus dry soil','C(context)[T.Rainy_Rhizo]':'Rainy rhizosphere minus dry soil','C(context)[T.Rainy_Soil]':'Rainy soil minus dry soil'}
main['contrast']=main.coefficient.replace(contrasts)
for name,df in [('pseudocount_0.1',eff),('pseudocount_1.0',eff),('within_panel_CLR',sens),('main_candidate_plot_sensitivity',eff)]:
 z=df[(df.analysis==name)&(df.level=='gene')][['study','feature','coefficient','beta','q']].rename(columns={'beta':name+'_beta','q':name+'_q'})
 main=main.merge(z,on=['study','feature','coefficient'],how='left')
main['pseudocount_robust']=(main.q<.05)&(main['pseudocount_0.1_q']<.05)&(main['pseudocount_1.0_q']<.05)&(main.beta*main['pseudocount_0.1_beta']>0)&(main.beta*main['pseudocount_1.0_beta']>0)
main['within_panel_supported']=(main.q<.05)&(main.within_panel_CLR_q<.05)&(main.beta*main.within_panel_CLR_beta>0)
save(main,'annotated_gene_results_with_robustness.tsv')
count=main.groupby(['study','contrast']).agg(tested_labels=('feature','size'),q05_labels=('q',lambda s:int((s<.05).sum())),pseudocount_robust=('pseudocount_robust','sum'),within_panel_supported=('within_panel_supported','sum')).reset_index();save(count,'gene_association_summary.tsv')
# Explicitly identify identical annotation sets rather than implying independence.
equiv={}
for gene,z in hits.groupby('image_gene'):equiv.setdefault(tuple(sorted(z.gene_id.unique())),[]).append(gene)
eq=pd.DataFrame([dict(labels=';'.join(v),n_labels=len(v),n_shared_loci=len(k)) for k,v in equiv.items() if len(v)>1]);save(eq,'indistinguishable_image_labels.tsv')
paths=[]
for pathway,z in targets.groupby('pathway'):
 detected=det[(det.pathway==pathway)&(det.annotated_loci>0)].image_gene.tolist()
 note='Marker abundance; no complete-pathway or net-carbon-flux inference'
 if pathway=='Psl polysaccharide':note='Only shared pslB/algA precursor detected; does NOT establish Psl synthesis'
 if pathway=='Pel polysaccharide':note='No targeted Pel annotations detected; biological absence not established'
 if pathway in ['Broad glycosyltransferases','Cellulose degradation','Hemicellulose degradation','Aromatic degradation','Starch degradation']:note+='; CAZy families are functionally broad'
 paths.append(dict(pathway=pathway,image_labels=len(z),detected_labels=len(detected),detected_markers=';'.join(detected),interpretation=note))
scope=pd.DataFrame(paths);save(scope,'pathway_evidence_scope.tsv')
# Co-occurrence of specified core markers, not a full pathway-completeness claim.
cores={'Alginate core':['alg8','alg44'],'Cellulose core':['bcsA','bcsB'],'PNAG core':['pgaC','pgaD'],'Psl core':['pslA','pslD'],'Pel core':['pelA','pelF'],
 'PHA synthesis core':['phaA','phaB','phaC'],'Glycogen synthesis core':['glgC','glgA','glgB'],'Trehalose Ots route':['otsA','otsB'],'Trehalose TreYZ route':['treY','treZ'],
 'Catechol ortho markers':['catA','catB','catC'],'Protocatechuate markers':['pcaG','pcaH','pcaB','pcaC','pcaD']}
present=hits.groupby('MAG').image_gene.apply(set);core=[]
for route,genes in cores.items():
 for mag,found in present.items():core.append(dict(route=route,MAG=mag,required_markers=';'.join(genes),detected_markers=';'.join(sorted(set(genes)&found)),all_core_markers=set(genes).issubset(found)))
core=pd.DataFrame(core);save(core,'MAG_core_marker_cooccurrence.tsv');core_summary=core.groupby('route').all_core_markers.sum().reset_index(name='MAGs_with_all_core_markers');save(core_summary,'core_marker_summary.tsv')

short={'C(shrub)[T.Shrub]':'Shrub − no shrub','C(organic_matter)[T.noOM]':'No OM − OM','C(watering)[T.watered]':'Watered − drought','C(phase)[T.droughtStart]':'Start − end',
 'C(fertilizer)[T.1x]':'Fertilizer 1x − 0x','C(context)[T.Rainy_Endo]':'Rainy endo − dry soil','C(context)[T.Rainy_Rhizo]':'Rainy rhizo − dry soil','C(context)[T.Rainy_Soil]':'Rainy soil − dry soil'}
for study in ['GC','OSS']:
 for level in ['pathway','gene']:
  a=eff[(eff.study==study)&(eff.level==level)&(eff.analysis=='main')].copy()
  if level=='gene':
   keep=a.groupby('feature').q.min().nsmallest(35).index;a=a[a.feature.isin(keep)]
  a.coefficient=a.coefficient.replace(short);pv=a.pivot(index='feature',columns='coefficient',values='beta');qv=a.pivot(index='feature',columns='coefficient',values='q')
  if 'Psl polysaccharide' in pv.index:pv=pv.rename(index={'Psl polysaccharide':'Psl shared precursor only'});qv=qv.rename(index={'Psl polysaccharide':'Psl shared precursor only'})
  if level=='gene':
   from matplotlib.patches import Patch
   contrast_palette={
    'Shrub − no shrub': {'pos':'#2e8b57','neg':'#8b5a2b'},
    'No OM − OM': {'pos':'#f4d03f','neg':'#f39c12'}
   }
   fig,axes=plt.subplots(1,2,figsize=(16,8),sharey=True)
   for ax,(contrast,colors) in zip(axes,contrast_palette.items()):
    d=a[a.coefficient==contrast].sort_values('beta',key=lambda s:s.abs(),ascending=False).head(18).copy()
    if d.empty: continue
    x=np.arange(len(d))
    point_colors=np.where(d.beta>=0,colors['pos'],colors['neg'])
    ax.vlines(x,0,d.beta,color=point_colors,linewidth=1.5,alpha=0.9)
    ax.scatter(x,d.beta,s=52,color=point_colors,edgecolor='black',linewidth=0.5,zorder=3)
    for xi,val,feat,q in zip(x,d.beta,d.feature,d.q):
     if q<.05:
      offset=0.08 if val>=0 else -0.08
      ax.text(xi,val+offset,'*',ha='center',va='bottom' if val>=0 else 'top',fontsize=12,color='black')
    ax.axhline(0,color='black',linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d.feature,rotation=90)
    ax.set_ylabel('Coefficient (beta)')
    ax.set_xlabel('Gene')
    ax.set_title(f'{study}: {contrast}')
   legend_handles=[Patch(facecolor='#2e8b57',label='Shrub'),Patch(facecolor='#8b5a2b',label='No shrub'),Patch(facecolor='#f39c12',label='OM'),Patch(facecolor='#f4d03f',label='No OM')]
   fig.legend(handles=legend_handles,loc='lower center',ncol=4,frameon=False)
   fig.suptitle(f'{study}: gene effects, lollipop view\nColor indicates direction of association within the modeled contrast',fontsize=12)
   fig.tight_layout(rect=[0,0.05,1,0.98]);figsave(fig,f'{study}_gene_effects')
  else:
   annot=pv.applymap(lambda x:f'{x:.2f}')+qv.applymap(lambda q:'*' if q<.05 else '')
   fig,ax=plt.subplots(figsize=(12,max(7,len(pv)*.29)));sns.heatmap(pv,center=0,cmap='vlag',annot=annot if level=='pathway' else qv.applymap(lambda q:'*' if q<.05 else ''),fmt='',ax=ax,cbar_kws={'label':'Adjusted log2(TPM + 0.5) effect'})
   ax.set_title(f'{study}: {level} marker associations\n* BH q < 0.05; relative abundance, conditional on recorded design');ax.set(xlabel='',ylabel='');ax.tick_params(axis='x',rotation=25);plt.setp(ax.get_xticklabels(),ha='right');fig.tight_layout();figsave(fig,f'{study}_{level}_effects')

# Dominant direct-read MAG carriers by function and treatment.
features=tab('feature_labels.tsv')
direct=tab('direct_read_MAG_gene_contributions.tsv').merge(features,left_on='gene',right_on='image_gene',how='left')
direct['function_group']=direct['mechanism']
direct.loc[direct.gene.str.contains('glg', case=False, na=False), 'function_group'] = 'Glycogen'
direct.loc[direct.gene.str.contains('pha', case=False, na=False), 'function_group'] = 'PHA'
direct.loc[direct.gene.str.contains(r'^(ots|tre)', case=True, na=False), 'function_group'] = 'Trehalose'
selected = direct[(direct.study=='GC') & (direct.function_group.isin(['EPS production','Biomass and necromass','Glycogen','PHA','Trehalose']))].copy()
summary = selected.groupby(['function_group','shrub','MAG'], as_index=False).agg(mean_direct_TPM=('mean_direct_TPM','sum'), genes=('gene','nunique'))
summary['treatment_group'] = summary['shrub']
labels = direct[['MAG','phylum','genus']].drop_duplicates().copy()
labels['tax_label'] = labels['MAG'] + '\n' + labels['genus'].fillna('unclassified') + ' (' + labels['phylum'].fillna('unclassified') + ')'
mag_tax = labels.set_index('MAG')['tax_label']
for treatment in ['Shrub','noShrub']:
    top = summary[summary.shrub==treatment].sort_values(['function_group','mean_direct_TPM'], ascending=[True,False]).groupby('function_group').head(5)
    pivot = top.pivot(index='function_group', columns='MAG', values='mean_direct_TPM').fillna(0)
    if pivot.empty: continue
    pivot = pivot.rename(columns=mag_tax)
    fig, ax = plt.subplots(figsize=(max(8, 0.95*len(pivot.columns)+3), 5.2))
    sns.heatmap(pivot, cmap='YlGnBu', vmin=0, vmax=max(pivot.to_numpy().max(), 1), annot=np.round(pivot.values, 1), fmt='g', ax=ax, cbar_kws={'label':'mean direct TPM'})
    ax.set_title(f'GC dominant MAG carriers: {treatment}')
    ax.set_xlabel('MAG (genus; phylum)')
    ax.set_ylabel('Carbon function')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    figsave(fig, f'GC_MAG_dominant_carriers_{treatment.replace(" ","_").lower()}')

# OM/no-OM split using the same sample-level gene-to-MAG mapping logic as the direct-read audit.
meta = pd.read_csv('inputs/GC_metadata.tsv', sep='\t', index_col=0).reset_index().rename(columns={'index':'sample_id'})
counts_dir = ROOT / 'protein_cluster_pipeline' / 'results' / 'diamond_mcl_mapping' / 'MAG' / 'counts'
links = pd.read_csv(OUT / 'tables' / 'image_gene_locus_cluster_MAG_taxonomy.tsv', sep='\t')[['catalog_gene_id','image_gene','MAG']].dropna().drop_duplicates()
keep = set(links['catalog_gene_id'])
rows=[]
for count_file in sorted(counts_dir.glob('GC*.gene_counts.tsv')):
    d = pd.read_csv(count_file, sep='\t')
    raw = str(d['sample'].iloc[0])
    sample_id = 'sample' + raw.split('_S')[0]
    om = meta.loc[meta['sample_id'] == sample_id, 'organic_matter'].iloc[0]
    d['rpk'] = d['mapped_reads'] / d['length'] * 1000
    total = d['rpk'].sum()
    if total == 0:
        continue
    z = d[d['gene_id'].isin(keep)][['gene_id','rpk']].merge(links, left_on='gene_id', right_on='catalog_gene_id')
    z = z.groupby(['image_gene','MAG'], as_index=False).agg(rpk=('rpk','sum'))
    z['rpk'] = z['rpk'] / total * 1e6
    z['sample_id'] = sample_id
    z['organic_matter'] = om
    rows.append(z)
if rows:
    gc_direct_om = pd.concat(rows, ignore_index=True)
    gc_direct_om = gc_direct_om.merge(features[['image_gene','mechanism']], on='image_gene', how='left')
    gc_direct_om['function_group'] = gc_direct_om['mechanism']
    gc_direct_om.loc[gc_direct_om['image_gene'].str.contains('glg', case=False, na=False), 'function_group'] = 'Glycogen'
    gc_direct_om.loc[gc_direct_om['image_gene'].str.contains('pha', case=False, na=False), 'function_group'] = 'PHA'
    gc_direct_om.loc[gc_direct_om['image_gene'].str.contains(r'^(ots|tre)', case=True, na=False), 'function_group'] = 'Trehalose'
    gc_direct_om = gc_direct_om[(gc_direct_om['function_group'].isin(['EPS production','Biomass and necromass','Glycogen','PHA','Trehalose']))].copy()
    om_summary = (gc_direct_om.groupby(['function_group','organic_matter','MAG'], as_index=False)
        .agg(mean_direct_TPM=('rpk','sum')))
    for treatment in ['OM','noOM']:
        top = om_summary[om_summary['organic_matter']==treatment].sort_values(['function_group','mean_direct_TPM'], ascending=[True,False]).groupby('function_group').head(5)
        pivot = top.pivot(index='function_group', columns='MAG', values='mean_direct_TPM').fillna(0)
        if pivot.empty:
            continue
        pivot = pivot.rename(columns=mag_tax)
        fig, ax = plt.subplots(figsize=(max(8, 0.95*len(pivot.columns)+3), 5.2))
        sns.heatmap(pivot, cmap='YlGnBu', vmin=0, vmax=max(pivot.to_numpy().max(), 1), annot=np.round(pivot.values, 1), fmt='g', ax=ax, cbar_kws={'label':'mean direct TPM'})
        ax.set_title(f'GC dominant MAG carriers: {treatment}')
        ax.set_xlabel('MAG (genus; phylum)')
        ax.set_ylabel('Carbon function')
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        figsave(fig, f'GC_MAG_dominant_carriers_{treatment.lower()}')

# Metadata ranking shows non-significance rather than picking a nominal winner.
e=tab('metadata_community_associations.tsv');e=e[(e.study==study)&(e.variable!='chemistry_PC1_PC2_joint')].sort_values('partial_R2',ascending=False).head(15)
fig,ax=plt.subplots(figsize=(8,6));ax.barh(e.variable[::-1],e.partial_R2[::-1],color='#798c9b');ax.set(xlabel='Partial R² after recorded design adjustment',title=f'{study}: measured metadata (all displayed q ≥ 0.05)');fig.tight_layout();figsave(fig,f'{study}_metadata_associations')
cr=tab('cross_project_shrub_comparison.tsv');a=cr[cr.level=='gene'];fig,ax=plt.subplots(figsize=(7,6));good=(a.GC_q<.05)&(a.OSS_q<.05)&a.same_direction
ax.scatter(a.GC_beta,a.OSS_beta,c=np.where(good,'#16816d','#aeb6be'),alpha=.8)
for r in a[good].itertuples():ax.annotate(r.feature,(r.GC_beta,r.OSS_beta),xytext=(4,4),textcoords='offset points',fontsize=9)
ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set(xlabel='GC adjusted shrub effect',ylabel='OSS adjusted shrub effect',title='Shrub responses across projects\nGreen: same direction and q < 0.05 in both primary models');fig.tight_layout();figsave(fig,'cross_project_shrub_effects')

genecommunity=community[(community.analysis=='main')&(community.level=='gene')][['study','term','n','partial_R2','R2_total','p','q']]
rn=tab('DNA_adjusted_RNA_models.tsv');env=tab('metadata_gene_pathway_associations.tsv');vpart=tab('variance_partitioning.tsv')
rep=cr[(cr.level=='gene')&(cr.GC_q<.05)&(cr.OSS_q<.05)&cr.same_direction]
unitrows=unit[(unit.level=='gene')&(unit.analysis.isin(['within_candidate_plot_context','soil_season_within_candidate_plot']))]
siggenes=main[main.q<.05].sort_values(['study','q'])
direct=tab('direct_read_MAG_gene_contributions.tsv');topdirect=direct.sort_values('mean_direct_TPM',ascending=False).groupby(['study','shrub','gene']).head(3);save(topdirect,'top_direct_read_gene_carriers.tsv')
qmap=tab('reference_mapping_concentration_QC.tsv');qmap=qmap.merge(pd.concat([pd.read_csv(OUT/'inputs'/f'{s}_metadata.tsv',sep='\t',index_col=0).rename_axis('sample_id').reset_index().assign(study=s) for s in ['GC','OSS']])[['study','sample_id','shrub']],on=['study','sample_id'])
readme=f'''# Carbon-circle genes and metadata: GC and OSS

The requested analyses have been run on the available annotations and read-count tables. Results are **associations conditional on the recorded sampling design**, not proof that an environmental measurement causes a gene or carbon-stabilization change. Pot/plot identities remain unconfirmed.

## Main findings

- **GC:** shrub and organic matter explain the clearest differences in target-gene composition. Partial R² is 0.298 for shrub and 0.305 for organic matter, with BH q = 0.001 for each. A shrub × organic-matter interaction is also supported (partial R² = 0.093; q = 0.003), so the shrub response depends on OM conditions. Watering and phase are not supported in the overall target-gene test after correction.
- **OSS:** the four observed sampling contexts explain the strongest target-gene difference (partial R² = 0.449; q = 0.0009). This is not an independently crossed season-by-compartment experiment. The within-candidate-plot sensitivity supports context and the soil-only seasonal contrast. Context also differs in within-group dispersion (q = 0.0015), so the pattern includes heterogeneity as well as possible shifts in group position; the PERMANOVA result cannot be attributed solely to centroid separation.
- **Measured metadata:** none of the individual soil/plant variables survives the broad gene-level screen after BH correction. None survives the community-level metadata screen either. Do not select the lowest nominal p-value and call that variable a driver. The joint chemistry-PC analyses are a separate, lower-dimensional exploratory analysis.
- **Cross-project consistency:** {len(rep)} gene/family labels have same-direction shrub associations with q < 0.05 in both primary models: {', '.join(rep.feature)}. These are not independent replication experiments with identical conditions, and broad CAZy-family labels do not specify one substrate.
- **RNA:** {rn[rn.level=='gene'].feature.nunique()} gene labels and {rn[rn.level=='pathway'].feature.nunique()} pathway marker sets met the detection criteria. RNA results are exploratory because target reads are sparse despite large total shared-reference libraries. Four pathway associations pass q < 0.05 at pseudocount 0.5, but none passes at pseudocount 1.0; they are not robust transcriptional findings.

## Files to open

- [Results workbook](RESULTS.xlsx): main gene results, robustness, pathways, metadata, taxonomy and QC in separate sheets.
- [Interactive browser report](REPORT.html): searchable significant-gene table and figures.
- [Annotated gene results with robustness](tables/annotated_gene_results_with_robustness.tsv): all main gene contrasts, confidence intervals and sensitivity results.
- [Direct read gene carriers](tables/top_direct_read_gene_carriers.tsv): three leading MAG contributors for each gene and shrub group.
- [Image audit](tables/image_target_crosswalk.tsv) and [pathway evidence limits](tables/pathway_evidence_scope.tsv).

## Community tests

{mdtable(genecommunity)}

Partial R² is the fraction of reduced-model residual variation associated with a term; it is **not** the fraction of total community variation. `R2_total` supplies the latter denominator. Marginal term fractions need not sum to one.

## Individual gene/family labels

{mdtable(count)}

Counts refer to image labels, not independent biochemical functions. For example, amiA/amiB/amiC share a KO, treA/treF share a KO, and pslB/algA share a precursor function. [Identical annotation sets](tables/indistinguishable_image_labels.tsv) are listed explicitly. Positive shrub effects mean higher abundance with shrubs. Positive organic-matter coefficients mean **higher without OM**, because that contrast is noOM minus OM.

The primary abundance models use the entire MAG CDS reference as their denominator. `within_panel_supported` checks whether the association also occurs relative to the other target markers. This distinguishes a shift of the whole target panel relative to the reference from redistribution among its members. It does not provide absolute abundance. Reference mapping is highly concentrated in some OSS samples: the largest single-MAG read fraction is 98.4%, and the largest top-ten-locus fraction is 31.3%. Large reference-normalized shifts need interpretation alongside these diagnostics and the within-panel results.

## Experimental-unit sensitivity

{mdtable(unitrows[['analysis','n','partial_R2','p','q']])}

OSS candidate plots were inferred by removing compartment suffixes from sample IDs. Their treatment/block labels are consistent, but the interpretation has not been confirmed. CSC3E1 and CSC3E2 are retained as separate recorded samples, grouped into candidate plot CSC3 in cluster-robust analyses. CSA3E is absent from the metadata. Complete-context plot analyses average duplicates within a context, then average contexts equally; only ten candidate plots have every context, so these tests have low power. The complete-plot feature models were not estimable with the prespecified minimum residual degrees of freedom and are recorded as skipped.

GC has 24 droughtStart and 24 droughtEnd samples. Phase is partially confounded with block: only block2 includes both phases. Sample-number suffixes are not assumed to identify repeated pots because their recorded blocks do not establish that relationship. The GC p-values remain conditional on independent sampling units within recorded block/phase; they should be revised if repeated-pot identities are supplied.

## Target and sample audit

- 48 GC and 58 OSS metagenomes matched exactly to metadata; six OSS metadata rows lack the corresponding analyzed metagenomes and are listed in the sample audit.
- 153 image labels were transcribed; 110 have detected annotations, covering 23,867 distinct loci across 262 MAGs. Five loci have no cluster-catalog mapping and are unavailable for direct quantification. One MAG, 2021_COA1R.bin.17, lacks the available DRAM annotations and must not be interpreted as gene-negative.
- All detected markers were quantified before prevalence filtering. The main gene models retain 108 GC and 110 OSS labels.
- Existing DRAM annotations were combined across both batches. Existing threshold-validated KOfam calls replace the similarity-only calls for the EPS KO panel previously scanned. No new de novo annotation scan was performed; unresolved mappings and nondetections are not proof of biological absence.
- The only detected Psl label is pslB/algA, a shared precursor enzyme. Its abundance **does not establish Psl synthesis**. No targeted Pel annotations were detected. CAZy-family aggregates are broad marker sets. Pathway sums are deduplicated within each pathway, but markers may belong to more than one pathway; pathways are therefore not independent or additive.
- [Core-marker co-occurrence](tables/core_marker_summary.tsv) describes prespecified marker sets within MAGs. It does not establish a complete operon, full pathway activity or carbon sequestration.

## Methods

Read counts from the existing competitive MAG CDS mappings were divided by locus length, summed once per locus within each image label/pathway, and normalized to one million RPK across the **full MAG CDS reference**. This is relative TPM within the reference, not absolute cells, whole-soil activity or a fraction of all shotgun reads. Mapping-concentration diagnostics and the existing alignment/counting protocol are included in provenance. MAG-derived abundance × copy-number estimates are analyzed separately as inferred potential. Taxonomic contribution tables include both inferred contributions and the direct read-supported contributions.

The primary study-specific formulas are:

```text
GC:  log2(TPM + 0.5) ~ shrub + watering + organic matter + phase + block
OSS: log2(TPM + 0.5) ~ shrub + fertilizer + observed context + block
```

Ordinary least squares uses HC3 robust standard errors and t inference. MaAsLin 3 was suggested initially, but these runs use transparent adjusted relative-abundance models rather than claiming a MaAsLin run. Features require presence in at least 10% of samples and at least five samples; rank-deficient models or models with fewer than eight residual degrees of freedom are skipped. Pseudocounts 0.1 and 1.0, within-panel CLR, OSS candidate-plot clustered covariance, phase/context subsets and joint chemistry-PC adjustment provide sensitivity checks. Clustered covariance uses CR1 and candidate-plot count minus one degrees of freedom; inference with few clusters is exploratory.

Separate interaction models add shrub × OM, shrub × watering and watering × phase in GC, and shrub × context and shrub × fertilizer in OSS. Community interaction tests and gene/pathway interaction coefficients are included in the workbook and [all adjusted feature models](tables/adjusted_feature_models.tsv). Main-effect models summarize additive average associations; the supported GC shrub × OM interaction should guide interpretation of individual treatment combinations.

Community tests use full Hellinger-transformed abundance profiles, not just their first ordination axes. Marginal nested-model pseudo-F tests are Euclidean-distance PERMANOVA/RDA tests with 1,999 Freedman–Lane residual permutations. Permutations stay within block, and within block × phase for GC treatment tests. Selected candidate-plot checks use 4,999 permutations. Dispersion diagnostics test distances to group spatial medians with bias correction and the same adjusted permutation framework. These diagnostics are reported separately and do not erase potential location/dispersion ambiguity.

BH correction is applied across coefficients/features within study × feature level × analysis for the primary gene models; across all measured-variable/feature combinations within study × level for the metadata screen; and across the corresponding prespecified community test families. All tables identify their analysis family. Sensitivity analyses are not independent confirmations. Additional numerical soil/plant variables are screened **one at a time after design adjustment**, so these coefficients are not mutually adjusted for all other measured metadata. Gene-derived metadata summaries are excluded to avoid circular prediction.

Variance partitioning uses adjusted R², conditions on block and compares design variables with two standardized chemistry principal components. GC uses Al, Ca, Cu, Fe, K, Mg, Mn, Na, P and Zn (47 complete samples). OSS uses pH, carbon and nitrogen (26 complete samples). Negative adjusted fractions are retained. Shared explained variation is not evidence of mediation or causation. Adding post-treatment chemistry can remove part of a treatment-associated signal, so the design-only and chemistry-adjusted estimates answer different questions.

GC RNA and DNA use existing mappings against the same shared-cluster reference. Clusters ambiguously linked to several image labels are excluded from label-level RNA aggregation. Models relate log2(RNA CPM + 0.5) to log2(DNA CPM + 0.5), recorded design and block, requiring at least ten RNA-positive and ten DNA-positive samples and at least 20 usable samples. Zeros from a nonempty reference library are retained as nondetections. Pseudocount sensitivity is supplied. These sparse marker-transcription models do not demonstrate enzyme activity or carbon flux.

## Interpretation

Your strongest current result is that **GC shrub/organic-matter treatments and OSS sampling context are associated with microbial functional differences**. The available broad metadata screen does not isolate a single measured environmental cause. Functional genes are evidence of potential, RNA supplies limited transcriptional evidence, and carbon stabilization itself requires appropriate carbon-fraction, aggregate-protection or flux measurements. The image's labels “direct” and “opposes stabilization” are hypotheses, not signed carbon-balance measurements.

## Reproduction and sources

Run [scripts/run_all.sh](scripts/run_all.sh) from the repository root. Generated tables are in `tables/`, analysis matrices in `inputs/`, PNG/PDF figures in `figures/`, and source definitions/manifests in `sources/`. Existing raw data and analyses are preserved.

- [KEGG KO definitions](https://rest.kegg.jp/list/ko), downloaded 2026-09-10 and archived locally; ambiguous aliases were explicitly disambiguated by function.
- [Vegan marginal PERMANOVA documentation](https://vegandevs.github.io/vegan/reference/adonis.html).
- [Vegan adjusted variance partitioning](https://vegandevs.github.io/vegan/reference/varpart.html).
- [CAZy family interpretation](https://www.cazy.org/Glycoside-Hydrolases).
- [MaAsLin 3](https://github.com/biobakery/Maaslin3), an alternative framework discussed during planning, not the software used for these fits.
'''
(OUT/'REPORT.md').write_text(readme)
(OUT/'README.md').write_text('# Carbon-circle analysis\n\nStart with [REPORT.md](REPORT.md), [REPORT.html](REPORT.html), or [RESULTS.xlsx](RESULTS.xlsx).\n\nAll results are conditional associations; unresolved sample-unit questions are documented in the report.\n')
# A portable browser document with an interactive searchable table and local images.
body='<h1>Carbon-circle genes: GC and OSS</h1><p>Completed relative-abundance analyses. Associations are conditional on recorded design; pot/plot identities remain unconfirmed.</p>'
body+='<p><a href="RESULTS.xlsx">Download Excel workbook</a> · <a href="REPORT.md">Full methods and interpretation</a></p>'
body+='<h2>Main findings</h2><p>GC: shrub and organic matter. OSS: sampling context; soil season is supported in the candidate-plot sensitivity. No individual measured metadata variable survived the broad gene-association screen.</p>'
body+='<h2>Community associations</h2>'+genecommunity.to_html(index=False,float_format=lambda x:f'{x:.4g}')
for name in ['GC_pathway_effects','OSS_pathway_effects','cross_project_shrub_effects','GC_metadata_associations','OSS_metadata_associations']:
 body+=f'<figure><img src="figures/{name}.png" alt="{html.escape(name)}"></figure>'
body+='<h2>Significant gene/family associations</h2><p>Search by gene, project, contrast or pathway. These primary-model results require review against the robustness columns in the workbook.</p><input id="search" placeholder="Search results…" aria-label="Search gene results">'
body+=siggenes[['study','feature','pathways','contrast','beta','CI_low','CI_high','q','pseudocount_robust','within_panel_supported']].to_html(index=False,table_id='genes',float_format=lambda x:f'{x:.4g}')
body+='<h2>Limits that matter</h2><p>153 image labels; 110 detected. Five loci lack a cluster mapping and one MAG lacks annotation coverage. Shared pslB/algA does not establish Psl production. Sparse RNA is exploratory. CSC3E1/CSC3E2 and GC pot identities need clarification; results do not establish causation.</p>'
htmlpage='<!doctype html><html><head><meta charset="utf-8"><title>GC and OSS carbon-circle analysis</title><style>body{font:16px system-ui;margin:32px auto;max-width:1250px;padding:0 24px;color:#19313c}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto;max-height:650px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#e6eff0}img{max-width:100%}input{padding:12px;width:90%;margin:12px 0}figure{margin:24px 0}</style></head><body>'+body+'<script>document.getElementById("search").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("#genes tbody tr").forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?"":"none")})</script></body></html>'
(OUT/'REPORT.html').write_text(htmlpage)
sheets={'Community':community,'Gene_summary':count,'Gene_main_robustness':main,'Pathway_main':eff[(eff.level=='pathway')&(eff.analysis=='main')],'Interactions':eff[eff.analysis.str.startswith('interaction_')],'Subsets':eff[(eff.analysis.str.startswith('subset_'))&(eff.level.isin(['gene','pathway']))],'Dispersion':tab('dispersion_diagnostics.tsv'),'Metadata_community':tab('metadata_community_associations.tsv'),'Metadata_gene_pathway':env,'Variance_partition':vpart,'Joint_chemistry_sensitivity':sens,'Unit_sensitivity':unit,'Cross_project_shrub':cr,'RNA_models':rn,'RNA_sensitivity':tab('RNA_pseudocount_sensitivity.tsv'),'Image_targets':targets,'Target_detection':det,'Pathway_scope':scope,'Core_marker_sets':core_summary,'Top_gene_carriers':topdirect,'Sample_audit':tab('sample_matching_audit.tsv'),'Metadata_missing':tab('metadata_missingness.tsv'),'Mapping_QC':qmap,'Validation':tab('validation_checks.tsv')}
with tempfile.TemporaryDirectory(prefix='carbon-circle-workbook-') as tmp:
 registry={}
 for name,d in sheets.items():
  p=Path(tmp)/(name+'.tsv');d.to_csv(p,sep='\t',index=False);registry[name]=str(p)
 reg=Path(tmp)/'sheets.json';reg.write_text(json.dumps(registry))
 subprocess.run([os.environ.get('CARBON_CIRCLE_IO_PYTHON','python'),str(OUT/'scripts/export_workbook.py'),str(reg),str(OUT/'RESULTS.xlsx')],check=True)
manifest=[]
sourcefiles=[ROOT/'GC_metadata_omics_updated_afaf.xlsx',ROOT/'metadata_OSS_PLFA_soils_afaf.csv',ROOT/'circular carbon stabilization circle.png',ROOT/'protein_cluster_pipeline/DIAMOND_MCL_READ_MAPPING.md']+list((OUT/'scripts').glob('*'))+list((OUT/'sources').glob('KEGG*'))
for f in sourcefiles:
 manifest.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
save(pd.DataFrame(manifest),'provenance_sha256.tsv')
(OUT/'sources/software_versions.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()},indent=2))
print('Report, figures and workbook complete',flush=True)
