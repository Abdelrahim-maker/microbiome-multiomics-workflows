"""Build annotated figures and an auditable report from completed statistics."""
from pathlib import Path
import json
import hashlib
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent.parent
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(name):return pd.read_csv(OUT/'tables'/name,sep='\t')
mv=read('community_PERMANOVA.tsv');disp=read('PERMDISP.tsv');eff=read('differential_features.tsv');variance=read('ordination_variance.tsv')
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
manifest=[]
def save(fig,name,source):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    manifest.append({'figure':name,'source':source})
def fmt(x):return f'{x:.3g}' if pd.notna(x) else 'NA'
def stattext(z):
    return '\n'.join(f"{r.term}: partial R2={r.partial_R2:.3f}, p={fmt(r.p)}, q={fmt(r.q)}" for r in z.itertuples())
for file in sorted((OUT/'tables').glob('*_PCA_scores.tsv')):
    stem=file.stem.removesuffix('_PCA_scores');study=stem.split('_')[0]
    level=stem[len(study)+1:];analysis='factorial' if study=='GC' else 'main'
    for subset in ['OM_only','noOM_only','Shrub_only','noShrub_only','rainy_only','soil_only']:
        if level.endswith('_'+subset):level=level[:-(len(subset)+1)];analysis=subset;break
    z=pd.read_csv(file,sep='\t');test=mv[(mv.study==study)&(mv.level==level)&(mv.analysis==analysis)]
    assert len(test),stem
    ds=disp[(disp.study==study)&(disp.level==level)&(disp.analysis==('group_dispersion' if analysis in ['main','factorial'] else analysis))]
    fig,ax=plt.subplots(figsize=(8.6,7.7));hue='shrub';style=None
    if study=='GC' and analysis=='factorial':z['group']=z.shrub+' / '+z.organic_matter;hue='group'
    elif study=='GC' and analysis in ['Shrub_only','noShrub_only']:hue='organic_matter'
    sns.scatterplot(data=z,x='PC1',y='PC2',hue=hue,style=style,s=62,ax=ax,palette='colorblind')
    v=variance[(variance.study==study)&(variance.level==(level if analysis in ['main','factorial'] else level+'__'+analysis))].iloc[0]
    ax.set(xlabel=f'PC1 ({v.PC1_percent:.1f}%)',ylabel=f'PC2 ({v.PC2_percent:.1f}%)',title=f'{study}: {level.replace("_"," ")} | {analysis.replace("_"," ")} (n={len(z)})')
    note=stattext(test)+'\nDispersion q: '+', '.join(f'{r.term}={fmt(r.q)}' for r in ds.itertuples())+'\nAdjusted full-space Hellinger tests; PCA axes are descriptive.'
    fig.subplots_adjust(bottom=.32);fig.text(.08,.025,note,fontsize=8,va='bottom')
    save(fig,stem+'_with_stats',f'{file.relative_to(OUT)}; tables/community_PERMANOVA.tsv; tables/PERMDISP.tsv')

# Re-render historical OSS coordinates with their matching historical tests.
old=ROOT/'OSS/01_MAG_MetaG_target_clusters/tables'
tests=pd.read_csv(old/'all_omnibus_and_pairwise_factor_tests.tsv',sep='\t')
tests.to_csv(OUT/'tables/previous_OSS_joint_RPCA_tests.tsv',sep='\t',index=False)
sources=[(old/'sample_scores_metadata.tsv','PC1','PC2','OSS_previous_joint_RPCA')]
for factor in ['shrub','fertilizer','season','sample_type']:
    sources.append((ROOT/f'OSS/adjusted_comparisons_no_CN/tables/OSS_{factor}_nuisance_residual_scores.tsv','rPC1','rPC2','OSS_previous_adjusted_'+factor))
for file,x,y,name in sources:
    z=pd.read_csv(file,sep='\t')
    for factor in (['shrub','fertilizer','season','sample_type'] if x=='PC1' else [name.removeprefix('OSS_previous_adjusted_')]):
        t=tests[(tests.factor==factor)&(tests.test_type=='omnibus')].iloc[0]
        fig,ax=plt.subplots(figsize=(8,6));sns.scatterplot(data=z,x=x,y=y,hue=factor,ax=ax,s=60,palette='colorblind')
        ax.set_title(f'OSS previous Joint-RPCA: {factor}')
        fig.subplots_adjust(bottom=.2)
        fig.text(.08,.035,f'Historical adjusted test: partial R2={t.partial_R2:.3f}; p={fmt(t.permutation_p)}; q={fmt(t.BH_q_within_model)}\nAdjusted for: {t.adjusted_for}. Tests use full joint coordinates, not only displayed axes.',fontsize=8)
        save(fig,name+'_'+factor,str(file.relative_to(ROOT))+'; tables/previous_OSS_joint_RPCA_tests.tsv')

for study in ['GC','OSS']:
    z=mv[(mv.study==study)&mv.analysis.isin(['main','factorial'])]
    fig,ax=plt.subplots(figsize=(11,5));sns.barplot(data=z,x='term',y='partial_R2',hue='level',ax=ax)
    ax.set(title=f'{study}: adjusted community effect sizes',ylabel='Partial R2',xlabel='');ax.tick_params(axis='x',rotation=15)
    ax.legend(fontsize=7);fig.tight_layout();save(fig,study+'_cross_level_effect_sizes','tables/community_PERMANOVA.tsv')
    for level in ['carbon_gene','carbon_pathway','carbon_category']:
        a=eff[(eff.study==study)&(eff.level==level)&eff.analysis.isin(['main','factorial'])]
        if study=='GC':a=a[a.contrast.isin(['OM_average','Shrub_average','OM_x_Shrub'])]
        else:a=a[a.contrast.str.contains('fertilizer|Shrub',regex=True)]
        p=a.pivot(index='feature',columns='contrast',values='beta');q=a.pivot(index='feature',columns='contrast',values='q')
        annot=p.copy().astype(str)
        for i in p.index:
            for c in p.columns:annot.loc[i,c]=f'{p.loc[i,c]:.2f}\nq={fmt(q.loc[i,c])}'
        fig,ax=plt.subplots(figsize=(max(8,len(p.columns)*2),max(4,len(p)*.43)))
        sns.heatmap(p,cmap='RdBu_r',center=0,annot=annot,fmt='',annot_kws={'size':7},ax=ax,cbar_kws={'label':'Adjusted log2(TPM + 0.5) difference'})
        ax.set_title(f'{study}: {level.replace("_"," ")} effects and FDR');fig.tight_layout();save(fig,f'{study}_{level}_effect_heatmap','tables/differential_features.tsv')
cp=pd.read_csv(OUT/'inputs/MAG_category_copy_number.tsv',sep='\t',index_col=0)
fig,ax=plt.subplots(figsize=(12,max(7,len(cp)*.13)))
sns.heatmap(np.log2(cp+1),cmap='viridis',ax=ax,yticklabels=True,cbar_kws={'label':'log2(locus copies + 1)'})
ax.tick_params(axis='y',labelsize=5);ax.set_title('MAG carbon mechanisms: annotated locus copy number');fig.tight_layout();save(fig,'MAG_carbon_mechanism_carriers','inputs/MAG_category_copy_number.tsv')
fert=read('fertilizer_all_pairwise_features.tsv');fm=read('fertilizer_community_tests.tsv')
summary=fert.groupby(['level','analysis','contrast']).agg(tested=('feature','size'),q_lt_005=('q',lambda x:int((x<.05).sum()))).reset_index()
summary.to_csv(OUT/'tables/fertilizer_significance_summary.tsv',sep='\t',index=False)
for level in ['carbon_gene','carbon_pathway','carbon_category']:
    a=fert[(fert.level==level)&(fert.analysis=='main')]
    p=a.pivot(index='feature',columns='contrast',values='beta');q=a.pivot(index='feature',columns='contrast',values='q')
    labels=p.astype(str)
    for i in p.index:
        for c in p.columns:labels.loc[i,c]=f'{p.loc[i,c]:.2f}\nq={fmt(q.loc[i,c])}'
    fig,ax=plt.subplots(figsize=(10,max(5,len(p)*.43)));sns.heatmap(p,center=0,cmap='RdBu_r',annot=labels,fmt='',annot_kws={'size':7},ax=ax)
    ax.set_title('OSS fertilizer: '+level.replace('_',' '));fig.tight_layout();save(fig,'OSS_fertilizer_'+level,'tables/fertilizer_all_pairwise_features.tsv')
manifest.extend(read('legacy_figure_provenance.tsv').to_dict('records'))
pd.DataFrame(manifest).to_csv(OUT/'tables/figure_provenance.tsv',sep='\t',index=False)
def table(d):
    def cell(v):return (f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v)).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
primary=mv[mv.scope.eq('primary_direct_and_MAG')&mv.analysis.isin(['factorial','main'])]
interaction=primary[primary.term.eq('OM_x_Shrub')]
sig=eff[eff.analysis.isin(['main','factorial'])].groupby(['study','level','contrast']).agg(tested=('feature','size'),significant=('q',lambda x:int((x<.05).sum()))).reset_index()
sig.to_csv(OUT/'tables/differential_significance_summary.tsv',sep='\t',index=False)
report=['# OM, shrub, fertilizer and carbon mechanisms','',
'See [RESULTS_INTERPRETATION.md](RESULTS_INTERPRETATION.md) for the biological findings and [VALIDATION.json](VALIDATION.json) for verification.','',
'## Status and scope','',
'Completed: separate MAG, direct carbon-gene, direct carbon-pathway and biological-category analyses; factorial OM x shrub tests; OM-only/noOM-only shrub ordinations; reciprocal shrub-stratified OM ordinations; context/watering/phase adjustment; dispersion diagnostics; differential effects with confidence intervals and FDR; all fertilizer pairwise contrasts and shrub interactions; MAG/taxonomy carrier crosswalk; carrier redundancy and paired cross-level effect-size bootstrap; PCA loadings; annotated historical OSS Joint-RPCA and nuisance-residual views. Broad KO/pathway analyses are supplementary MAG-copy-number inferences, not direct whole-metagenome measurements.','',
'The supplied requested steps are preserved in requested_analysis_plan.txt. Existing inputs and completed statistical tables were reused after inspection; fertilizer.py and finalize_story.py complete the previously unfinished package. Historical figures are preserved; annotated versions are in figures/.','',
'## Design and methods','',
'GC: 48 samples, OM and shrub coded -0.5/+0.5; model = OM + Shrub + OM:Shrub + watering + phase + block. Main effects are equal-weight averages across the other treatment; interaction is a difference of differences. OSS: 58 samples; model = shrub + fertilizer + observed season/compartment context + block. OM is a GC factor; fertilizer is an OSS factor. Context is not separately estimable from study across these different experiments. Dry root compartments are absent, so context combines observed season/compartment cells.','',
'Features require presence in at least max(5, ceil(10% of samples)) and nonzero variance. PCA and multivariate tests use Hellinger-transformed relative profiles. Full-dimensional reduced/full model residual permutations (4,999; seed 20260911, fertilizer seed 20260914) estimate marginal partial R2, total-denominator R2, pseudo-F and p. Permutations are within block and, for GC non-phase tests, within block x phase. Partial R2 values do not sum to one. Reported q uses BH within study/scope/analysis for community tests; fertilizer community q is within analysis across four levels.','',
'Dispersion diagnostics test adjusted distances to group spatial medians, with group-size bias correction and residual permutations. These are adjusted distance-dispersion diagnostics, not an exact replacement for every standard PERMDISP implementation. Significant dispersion can contribute to a multivariate separation. Feature models use log2(TPM + 0.5), HC3 standard errors and t confidence intervals. Effects describe relative abundance, not absolute expression or carbon flux. Feature BH families are all features and contrasts within study/level/analysis; fertilizer uses level/analysis. Primary direct features also have pseudocount 0.1 and 1.0 sensitivity fits. Categories deduplicate loci within each category; categories can overlap.','',
'OSS candidate_plot was inferred from sample names, not verified plot records. Candidate-plot CR1 sensitivity estimates are supplied, but block-permuted p values and row-bootstrap intervals remain conditional on sample exchangeability. Before manuscript claims, confirm OSS experimental-unit IDs and GC repeated-pot IDs, then refit restricted permutations/cluster models if necessary. This is a pending design confirmation, not evidence of independent replication.','',
'## Community results','',table(primary[['study','level','term','partial_R2','R2_total','p','q']]),'',
'## OM x shrub interaction','',table(interaction[['level','partial_R2','p','q']]),'',
'Within-OM and within-noOM shrub follow-ups appear in every primary-level *_OM_only_with_stats and *_noOM_only_with_stats figure. They are follow-up contrasts; significance in one subgroup and not the other is not itself an interaction test.','',
'## Fertilizer and genes of interest','',table(fm[['level','analysis','partial_R2','p','q']]),'',table(summary[summary.level.eq('carbon_gene')]),'',
'Fertilizer is associated with MAG composition (partial R2=0.0678; q=0.0008), but not with overall targeted carbon-gene, pathway or category composition after FDR correction. The main HC3 gene models do not identify FDR-supported fertilizer-responsive target genes; the sensitivity models and shrub interactions are tabulated separately.','',
'Use fertilizer_all_pairwise_features.tsv for each gene, fertilizer contrast, log-scale effect, 95% CI, p and q. Positive coefficients mean the first named fertilizer level in the contrast has higher adjusted abundance. Interaction contrasts report the shrub-minus-noShrub difference in fertilizer response. Unsupported q values mean insufficient evidence under this model, not proof of no fertilizer effect. Candidate-plot sensitivity rows should be considered alongside HC3 rows.','',
'## Carrier interpretation and redundancy','',
'carbon_gene_MAG_taxonomy_treatment_response.tsv links targets to MAG, taxonomy and treatment coefficients. MAG_carbon_mechanism_carriers displays encoded copy number, not treatment-induced genome gain/loss. The reference MAG repertoire is fixed, so these data cannot establish within-organism repertoire evolution. sample_gene_redundancy.tsv supplies abundance-weighted effective carrier numbers, detected carriers and dominant-carrier fractions.','',table(read('paired_cross_level_R2_bootstrap.tsv')),'',
'Cross-level differences are consistent with differential taxonomic/functional turnover, but do not prove functional redundancy: feature resolution, normalization, shared annotation and restricted carbon-target scope affect the comparison. Broad inferred functions mathematically depend on MAG abundance and are not independent confirmation. Biomass, aggregation and carbon persistence are mechanistic hypotheses requiring phenotype/flux measurements.','',
'## Ordination statistics and previous OSS figures','',
'Every PCA figure in this package includes matching adjusted partial R2, p and q plus available dispersion q values. Historical OSS raw Joint-RPCA and nuisance-residual coordinate views are re-rendered with matching full-coordinate statistics in figures/OSS_previous*. Those tests used 999 permutations and their original BH family. In addition, legacy_oss.py reanalyzes all 27 observed historical adjusted, C/N-adjusted, treatment-subset and fixed-stratum views with 4,999 permutations (seed 20260915), BH across views, and dispersion diagnostics. Original joint coordinates are retained; C/N subsets use complete cases. Nonestimable views are explicitly labeled. Empty dry-root cells have no samples and are not tested. See legacy_OSS_reanalysis.tsv, legacy_OSS_dispersion.tsv and figure_provenance.tsv.','',
'## Deliverables and reproduction','',
'- figures/: PNG and vector PDF figures.\n- tables/figure_provenance.tsv: each figure and its exact numerical sources.\n- tables/community_PERMANOVA.tsv and PERMDISP.tsv: all adjusted tests.\n- tables/differential_features.tsv: all gene/pathway/MAG/category estimates and sensitivity fits.\n- tables/category_definitions.tsv and category_locus_membership.tsv: mechanism definitions.\n- tables/fertilizer_all_pairwise_features.tsv: all fertilizer contrasts.\n- RESULTS.xlsx: key summary and result tables.\n- input_manifest.tsv: input checksums.','',
'Run from the workspace root:','',
'```bash\nbash "fnal results/manuscript_story/scripts/run_all.sh"\n```','',
'## Remaining work','',
'Confirm experimental-unit identities and fertilizer application units/rates before causal interpretation. All available categories have been analyzed; excluded/unresolved annotation targets remain documented in the parent report. No additional networks, WGCNA, machine learning or GEM analyses were added, as the requested plan classifies these as optional or separate work. Manuscript figure selection and mechanistic synthesis should follow the supported effects rather than the illustrative hypotheses in the attachment.','',
'## Figure gallery','']
for item in manifest:report+=['### '+item['figure'],'',f"![{item['figure']}](figures/{item['figure']}.png)",'']
(OUT/'REPORT.md').write_text('\n'.join(report))
subprocess.run([os.environ.get('CARBON_CIRCLE_IO_PYTHON','/usr/bin/python'),str(OUT/'scripts/export_story.py')],check=True)
inputs=[]
for p in sorted((OUT/'inputs').glob('*.tsv')):
    inputs.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(inputs).to_csv(OUT/'input_manifest.tsv',sep='\t',index=False)
(OUT/'README.md').write_text('# Manuscript analysis package\n\nStart with [REPORT.md](REPORT.md) or [RESULTS.xlsx](RESULTS.xlsx). Figures are in [figures](figures/). See the report for methods, findings, limitations and reproduction.\n')
print(f'Completed report, workbook and {len(manifest)} annotated figures',flush=True)
