"""Independent numerical and data-integrity checks for the completed analysis."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import patsy
from scipy import stats
from statsmodels.stats.multitest import multipletests
OUT=Path(__file__).resolve().parents[1];RNG=np.random.default_rng(19);NPERM=99
tree=ast.parse((OUT/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'helpers','exec'))
checks=[]
def check(name,condition,detail=''):
 checks.append(dict(check=name,passed=bool(condition),detail=str(detail)))
 if not condition:raise AssertionError(name+': '+str(detail))
effects=pd.read_csv(OUT/'tables/adjusted_feature_models.tsv',sep='\t')
for study in ['GC','OSS']:
 md=rd(f'{study}_metadata.tsv');md['shrub']=pd.Categorical(md.shrub,categories=['noShrub','Shrub'])
 x=rd(f'{study}_direct_gene_TPM.tsv');check(study+'_unique_samples',x.columns.is_unique and md.index.is_unique)
 check(study+'_sample_alignment',set(x.columns)==set(md.index))
 check(study+'_finite_nonnegative_abundance',np.isfinite(x.values).all() and (x.values>=0).all())
 check(study+'_TPM_bounds',(x.values<=1e6+1e-7).all())
 terms=['C(shrub)','C(watering)','C(organic_matter)','C(phase)','C(block)'] if study=='GC' else ['C(shrub)','C(fertilizer)','C(context)','C(block)']
 test=effects[(effects.study==study)&(effects.level=='gene')&(effects.analysis=='main')]
 for gene in ['murA','phaC','GH13']:
  if gene not in set(test.feature):continue
  d=md.copy();d['y']=np.log2(x.loc[gene,d.index]+.5);fit=smf.ols('y ~ '+' + '.join(terms),d).fit(cov_type='HC3',use_t=True)
  z=test[test.feature==gene]
  for r in z.itertuples():
   check(f'{study}_{gene}_{r.coefficient}_independent_HC3',np.allclose([r.beta,r.SE,r.p],[fit.params[r.coefficient],fit.bse[r.coefficient],fit.pvalues[r.coefficient]],rtol=1e-6,atol=1e-9))
 # Validate the multivariate engine's univariate F against statsmodels nested models.
 d=md.copy();d['y']=np.log2(x.loc['murA',d.index]+.5)
 f1=smf.ols('y ~ '+' + '.join(terms),d).fit();f0=smf.ols('y ~ '+' + '.join(terms[1:]),d).fit()
 result=test_mv(d.y.to_numpy()[:,None],d,terms,terms[0],99)
 check(study+'_permutation_engine_F_matches_independent_OLS',np.isclose(result['F'],f1.compare_f_test(f0)[0],rtol=1e-7))
 check(study+'_permutation_p_bounds',.01<=result['p']<=1)
 check(study+'_Hellinger_unit_norm',np.allclose(np.square(hellinger(x)).sum(axis=1),1))
for keys,g in effects.groupby(['study','level','analysis']):
 check('BH_'+str(keys),np.allclose(g.q,bh(g.p),equal_nan=True))
vp=pd.read_csv(OUT/'tables/variance_partitioning.tsv',sep='\t')
check('variance_partition_sums',np.allclose(vp[['block_adjusted_R2','unique_design','unique_chemistry','shared','unexplained']].sum(axis=1),1))
h=pd.read_csv(OUT/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
missing=h[h.catalog_gene_id.isna()];missing.to_csv(OUT/'tables/annotated_loci_without_cluster_mapping.tsv',sep='\t',index=False)
check('quantifiable_target_loci_have_cluster_crosswalk',h.loc[h.catalog_gene_id.notna(),'local_cluster_id'].notna().all(),f'{missing.gene_id.nunique()} annotated loci explicitly unavailable for direct quantification')
det=pd.read_csv(OUT/'tables/image_target_detection.tsv',sep='\t');check('153_image_labels_audited',len(det)==153)
q=pd.read_csv(OUT/'tables/mapping_sample_QC.tsv',sep='\t');check('106_valid_reference_libraries',len(q)==106 and (q.reference_RPK>0).all())
for name in ['direct_read_MAG_gene_contributions.tsv']:
 p=OUT/'tables'/name
 if p.exists():
  c=pd.read_csv(p,sep='\t');s=c.groupby(['study','shrub','gene']).fraction_gene_reads.sum();check('direct_carrier_fractions_sum_to_one',np.allclose(s[s>0],1))
rna=OUT/'tables/expanded_DNA_RNA_mapping_QC.tsv'
if rna.exists():
 r=pd.read_csv(rna,sep='\t');check('paired_RNA_DNA_samples',set(r[r.source=='GC_MetaG'].sample_id)==set(r[r.source=='GC_MetaT'].sample_id))
out=pd.DataFrame(checks);out.to_csv(OUT/'tables/validation_checks.tsv',sep='\t',index=False)
print(f'{len(out)} validation checks passed',flush=True)
