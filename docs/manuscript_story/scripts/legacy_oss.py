"""Reanalyze every historical OSS ordination view on its original joint scores."""
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent.parent
RNG=np.random.default_rng(20260915);NPERM=4999
tree=ast.parse((OUT.parent/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'helpers','exec'))
source=ROOT/'protein_cluster_target_analysis_263MAGs/joint_RPCA_MAG_protein_clusters/tables/OSS_joint_scores_metadata.tsv'
d=pd.read_csv(source,sep='\t'); jobs=[];results=[];diagnostics=[]
allterms=['C(shrub)','C(season)','C(sample_type)','C(fertilizer)','C(block)']
for focal in ['shrub','season','sample_type','fertilizer']:
    jobs.append((f'OSS_legacy_all_adjusted_{focal}',d,allterms,f'C({focal})',focal,True))
for cn in [False,True]:
    for fixed in ['season','sample_type','fertilizer']:
        for value in sorted(d[fixed].unique()):
            z=d[d[fixed]==value].copy();ts=[t for t in allterms if t!=f'C({fixed})']
            if cn:z=z.dropna(subset=['carbon','nitrogen']);ts+=['carbon','nitrogen']
            jobs.append((f'OSS_legacy_shrub_within_{fixed}_{value}_'+('CN' if cn else 'no_CN'),z,ts,'C(shrub)','shrub',True))
z=d.dropna(subset=['carbon','nitrogen'])
jobs.append(('OSS_legacy_all_adjusted_shrub_CN',z,allterms+['carbon','nitrogen'],'C(shrub)','shrub',True))
for values,z in d.groupby(['fertilizer','season','sample_type']):
    jobs.append(('OSS_legacy_fixed_'+'_'.join(values),z,['C(shrub)','C(block)'],'C(shrub)','shrub',False))
panels={}
for name,z,terms,term,focal,adjusted in jobs:
    terms=[t for t in terms if (z[t[2:-1]].nunique()>1 if t.startswith('C(') else z[t].nunique()>1)]
    Y=z[['PC1','PC2','PC3']].to_numpy()
    if len(z)<5 or term not in terms:
        r=dict(n=len(z),p=np.nan,partial_R2=np.nan,status='nonestimable_small_or_missing_group')
        coords=Y
    else:
        r=test_mv(Y,z,terms,term)
        coords=residual(design(z,[t for t in terms if t!=term]),Y) if adjusted else Y
        dist=np.zeros(len(z))
        for group in z[focal].unique():
            ix=np.flatnonzero(z[focal].to_numpy()==group)
            dist[ix]=np.linalg.norm(Y[ix]-spatial_median(Y[ix]),axis=1)*np.sqrt(len(ix)/max(len(ix)-1,1))
        diagnostics.append(dict(view=name,**test_mv(dist[:,None],z,terms,term)))
    results.append(dict(view=name,factor=focal,adjusted_for=' + '.join(t for t in terms if t!=term),**r))
    zz=z.copy();zz['display1']=coords[:,0];zz['display2']=coords[:,1];panels[name]=(zz,focal,adjusted)
r=pd.DataFrame(results);r['q']=bh(r.p);r.to_csv(OUT/'tables/legacy_OSS_reanalysis.tsv',sep='\t',index=False)
dd=pd.DataFrame(diagnostics);dd['q']=bh(dd.p);dd.to_csv(OUT/'tables/legacy_OSS_dispersion.tsv',sep='\t',index=False)
manifest=[]
for row in r.itertuples():
    z,focal,adjusted=panels[row.view];fig,ax=plt.subplots(figsize=(8.5,6.5))
    sns.scatterplot(data=z,x='display1',y='display2',hue=focal,palette='colorblind',s=65,ax=ax)
    ax.set(title=row.view.replace('OSS_legacy_','OSS: ').replace('_',' '),xlabel=('Nuisance-residual ' if adjusted else '')+'joint PC1',ylabel=('Nuisance-residual ' if adjusted else '')+'joint PC2')
    dg=dd[dd.view==row.view];dq=dg.q.iloc[0] if len(dg) else np.nan
    note=f'n={row.n}; partial R2={row.partial_R2:.3f}; p={row.p:.4g}; q={row.q:.4g}; dispersion q={dq:.4g}\nAdjusted for: {row.adjusted_for}\nReanalysis: 4,999 block-restricted permutations; all three original joint axes. Status: {row.status}'
    fig.subplots_adjust(bottom=.22);fig.text(.07,.025,note,fontsize=7.5)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{row.view}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    z.to_csv(OUT/'tables'/f'{row.view}_display_scores.tsv',sep='\t',index=False)
    manifest.append(dict(figure=row.view,source=str(source.relative_to(ROOT))+'; tables/legacy_OSS_reanalysis.tsv; tables/legacy_OSS_dispersion.tsv'))
pd.DataFrame(manifest).to_csv(OUT/'tables/legacy_figure_provenance.tsv',sep='\t',index=False)
print(f'Completed {len(manifest)} previous OSS views',flush=True)
