"""Separate GC target-gene ordinations comparing OM within each shrub group."""
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
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parents[1]
DEST=OUT/'figures/GC_OM_by_shrub';DEST.mkdir(exist_ok=True)
RNG=np.random.default_rng(20260912);NPERM=4999
tree=ast.parse((OUT/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'analysis_helpers','exec'))
metadata=rd('GC_metadata.tsv');abundance=rd('GC_direct_gene_TPM.tsv')
filters=pd.read_csv(OUT/'tables/feature_filtering.tsv',sep='\t')
keep=filters[(filters.study=='GC')&(filters.level=='gene')&filters.retained].feature
abundance=abundance.loc[keep]
results=[];panels=[]
for shrub,label in [('Shrub','Shrub samples only'),('noShrub','No-shrub samples only')]:
 md=metadata[metadata.shrub==shrub].copy();x=abundance.loc[:,md.index]
 assert md.organic_matter.nunique()==2 and len(md)==24
 Y=hellinger(x);centered=Y-Y.mean(0)
 u,s,v=np.linalg.svd(centered,full_matrices=False);scores=u[:,:2]*s[:2]
 variance=s[:2]**2/(s*s).sum()*100
 # Orient each independently fitted axis consistently by its largest loading.
 for j in range(2):
  sign=1 if v[j,np.argmax(abs(v[j]))]>=0 else -1
  scores[:,j]*=sign;v[j]*=sign
 terms=['C(organic_matter)','C(watering)','C(phase)','C(block)']
 res=test_mv(Y,md,terms,'C(organic_matter)')
 results.append(dict(shrub_group=shrub,test='OM_adjusted_composition',**res))
 distance=np.zeros(len(md))
 for group in ['OM','noOM']:
  ix=np.flatnonzero(md.organic_matter.to_numpy()==group)
  distance[ix]=np.linalg.norm(Y[ix]-spatial_median(Y[ix]),axis=1)*np.sqrt(len(ix)/(len(ix)-1))
 results.append(dict(shrub_group=shrub,test='OM_adjusted_dispersion',**test_mv(distance[:,None],md,terms,'C(organic_matter)')))
 table=pd.DataFrame(scores,index=md.index,columns=['PC1','PC2']).join(md[['organic_matter','watering','phase','block']])
 table.to_csv(DEST/f'GC_{shrub}_sample_scores.tsv',sep='\t',index_label='sample_id')
 pd.DataFrame({'gene_or_family':x.index,'PC1_loading':v[0],'PC2_loading':v[1]}).to_csv(DEST/f'GC_{shrub}_gene_loadings.tsv',sep='\t',index=False)
 panels.append((shrub,label,table,variance))
tests=pd.DataFrame(results);tests['BH_q']=tests.groupby('test').p.transform(bh)
tests.to_csv(DEST/'OM_composition_and_dispersion_tests.tsv',sep='\t',index=False)
colors={'OM':'#168275','noOM':'#CE663E'};markers={'droughtStart':'o','droughtEnd':'^'}
def draw(ax,panel):
 shrub,label,d,variance=panel
 for om in ['OM','noOM']:
  z=d[d.organic_matter==om];centroid=z[['PC1','PC2']].mean()
  for _,r in z.iterrows():ax.plot([r.PC1,centroid.PC1],[r.PC2,centroid.PC2],color=colors[om],alpha=.18,lw=.7,zorder=1)
  for phase,marker in markers.items():
   a=z[z.phase==phase];ax.scatter(a.PC1,a.PC2,c=colors[om],marker=marker,s=85,edgecolors='white',linewidth=.7,zorder=3)
  ax.scatter(centroid.PC1,centroid.PC2,c=colors[om],marker='X',s=190,edgecolors='black',linewidth=.8,zorder=4)
 t=tests[(tests.shrub_group==shrub)&(tests.test=='OM_adjusted_composition')].iloc[0]
 ax.set_title(f'{label}\nOM n={(d.organic_matter=="OM").sum()} | No OM n={(d.organic_matter=="noOM").sum()}',fontsize=13,weight='bold')
 ax.set_xlabel(f'PC1 ({variance[0]:.1f}%)');ax.set_ylabel(f'PC2 ({variance[1]:.1f}%)')
 ax.text(.02,.02,f'Adjusted OM test: partial R² = {t.partial_R2:.3f}\np = {t.p:.4f}; BH q = {t.BH_q:.4f}',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'),va='bottom')
 ax.margins(.2);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12)
 handles=[Line2D([],[],marker='o',linestyle='',color=colors[g],label='OM' if g=='OM' else 'No OM',markersize=8) for g in ['OM','noOM']]
 handles += [Line2D([],[],marker=m,linestyle='',color='#596169',label='Drought start' if p=='droughtStart' else 'Drought end',markersize=7) for p,m in markers.items()]
 handles += [Line2D([],[],marker='X',linestyle='',color='#596169',label='Group centroid',markersize=9)]
 ax.legend(handles=handles,loc='best',fontsize=8,framealpha=.9)
for panel in panels:
 fig,ax=plt.subplots(figsize=(8,7));draw(ax,panel)
 fig.suptitle('GC carbon-circle target genes: Hellinger PCA',fontsize=14)
 fig.text(.5,.01,'Ordination is unadjusted; OM test adjusts for watering, phase and block.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.035,1,.95])
 for ext in ['png','pdf']:fig.savefig(DEST/f'GC_OM_vs_noOM_{panel[0]}.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(15,7))
for ax,panel in zip(axes,panels):draw(ax,panel)
fig.suptitle('GC carbon-circle target genes: OM effects within shrub groups',fontsize=15,weight='bold')
fig.text(.5,.01,'Separate Hellinger PCAs; axes are fitted independently. Tests use all dimensions and adjust for watering, phase and block.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.035,1,.95])
for ext in ['png','pdf']:fig.savefig(DEST/f'GC_OM_vs_noOM_separate_shrub_groups.{ext}',dpi=250,bbox_inches='tight')
plt.close(fig)
(DEST/'README.md').write_text('''# GC: OM versus no OM within shrub groups

Two independent ordinations of directly quantified carbon-circle gene/family profiles:
24 shrub samples (12 OM, 12 no OM) and 24 no-shrub samples (12 OM, 12 no OM).
The same 108 gene/family labels retained in the GC analysis are used in each panel.
Sample-wise target profiles are normalized to sum to one and square-root transformed
(Hellinger), then centered and decomposed by PCA. Axis percentages refer to variance
within that shrub subset; axes cannot be compared directly between panels.

Colors identify OM treatment, symbols identify phase, and X symbols are arithmetic
group centroids. Lines connect samples to their group centroids, not repeated samples.
Plots show unadjusted profiles. Marginal full-dimensional OM tests adjust for watering,
phase and block, using 4,999 Freedman–Lane residual permutations within block × phase.
BH correction is across the two composition tests; dispersion tests are corrected
separately. Dispersion tests use bias-corrected distances to group spatial medians.
Pot identities are unconfirmed; permutation inference remains conditional on the
recorded design. Source sample scores, gene loadings and exact tests are included.

Reproduce from the repository root with:
`MPLCONFIGDIR=/tmp/carbon-circle-mpl OPENBLAS_NUM_THREADS=1 protein_cluster_target_analysis_263MAGs/joint_rpca_env/bin/python "fnal results/scripts/ordination_OM_by_shrub.py"`
''')
print(tests[['shrub_group','test','n','partial_R2','p','BH_q']].to_string(index=False))
