"""Gene drivers, carrier structure and measured perC associations."""
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
OUT=Path(__file__).resolve().parents[1]; BASE=OUT.parent
NPERM=4999
tree=ast.parse((BASE/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'helpers','exec'))
tree2=ast.parse((OUT/'scripts/analyze_story.py').read_text())
exec(compile(ast.Module(body=[n for n in tree2.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'story_helpers','exec'))
md={s:rd(f'{s}_metadata.tsv') for s in ['GC','OSS']}
for s,d in md.items():
 d['Shrub']=np.where(d.shrub=='Shrub',.5,-.5)
 if s=='GC': d['OM']=np.where(d.organic_matter=='OM',.5,-.5); d['OM_x_Shrub']=d.OM*d.Shrub
GC=['OM','Shrub','OM_x_Shrub','C(watering)','C(phase)','C(block)']; OSS=['Shrub','C(fertilizer)','C(context)','C(block)']
cats=pd.read_csv(OUT/'tables/category_definitions.tsv',sep='\t')
labels=pd.read_csv(BASE/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')[['image_gene','mechanism']].drop_duplicates()
target=sorted(set(g for x in cats.image_labels.dropna() for g in x.split(';')).intersection(rd('GC_carbon_gene.tsv').index))
rows=[]
for study,d in md.items():
 x=rd(f'{study}_carbon_gene.tsv').loc[target,d.index]; terms=GC if study=='GC' else OSS
 contrasts=({'OM_average':{'OM':1},'Shrub_average':{'Shrub':1},'OM_x_Shrub':{'OM_x_Shrub':1}} if study=='GC' else {'Shrub':{'Shrub':1},'fertilizer':{'C(fertilizer)[T.1x]':1}})
 r,_=covariance_model(transform(x),d,terms,contrasts,study,'carbon_gene','main',x.index); rows+=r
e=pd.DataFrame(rows); e['q']=e.groupby(['study','contrast']).p.transform(bh); e=e.merge(labels,left_on='feature',right_on='image_gene',how='left'); e.to_csv(OUT/'tables/carbon_gene_driver_effects.tsv',sep='\t',index=False)
for study in ['GC','OSS']:
 for contrast in sorted(e.loc[e.study==study,'contrast'].unique()):
  z=e[(e.study==study)&(e.contrast==contrast)].sort_values('beta')
  fig,ax=plt.subplots(figsize=(9,max(4,len(z)*.25))); y=np.arange(len(z)); col=np.where(z.q<.05,np.where(z.beta>=0,'#b2182b','#2166ac'),'#bdbdbd')
  ax.errorbar(z.beta,y,xerr=1.96*z.SE,fmt='none',ecolor='.4',alpha=.7); ax.scatter(z.beta,y,c=col,s=30); ax.axvline(0,color='black',lw=.8); ax.set_yticks(y,z.feature); ax.set_xlabel('Adjusted log2(TPM + 0.5) effect; colored = q < 0.05'); ax.set_title(f'{study}: individual carbon genes, {contrast}'); fig.tight_layout(); fig.savefig(OUT/'figures'/f'{study}_carbon_gene_drivers_{contrast}.png',dpi=180); fig.savefig(OUT/'figures'/f'{study}_carbon_gene_drivers_{contrast}.pdf'); plt.close(fig)
# Four carrier metrics: prevalence, effective diversity, dominance and potential.
cc=pd.read_csv(OUT/'inputs/MAG_category_copy_number.tsv',sep='\t',index_col=0); out=[]
for study,d in md.items():
 m=rd(f'{study}_MAG.tsv').reindex(cc.index,fill_value=0).loc[:,d.index]
 for category in cc.columns:
  a=cc[category].to_numpy()[:,None]*m.to_numpy(); total=a.sum(0); p=np.divide(a,total,where=total>0,out=np.zeros_like(a));
  for j,sample in enumerate(d.index): out.append(dict(study=study,sample_id=sample,category=category,functional_prevalence=int((a[:,j]>0).sum()),effective_carrier_diversity=(1/(p[:,j]**2).sum() if total[j]>0 else np.nan),carrier_dominance=(p[:,j].max() if total[j]>0 else np.nan),category_potential=total[j]))
summary=pd.DataFrame(out); summary.to_csv(OUT/'tables/carrier_function_structure.tsv',sep='\t',index=False)
models=[]
for study,d in md.items():
 z=summary[summary.study==study].set_index('sample_id'); terms=GC if study=='GC' else OSS; contrasts=({'OM_average':{'OM':1},'Shrub_average':{'Shrub':1},'OM_x_Shrub':{'OM_x_Shrub':1}} if study=='GC' else {'Shrub':{'Shrub':1},'fertilizer':{'C(fertilizer)[T.1x]':1}})
 for category in z.category.unique():
  zz=z[z.category==category].reindex(d.index)
  for metric in ['functional_prevalence','effective_carrier_diversity','carrier_dominance','category_potential']:
   y=zz[metric].to_numpy(float); ok=np.isfinite(y)
   if ok.sum()<20: continue
   r,_=covariance_model(y[ok,None],d.loc[ok],terms,contrasts,study,'carrier_'+metric,'main',[metric]); models += [dict(x,category=category) for x in r]
mm=pd.DataFrame(models); mm['q']=mm.groupby(['study','level','contrast']).p.transform(bh); mm.to_csv(OUT/'tables/carrier_structure_models.tsv',sep='\t',index=False)
# perC is concentration, not stock or change.
cr=[]
for study,d in md.items():
 y=pd.to_numeric(d.perC,errors='coerce'); masks={'all':y.notna()};
 if study=='OSS': masks['soil_only']=y.notna()&(d.sample_type=='Soil')
 for subset,mask in masks.items():
  z=d.loc[mask]; yy=y.loc[mask]
  if len(z)<15: continue
  terms=[t for t in (GC if study=='GC' else OSS) if z[t[2:-1] if t.startswith('C(') else t].nunique()>1]; X=design(z,terms).to_numpy(); B=np.linalg.pinv(X)@yy.to_numpy(); res=yy.to_numpy()-X@B; df=len(z)-np.linalg.matrix_rank(X); V=np.diag(np.linalg.pinv(X.T@X))*np.sum(res**2)/df
  for j,col in enumerate(design(z,terms).columns):
   if col=='Intercept' or col.startswith('C(block)'): continue
   p=2*stats.t.sf(abs(B[j]/np.sqrt(V[j])),df); cr.append(dict(study=study,subset=subset,analysis='perC_treatment',term=col,beta=B[j],SE=np.sqrt(V[j]),p=p,n=len(z)))
  for category in summary[summary.study==study].category.unique():
   trait=summary[(summary.study==study)&(summary.category==category)].set_index('sample_id').reindex(z.index).effective_carrier_diversity; ok=trait.notna()
   if ok.sum()<15: continue
   zz=z.loc[ok]; cy=yy.loc[ok]; A=design(zz,terms).to_numpy(); trait=(trait.loc[ok]-trait.loc[ok].mean())/trait.loc[ok].std(); AA=np.column_stack([A,trait.to_numpy()]); b=np.linalg.pinv(AA)@cy.to_numpy(); rr=cy.to_numpy()-AA@b; df2=len(zz)-np.linalg.matrix_rank(AA); se=np.sqrt(np.sum(rr**2)/df2*np.diag(np.linalg.pinv(AA.T@AA))[-1]); p=2*stats.t.sf(abs(b[-1]/se),df2); cr.append(dict(study=study,subset=subset,analysis='perC_trait',term=category,beta=b[-1],SE=se,p=p,n=len(zz)))
c=pd.DataFrame(cr); c['q']=c.groupby(['study','subset','analysis']).p.transform(bh); c.to_csv(OUT/'tables/perC_carbon_associations.tsv',sep='\t',index=False)
# Focused gene -> MAG carriers for FDR-supported OM x shrub genes.
hits=pd.read_csv(BASE/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
gene_int=e[(e.study=='GC')&(e.contrast=='OM_x_Shrub')&(e.q<.05)][['feature','beta','CI_low','CI_high','q','mechanism']].rename(columns={'feature':'image_gene','beta':'gene_interaction_beta','q':'gene_interaction_q'})
mag=pd.read_csv(OUT/'tables/differential_features.tsv',sep='\t')
mag=mag[(mag.study=='GC')&(mag.level=='MAG')&(mag.analysis=='factorial')&(mag.contrast=='OM_x_Shrub')][['feature','beta','CI_low','CI_high','q']].rename(columns={'feature':'MAG','beta':'MAG_interaction_beta','q':'MAG_interaction_q'})
carrier=gene_int.merge(hits[['image_gene','MAG','gene_id','phylum','genus','classification']].drop_duplicates(),on='image_gene').merge(mag,on='MAG',how='left')
carrier['MAG_FDR_supported_interaction']=carrier.MAG_interaction_q<.05
carrier=carrier.sort_values(['image_gene','MAG_interaction_q','MAG_interaction_beta'],key=lambda s:s.abs() if s.name=='MAG_interaction_beta' else s)
carrier.to_csv(OUT/'tables/OM_x_Shrub_responsive_gene_MAG_carriers.tsv',sep='\t',index=False)
focused=carrier[carrier.image_gene.str.match(r'^(alg|psl|pel|pga|bcs|wz|glg|pha|ots|tre)',case=False,na=False)].copy()
focused.to_csv(OUT/'tables/OM_x_Shrub_responsive_EPS_storage_gene_MAG_carriers.tsv',sep='\t',index=False)
selected=focused[focused.MAG_FDR_supported_interaction].copy()
if len(selected):
 # Keep all supported carrier MAGs visible; columns show functions they encode.
    shown=selected.groupby('MAG').MAG_interaction_beta.first().abs().sort_values(ascending=False).head(60).index
    grid=selected[selected.MAG.isin(shown)].pivot_table(index='MAG',columns='image_gene',values='MAG_interaction_beta',aggfunc='first').reindex(shown)
    fig,ax=plt.subplots(figsize=(max(8,len(grid.columns)*1.2),max(5,len(grid)*.22)))
    import seaborn as sns
    sns.heatmap(grid,cmap='RdBu_r',center=0,mask=grid.isna(),ax=ax,cbar_kws={'label':'MAG OM x shrub effect; log2 abundance scale'})
    ax.set_title('FDR-supported MAG carriers of OM x shrub-responsive EPS and storage genes');ax.set_ylabel('MAG');ax.set_xlabel('carbon gene')
    fig.tight_layout();fig.savefig(OUT/'figures/GC_OM_x_Shrub_responsive_gene_MAG_carriers.png',dpi=180);fig.savefig(OUT/'figures/GC_OM_x_Shrub_responsive_gene_MAG_carriers.pdf');plt.close(fig)
# Compact carrier-structure visualization for every defined mechanism.
z=mm[(mm.study=='GC')&mm.contrast.isin(['OM_average','Shrub_average','OM_x_Shrub'])].copy()
if len(z):
    fig,axes=plt.subplots(1,3,figsize=(16,7),sharey=True)
    for ax,metric in zip(axes,['functional_prevalence','effective_carrier_diversity','carrier_dominance']):
        a=z[z.feature==metric].copy();p=a.pivot(index='category',columns='contrast',values='beta');q=a.pivot(index='category',columns='contrast',values='q');ann=p.copy().astype(str)
        for i in p.index:
            for j in p.columns:ann.loc[i,j]=f'{p.loc[i,j]:.2g}\nq={q.loc[i,j]:.2g}'
        sns.heatmap(p,cmap='RdBu_r',center=0,annot=ann,fmt='',annot_kws={'size':6},ax=ax,cbar=False);ax.set_title(metric.replace('_',' '));ax.set_xlabel('')
    fig.suptitle('GC carrier structure: adjusted treatment effects',y=1.02);fig.tight_layout();fig.savefig(OUT/'figures/GC_carrier_structure_effects.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'figures/GC_carrier_structure_effects.pdf',bbox_inches='tight');plt.close(fig)
(OUT/'CARBON_FOLLOWUP.md').write_text('''# Carbon follow-up

## 1. Individual genes driving each mechanism

`carbon_gene_driver_effects.tsv` contains every target-gene effect, standard error, 95% confidence interval, p and BH q value. The forest plots are `GC_carbon_gene_drivers_OM_average`, `GC_carbon_gene_drivers_Shrub_average`, and `GC_carbon_gene_drivers_OM_x_Shrub` (plus OSS shrub and fertilizer plots). The GC plots cover EPS/export, glycogen, PHA, trehalose and the other defined carbon categories. Colored points have q < 0.05.

## 2. MAGs carrying responding carbon genes

`OM_x_Shrub_responsive_gene_MAG_carriers.tsv` is the complete genomic bridge: each row links an FDR-supported OM-by-shrub carbon gene to its MAG, locus, phylum, genus, gene interaction effect, and MAG interaction effect with q values. `OM_x_Shrub_responsive_EPS_storage_gene_MAG_carriers.tsv` is the focused EPS/export, glycogen, PHA and trehalose subset. `GC_OM_x_Shrub_responsive_gene_MAG_carriers.png` shows rows where both the selected carbon gene and carrying MAG have FDR-supported interaction effects. It should be read as treatment-associated carrier turnover, not in situ gene acquisition or gene expression.

## 3. Carrier prevalence, diversity and dominance

`carrier_function_structure.tsv` reports each sample and mechanism. Functional prevalence is the number of detected MAG carriers. Effective carrier diversity is the inverse Simpson diversity of abundance-weighted carrier contributions. Carrier dominance is the largest carrier's fraction of category potential. Total potential is the copy-number-weighted abundance sum. `carrier_structure_models.tsv` supplies adjusted treatment estimates and `GC_carrier_structure_effects.png` summarizes the primary GC effects. A stable prevalence with lower diversity and higher dominance means concentration into fewer abundant carriers rather than functional loss.

## 4. Measured soil carbon

`perC_carbon_associations.tsv` relates measured `perC` concentration to treatment and carrier diversity, separately for GC and OSS soil samples. It is not SOC stock because bulk density and depth were unavailable, and it is not Delta SOC because verified initial/final unit pairings were unavailable. These are conditional associations, not causal mediation or a carbon budget.

## Next steps

Obtain bulk density, sampling depth, verified repeated pot/plot identities, and respiration or CO2 data. Then calculate SOC stock and Delta SOC and test whether treatment-associated carrier and gene responses predict measured carbon change.
''')
print('Carbon follow-up complete')
