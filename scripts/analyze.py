"""Study-specific multivariable relative-abundance and multivariate analyses.

All inference is conditional on recorded design; candidate OSS plot clustering
is a sensitivity analysis until experimental-unit identities are confirmed.
"""
from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT=Path(__file__).resolve().parents[1];RNG=np.random.default_rng(20260910);NPERM=1999
warnings.filterwarnings('ignore',category=RuntimeWarning)
def rd(n):return pd.read_csv(OUT/'inputs'/n,sep='\t',index_col=0)
def save(d,n):d.to_csv(OUT/'tables'/n,sep='\t',index=False)
def bh(p):
 p=np.asarray(p,float);q=np.full(len(p),np.nan);ok=np.isfinite(p)
 if ok.any():q[ok]=multipletests(p[ok],method='fdr_bh')[1]
 return q
def design(md,terms):
 return patsy.dmatrix('1 + '+' + '.join(terms),md,return_type='dataframe') if terms else pd.DataFrame({'Intercept':np.ones(len(md))},index=md.index)
def pinvfit(X,Y):return np.linalg.pinv(np.asarray(X))@Y
def residual(X,Y):return Y-np.asarray(X)@pinvfit(X,Y)
def permutations(md):
 b=md.block.astype(str).to_numpy();return [np.where(b==v)[0] for v in np.unique(b)]
def test_mv(Y,md,terms,term,nperm=NPERM):
 X=design(md,terms);X0=design(md,[t for t in terms if t!=term]);n=len(md)
 rank=np.linalg.matrix_rank(X);r0=np.linalg.matrix_rank(X0);df=rank-r0;dfe=n-rank
 if df<=0 or dfe<=3:return dict(n=n,df=df,df_resid=dfe,F=np.nan,p=np.nan,partial_R2=np.nan,R2_total=np.nan,status='nonestimable')
 E=residual(X,Y);E0=residual(X0,Y);s1=np.square(E).sum();s0=np.square(E0).sum();ss=max(s0-s1,0)
 F=(ss/df)/(s1/dfe);fitted0=Y-E0;groups=permutations(md)
 # For GC treatment tests, retain phase as well as block in exchangeability sets.
 if 'phase' in md and term!='C(phase)':
  g=(md.block.astype(str)+'__'+md.phase.astype(str)).to_numpy();groups=[np.where(g==v)[0] for v in np.unique(g)]
 H=np.eye(n)-np.asarray(X)@np.linalg.pinv(X);H0=np.eye(n)-np.asarray(X0)@np.linalg.pinv(X0)
 ge=0
 for _ in range(nperm):
  ix=np.arange(n)
  for g in groups:ix[g]=RNG.permutation(g)
  yp=fitted0+E0[ix];a=np.square(H@yp).sum();b0=np.square(H0@yp).sum();fp=((b0-a)/df)/(a/dfe)
  ge+=fp>=F-1e-10
 return dict(n=n,df=df,df_resid=dfe,F=F,p=(ge+1)/(nperm+1),partial_R2=ss/s0,R2_total=ss/np.square(Y-Y.mean(0)).sum(),status='conditional_block_permutation')
def regress(Y,md,terms,study,level,analysis,feature_names,cluster=False):
 X=design(md,terms);xx=np.asarray(X);n,k=xx.shape;rank=np.linalg.matrix_rank(xx)
 if rank<k or n-rank<8:return [],dict(study=study,level=level,analysis=analysis,n=n,columns=k,rank=rank,status='skipped_rank_or_residual_df')
 inv=np.linalg.pinv(xx.T@xx);B=inv@xx.T@Y;E=Y-xx@B;h=np.einsum('ij,jk,ik->i',xx,inv,xx)
 A=inv@xx.T
 if cluster:
  labels=md.candidate_plot.to_numpy();gs=np.unique(labels);G=len(gs)
  if G<=k:return [],dict(study=study,level=level,analysis=analysis,n=n,columns=k,rank=rank,status='skipped_too_few_candidate_plots')
  V=np.zeros_like(B)
  for g in gs:
   ix=np.where(labels==g)[0];s=inv@xx[ix].T@E[ix];V+=s*s
  V*=G/(G-1)*(n-1)/(n-k);dof=G-1;method='candidate_plot_cluster_CR1'
 else:V=(A*A)@(E/np.maximum(1-h[:,None],1e-7))**2;dof=n-rank;method='HC3'
 SE=np.sqrt(np.maximum(V,0));T=np.divide(B,SE,out=np.zeros_like(B),where=SE>0);P=2*stats.t.sf(abs(T),dof);crit=stats.t.ppf(.975,dof)
 rows=[]
 for j,c in enumerate(X.columns):
  if c=='Intercept' or c.startswith('C(block)'):continue
  for i,f in enumerate(feature_names):rows.append(dict(study=study,level=level,analysis=analysis,feature=f,coefficient=c,beta=B[j,i],SE=SE[j,i],CI_low=B[j,i]-crit*SE[j,i],CI_high=B[j,i]+crit*SE[j,i],p=P[j,i],n=n,df_resid=dof,method=method))
 return rows,dict(study=study,level=level,analysis=analysis,n=n,columns=k,rank=rank,condition_number=np.linalg.cond(xx),status='fitted',method=method)
def transform(x,pseudo=.5):return np.log2(x.T.to_numpy(float)+pseudo)
def hellinger(x):
 a=x.T.to_numpy(float);return np.sqrt(a/np.maximum(a.sum(1,keepdims=True),1e-20))
def spatial_median(a):
 c=a.mean(0)
 for _ in range(200):
  w=1/np.maximum(np.linalg.norm(a-c,axis=1),1e-10);new=(a*w[:,None]).sum(0)/w.sum()
  if np.linalg.norm(new-c)<1e-9:return new
  c=new
 return c
BASE={'GC':['C(shrub)','C(watering)','C(organic_matter)','C(phase)','C(block)'],
 'OSS':['C(shrub)','C(fertilizer)','C(context)','C(block)']}
md={s:rd(f'{s}_metadata.tsv') for s in BASE}
# Explicit references: reported shrub coefficients are Shrub minus noShrub.
for s,d in md.items():d['shrub']=pd.Categorical(d.shrub,categories=['noShrub','Shrub'])
allrows=[];qc=[];mv=[];disp=[];filterrows=[];allmat={};interactionrows=[]
for study,d in md.items():
 print('Analyzing',study,flush=True)
 for level in ['gene','pathway','mechanism','MAG','target_cluster']:
  name=f'{study}_MAG_abundance.tsv' if level=='MAG' else f'{study}_direct_{level}_TPM.tsv'
  x=rd(name).loc[:,d.index];keep=(x.gt(0).sum(axis=1)>=max(5,int(np.ceil(.1*len(d)))))&(x.std(axis=1)>1e-10)
  for f in x.index:filterrows.append(dict(study=study,level=level,feature=f,prevalence=float(x.loc[f].gt(0).mean()),retained=bool(keep[f])))
  x=x.loc[keep];allmat[(study,level)]=x
  if level=='MAG':x=x.div(x.sum(axis=0))*1e6
  Y=transform(x);rows,q=regress(Y,d,BASE[study],study,level,'main',x.index);allrows+=rows;qc.append(q)
  if study=='OSS':
   rows,q=regress(Y,d,BASE[study],study,level,'main_candidate_plot_sensitivity',x.index,True);allrows+=rows;qc.append(q)
  if level in ['gene','pathway']:
   for pc in [.1,1.0]:
    rows,q=regress(transform(x,pc),d,BASE[study],study,level,f'pseudocount_{pc}',x.index);allrows+=rows;qc.append(q)
   inter=['C(shrub):C(organic_matter)','C(shrub):C(watering)','C(watering):C(phase)'] if study=='GC' else ['C(shrub):C(context)','C(shrub):C(fertilizer)']
   for term in inter:
    rows,q=regress(Y,d,BASE[study]+[term],study,level,'interaction_'+term,x.index);allrows += [r for r in rows if ':' in r['coefficient']];qc.append(q)
  if level in ['gene','pathway','MAG']:
   H=hellinger(x);u,s,v=np.linalg.svd(H-H.mean(0),full_matrices=False);scores=pd.DataFrame(u[:,:2]*s[:2],index=d.index,columns=['PC1','PC2']).join(d.astype(str))
   save(scores.reset_index(),f'{study}_{level}_ordination_scores.tsv')
   fig,ax=plt.subplots(figsize=(7,5));sns.scatterplot(data=scores,x='PC1',y='PC2',hue='shrub',style='phase' if study=='GC' else 'context',ax=ax,s=65)
   ax.set(xlabel=f'PC1 ({s[0]**2/s.dot(s)*100:.1f}%)',ylabel=f'PC2 ({s[1]**2/s.dot(s)*100:.1f}%)',title=f'{study}: {level}, Hellinger PCA');fig.tight_layout();fig.savefig(OUT/'figures'/f'{study}_{level}_ordination.png',dpi=180);plt.close(fig)
   for term in BASE[study]:
    if term=='C(block)':continue
    result=test_mv(H,d,BASE[study],term);mv.append(dict(study=study,level=level,analysis='main',term=term,**result))
    fac=term[2:-1];dist=np.zeros(len(d))
    for group in d[fac].dropna().unique():
     ix=np.where(d[fac].to_numpy()==group)[0];center=spatial_median(H[ix]);dist[ix]=np.sqrt(np.square(H[ix]-center).sum(1))*np.sqrt(len(ix)/max(len(ix)-1,1))
    dr=test_mv(dist[:,None],d,BASE[study],term)
    disp.append(dict(study=study,level=level,factor=fac,**dr,method='adjusted_permutation_distance_to_spatial_median_bias_corrected'))
   if level=='gene':
    for term in inter:
     interactionrows.append(dict(study=study,level=level,analysis='interaction',term=term,**test_mv(H,d,BASE[study]+[term],term)))
  # Comparisons on common observed contexts; no extrapolation to dry roots.
  if level in ['gene','pathway','MAG']:
   subsets={'droughtStart':d.phase=='droughtStart','droughtEnd':d.phase=='droughtEnd'} if study=='GC' else {c:d.context==c for c in sorted(d.context.unique())}
   if study=='OSS':subsets.update({'soil_only':d.sample_type=='Soil','rainy_only':d.season=='Rainy'})
   for label,mask in subsets.items():
    z=d.loc[mask];xx=x.loc[:,z.index];terms=[t for t in BASE[study] if t!='C(phase)' and t!='C(context)']
    terms=[t for t in terms if z[t[2:-1]].nunique()>1]
    if label=='soil_only':terms+=['C(season)']
    if label=='rainy_only':terms+=['C(sample_type)']
    rows,q=regress(transform(xx),z,terms,study,level,'subset_'+label,xx.index);allrows+=rows;qc.append(q)
    if level=='gene':
     for term in terms:
      if term!='C(block)':mv.append(dict(study=study,level=level,analysis='subset_'+label,term=term,**test_mv(hellinger(xx),z,terms,term)))

effects=pd.DataFrame(allrows);effects['q']=effects.groupby(['study','level','analysis']).p.transform(bh)
save(effects,'adjusted_feature_models.tsv');save(pd.DataFrame(qc),'model_design_QC.tsv');save(pd.DataFrame(filterrows),'feature_filtering.tsv')
community=pd.DataFrame(mv+interactionrows);community['q']=community.groupby(['study','analysis']).p.transform(bh);save(community,'adjusted_community_tests.tsv')
dd=pd.DataFrame(disp);dd['q']=dd.groupby('study').p.transform(bh);save(dd,'dispersion_diagnostics.tsv')

# Metadata screening is design-adjusted ONE variable at a time, not mutually adjusted.
envrows=[];envmv=[];envqc=[];corr=[];vp=[];loadings=[]
derived={'antioxidant','exopolysaccharide','nutrient','osmolytes','phytohormone','sum','SUM','rna.antioxidant','rna.exopolysaccharide','rna.nutrient','rna.osmolytes','rna.phytohormone','rna.sum'}
for study,d in md.items():
 print('Metadata screening',study,flush=True)
 nums=[c for c in d.select_dtypes(include='number') if c not in derived]
 meta_corr=d[nums].corr(method='spearman')
 for i,a in enumerate(nums):
  for b in nums[i+1:]:
   if abs(meta_corr.loc[a,b])>=.8:corr.append(dict(study=study,variable1=a,variable2=b,spearman_r=meta_corr.loc[a,b]))
 for c in nums:
  z=d.dropna(subset=[c]).copy();status='tested'
  if len(z)<20 or z[c].nunique()<5:status='skipped_low_n_or_variation'
  if status=='tested':
   z['environment_z']=(z[c]-z[c].mean())/z[c].std();terms=[t for t in BASE[study] if z[t[2:-1]].nunique()>1]
   for level in ['gene','pathway']:
    x=allmat[(study,level)].loc[:,z.index];rows,q=regress(transform(x),z,terms+['environment_z'],study,level,c,x.index)
    envrows += [dict(r,variable=c) for r in rows if r['coefficient']=='environment_z'];status=q['status']
   if status=='fitted':envmv.append(dict(study=study,variable=c,**test_mv(hellinger(allmat[(study,'gene')].loc[:,z.index]),z,terms+['environment_z'],'environment_z')))
  envqc.append(dict(study=study,variable=c,n_complete=len(z),status=status,role='measured_soil_or_plant_associate; may be outcome/mediator'))
 # Low-dimensional chemistry group, chosen without looking at gene associations.
 chem=['Al','Ca','Cu','Fe','K','Mg','Mn','Na','P','Zn'] if study=='GC' else ['pH','perC','perN']
 z=d.dropna(subset=chem).copy();terms=[t for t in BASE[study] if z[t[2:-1]].nunique()>1]
 if len(z)>=20:
  A=z[chem].astype(float);A=(A-A.mean())/A.std();u,s,v=np.linalg.svd(A,full_matrices=False);z['chemPC1']=u[:,0]*s[0];z['chemPC2']=u[:,1]*s[1]
  for i,c in enumerate(chem):loadings.append(dict(study=study,variable=c,PC1=v[0,i],PC2=v[1,i],PC1_variance=s[0]**2/s.dot(s),PC2_variance=s[1]**2/s.dot(s)))
  for level in ['gene','pathway','MAG']:
   H=hellinger(allmat[(study,level)].loc[:,z.index]);TSS=np.square(H-H.mean(0)).sum();n=len(z)
   def adj(ts):
    X=design(z,ts);rank=np.linalg.matrix_rank(X);R=1-np.square(residual(X,H)).sum()/TSS
    return 1-(1-R)*(n-1)/(n-rank)
   B=['C(block)'];D=[t for t in terms if t!='C(block)'];E=['chemPC1','chemPC2'];b=adj(B);a=adj(B+D)-b;e=adj(B+E)-b;both=adj(B+D+E)-b
   vp.append(dict(study=study,level=level,n=n,block_adjusted_R2=b,unique_design=both-e,unique_chemistry=both-a,shared=a+e-both,unexplained=1-adj(B+D+E),chemistry_variables=';'.join(chem),note='Adjusted R2 fractions can be negative; chemistry is associational'))
   envmv.append(dict(study=study,variable='chemistry_PC1_PC2_joint',level=level,**test_mv(H,z,terms+['chemPC1 + chemPC2'],'chemPC1 + chemPC2')))
er=pd.DataFrame(envrows)
if len(er):er['q']=er.groupby(['study','level']).p.transform(bh);save(er,'metadata_gene_pathway_associations.tsv')
em=pd.DataFrame(envmv)
if len(em):em['q']=em.groupby('study').p.transform(bh);save(em,'metadata_community_associations.tsv')
save(pd.DataFrame(envqc),'metadata_screening_QC.tsv');save(pd.DataFrame(corr),'correlated_metadata_pairs.tsv');save(pd.DataFrame(vp),'variance_partitioning.tsv');save(pd.DataFrame(loadings),'chemistry_PCA_loadings.tsv')

# Carrier contributions and direct/inferred concordance.
hits=pd.read_csv(OUT/'tables/image_gene_locus_cluster_MAG_taxonomy.tsv',sep='\t')
contrib=[];concord=[]
for study,d in md.items():
 m=rd(f'{study}_MAG_abundance.tsv');cp=rd('MAG_gene_copy_number.tsv');inf=rd(f'{study}_inferred_gene_abundance.tsv');direct=rd(f'{study}_direct_gene_TPM.tsv')
 for gene in direct.index.intersection(inf.index):
  rho,p=stats.spearmanr(direct.loc[gene],inf.loc[gene,direct.columns]);concord.append(dict(study=study,gene=gene,spearman_r=rho,p=p,n=len(direct.columns)))
  z=cp[gene]*m.mean(axis=1);total=z.sum()
  for mag,value in z[z>0].items():
   row=hits[hits.MAG==mag].iloc[0];contrib.append(dict(study=study,gene=gene,MAG=mag,phylum=row.get('phylum',''),genus=row.get('genus',''),mean_inferred_contribution=value,fraction_inferred_gene=value/total if total else np.nan,copy_number=cp.loc[mag,gene]))
save(pd.DataFrame(contrib),'MAG_gene_contributions.tsv');save(pd.DataFrame(concord),'direct_inferred_gene_concordance.tsv')

# Harmonized shrub response: same gene, same relative scale, separate study fits.
cross=[]
for level in ['gene','pathway']:
 a=effects[(effects.level==level)&(effects.analysis=='main')&(effects.coefficient=='C(shrub)[T.Shrub]')]
 g=a[a.study=='GC'].set_index('feature');o=a[a.study=='OSS'].set_index('feature')
 for gene in g.index.intersection(o.index):
  b1,b2=g.loc[gene,'beta'],o.loc[gene,'beta'];se1,se2=g.loc[gene,'SE'],o.loc[gene,'SE'];se=np.sqrt(se1**2+se2**2)
  cross.append(dict(level=level,feature=gene,GC_beta=b1,OSS_beta=b2,GC_q=g.loc[gene,'q'],OSS_q=o.loc[gene,'q'],same_direction=b1*b2>0,difference_beta=b1-b2,difference_p=2*stats.norm.sf(abs(b1-b2)/se),interpretation='different experimental contexts; not a causal project effect'))
cr=pd.DataFrame(cross);cr['difference_q']=cr.groupby('level').difference_p.transform(bh);save(cr,'cross_project_shrub_comparison.tsv')
for study in BASE:
 a=effects[(effects.study==study)&(effects.level=='pathway')&(effects.analysis=='main')]
 if len(a):
  pv=a.pivot(index='feature',columns='coefficient',values='beta');qs=a.pivot(index='feature',columns='coefficient',values='q');labels=pv.map(lambda v:f'{v:.2f}') if hasattr(pv,'map') else pv.applymap(lambda v:f'{v:.2f}')
  labels=labels+qs.applymap(lambda q:'*' if q<.05 else '')
  fig,ax=plt.subplots(figsize=(12,10));sns.heatmap(pv,cmap='vlag',center=0,annot=labels,fmt='',ax=ax,cbar_kws={'label':'Adjusted log2(TPM + 0.5) effect'})
  ax.set_title(f'{study}: pathway marker abundance (* BH q < 0.05)');fig.tight_layout();fig.savefig(OUT/'figures'/f'{study}_pathway_effects.png',dpi=180);plt.close(fig)
 print(study,'main significant genes',len(effects[(effects.study==study)&(effects.level=='gene')&(effects.analysis=='main')&(effects.q<.05)]),flush=True)
print('Analysis complete',flush=True)
