"""All fertilizer contrasts and shrub interaction, with explicit uncertainty."""
from pathlib import Path
import ast
import itertools
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.stats.multitest import multipletests
OUT=Path(__file__).resolve().parents[1]
RNG=np.random.default_rng(20260914); NPERM=4999
for path in [OUT.parent/'scripts/analyze.py',OUT/'scripts/analyze_story.py']:
    tree=ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(path),'exec'))
d=rd('OSS_metadata.tsv'); d['Shrub']=np.where(d.shrub=='Shrub',.5,-.5)
levels=sorted(d.fertilizer.unique()); rows=[]; qc=[]; mv=[]
for level in ['MAG','carbon_gene','carbon_pathway','carbon_category']:
    x=rd(f'OSS_{level}.tsv').loc[:,d.index]
    if level=='MAG': x=x.div(x.sum(0))*1e6
    x=x.loc[(x.gt(0).sum(1)>=max(5,int(np.ceil(.1*len(d)))))&(x.std(1)>1e-10)]
    for interaction in [False,True]:
        terms=['Shrub','C(fertilizer)','C(context)','C(block)']
        if interaction: terms+=['Shrub:C(fertilizer)']
        X=design(d,terms); contrasts={}
        for a,b in itertools.combinations(levels,2):
            for shrub in ([0] if not interaction else [-.5,.5]):
                grid=pd.concat([d.iloc[[0]].copy(),d.iloc[[0]].copy()],ignore_index=True)
                grid['fertilizer']=[a,b];grid['Shrub']=shrub
                xx=np.asarray(patsy.build_design_matrices([X.design_info],grid)[0])
                label=f'{b} minus {a}'+(' | '+('Shrub' if shrub>0 else 'noShrub') if interaction else ' | adjusted average')
                contrasts[label]=dict(zip(X.columns,xx[1]-xx[0]))
        if interaction:
            for a,b in itertools.combinations(levels,2):
                plus=contrasts[f'{b} minus {a} | Shrub'];minus=contrasts[f'{b} minus {a} | noShrub']
                contrasts[f'{b} minus {a} | shrub difference']= {c:plus[c]-minus[c] for c in X.columns}
        for cluster in [False,True]:
            analysis=('interaction' if interaction else 'main')+('_candidate_plot' if cluster else '')
            r,q=covariance_model(transform(x),d,terms,contrasts,'OSS',level,analysis,x.index,cluster)
            rows+=r;qc.append(q)
        term='Shrub:C(fertilizer)' if interaction else 'C(fertilizer)'
        mv.append(dict(study='OSS',level=level,analysis='interaction' if interaction else 'main',term=term,**test_mv(hellinger(x),d,terms,term)))
e=pd.DataFrame(rows);e['q']=e.groupby(['level','analysis']).p.transform(bh);save(e,'fertilizer_all_pairwise_features.tsv')
m=pd.DataFrame(mv);m['q']=m.groupby('analysis').p.transform(bh);save(m,'fertilizer_community_tests.tsv')
save(pd.DataFrame(qc),'fertilizer_model_QC.tsv')
print('Fertilizer comparisons complete',flush=True)
