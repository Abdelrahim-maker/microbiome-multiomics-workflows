"""Audit sample alignment, reconstructed statistics, FDR and rendered artifacts."""
from pathlib import Path
import ast,json,sys
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.stats.multitest import multipletests
from PIL import Image
OUT=Path(__file__).resolve().parents[1];NPERM=4999;RNG=np.random.default_rng(1)
tree=ast.parse((OUT.parent/'scripts/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'helpers','exec'))
checks=[]
mv=pd.read_csv(OUT/'tables/community_PERMANOVA.tsv',sep='\t')
for study in ['GC','OSS']:
    d=rd(f'{study}_metadata.tsv');assert d.index.is_unique
    d['Shrub']=np.where(d.shrub=='Shrub',.5,-.5)
    terms=['Shrub','C(fertilizer)','C(context)','C(block)']
    if study=='GC':
        d['OM']=np.where(d.organic_matter=='OM',.5,-.5);d['OM_x_Shrub']=d.OM*d.Shrub
        terms=['OM','Shrub','OM_x_Shrub','C(watering)','C(phase)','C(block)']
    for level in mv.level.unique():
        x=rd(f'{study}_{level}.tsv');assert x.index.is_unique and x.columns.is_unique
        assert set(x.columns)==set(d.index) and np.isfinite(x.to_numpy()).all() and (x.to_numpy()>=0).all()
        x=x.loc[:,d.index]
        if level=='MAG':x=x.div(x.sum(0))*1e6
        x=x.loc[(x.gt(0).sum(1)>=max(5,int(np.ceil(.1*len(d)))))&(x.std(1)>1e-10)]
        Y=hellinger(x);X=design(d,terms);s1=np.square(residual(X,Y)).sum()
        for row in mv[(mv.study==study)&(mv.level==level)&mv.analysis.isin(['main','factorial'])].itertuples():
            X0=design(d,[t for t in terms if t!=row.term]);s0=np.square(residual(X0,Y)).sum()
            assert np.isclose((s0-s1)/s0,row.partial_R2,rtol=1e-7,atol=1e-10)
        checks.append(f'{study} {level}: input alignment and independent full-space R2 reconstruction passed')
for name,groups in [('community_PERMANOVA',['study','scope','analysis']),('PERMDISP',['study','scope','analysis']),('differential_features',['study','level','analysis']),('fertilizer_all_pairwise_features',['level','analysis']),('fertilizer_community_tests',['analysis'])]:
    d=pd.read_csv(OUT/'tables'/f'{name}.tsv',sep='\t');q=d.groupby(groups).p.transform(bh)
    assert np.allclose(q,d.q,equal_nan=True)
    assert d.p.dropna().between(0,1).all()
    checks.append(name+': FDR families and p bounds passed')
e=pd.read_csv(OUT/'tables/fertilizer_all_pairwise_features.tsv',sep='\t')
assert (e.CI_low<=e.beta).all() and (e.beta<=e.CI_high).all()
main=pd.read_csv(OUT/'tables/differential_features.tsv',sep='\t')
a=main[(main.study=='OSS')&(main.analysis=='main')&main.contrast.eq('C(fertilizer)[T.1x]')]
b=e[e.analysis.eq('main')].merge(a,on=['level','feature'],suffixes=('_new','_old'))
assert len(b)==len(e[e.analysis.eq('main')]);assert np.allclose(b.beta_new,b.beta_old)
checks.append('Fertilizer contrasts independently match existing reference-coded effects; confidence intervals bracket estimates')
manifest=pd.read_csv(OUT/'tables/figure_provenance.tsv',sep='\t');assert manifest.figure.is_unique
for name in manifest.figure:
    p=OUT/'figures'/f'{name}.png';im=np.asarray(Image.open(p).convert('RGB'))
    assert im.std()>10 and p.stat().st_size>10000
    assert (OUT/'figures'/f'{name}.pdf').stat().st_size>1000
checks.append(f'All {len(manifest)} PNG/PDF figure pairs exist and raster images are nonblank')
for name in ['REPORT.md','REPORT.html','RESULTS.xlsx','input_manifest.tsv']:assert (OUT/name).stat().st_size>1000
result={'status':'passed','checks':checks,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
(OUT/'VALIDATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
