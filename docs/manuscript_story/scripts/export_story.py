"""Use the existing system IO environment for workbook and HTML export."""
from pathlib import Path
import pandas as pd
import markdown
OUT=Path(__file__).resolve().parents[1]
body=markdown.markdown((OUT/'REPORT.md').read_text(),extensions=['tables','fenced_code'])
(OUT/'REPORT.html').write_text('<html><head><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px sans-serif}img{max-width:100%}td,th{padding:5px;border:1px solid #ddd}table{border-collapse:collapse;display:block;overflow:auto}</style></head><body>'+body+'</body></html>')
with pd.ExcelWriter(OUT/'RESULTS.xlsx') as writer:
    for name in ['community_PERMANOVA','PERMDISP','differential_significance_summary','fertilizer_all_pairwise_features','fertilizer_significance_summary','fertilizer_community_tests','category_definitions','gene_carrier_redundancy','paired_cross_level_R2_bootstrap','redundancy_treatment_models','model_QC','fertilizer_model_QC','figure_provenance','legacy_OSS_reanalysis','legacy_OSS_dispersion']:
        pd.read_csv(OUT/'tables'/f'{name}.tsv',sep='\t').to_excel(writer,sheet_name=name[:31],index=False)
    e=pd.read_csv(OUT/'tables/differential_features.tsv',sep='\t')
    e[~e.level.str.contains('inferred')].to_excel(writer,sheet_name='direct_MAG_feature_effects',index=False)
