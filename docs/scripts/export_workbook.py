"""Export portable TSV sheets with the environment providing openpyxl."""
import sys,json
import pandas as pd
with pd.ExcelWriter(sys.argv[2],engine='openpyxl') as writer:
 for name,path in json.load(open(sys.argv[1])).items():
  d=pd.read_csv(path,sep='\t');d.to_excel(writer,sheet_name=name,index=False)
  ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
  for cells in ws.iter_cols(min_row=1,max_row=min(ws.max_row,80)):
   ws.column_dimensions[cells[0].column_letter].width=min(55,max(12,max(len(str(c.value or '')) for c in cells)+2))
