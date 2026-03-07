import pandas as pd
file_path = '/mnt/e/WMS_selection/Manhattan Extensions/LTBI SCALE Extension Tracker.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None)
df = all_sheets['Ext Gantt']
df_cleaned = df.dropna(how='all').dropna(axis=1, how='all')
md = df_cleaned.to_markdown(index=False)
print(f"Characters in 'Ext Gantt' markdown: {len(md)}")
