import pandas as pd
file_path = '/mnt/e/WMS_selection/Manhattan Extensions/LTBI SCALE Extension Tracker.xlsx'
try:
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in all_sheets.items():
        original_shape = df.shape
        df_cleaned = df.dropna(how='all').dropna(axis=1, how='all')
        print(f"Sheet: '{sheet_name}' | Original Shape: {original_shape} | Cleaned Shape: {df_cleaned.shape}")
except Exception as e:
    print(f"Error: {e}")
