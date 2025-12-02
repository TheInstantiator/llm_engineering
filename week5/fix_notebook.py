import json

notebook_path = 'week5/day1-mine.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell with imports
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the import cell
        if any('from dotenv import load_dotenv' in line for line in source):
            # Check if load_dotenv() is already called
            if not any('load_dotenv()' in line for line in source):
                print("Found target cell. Adding load_dotenv()...")
                # Add newline and load_dotenv()
                if not source[-1].endswith('\n'):
                    source[-1] += '\n'
                source.append('\n')
                source.append('load_dotenv()')
                print("Modified cell source.")
                break
            else:
                print("load_dotenv() already present.")
                break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
