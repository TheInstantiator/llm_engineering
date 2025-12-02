import json

notebook_path = 'week5/day2-mine.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell with tiktoken code
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the tiktoken cell
        if any('encoding = tiktoken.encoding_for_model(MODEL)' in line for line in source):
            print("Found target cell.")
            
            # Create new source code with try-except block
            new_source = [
                "# How many tokens in all the documents?\n",
                "\n",
                "try:\n",
                "    encoding = tiktoken.encoding_for_model(MODEL)\n",
                "except KeyError:\n",
                "    print(\"Model not found in tiktoken, falling back to cl100k_base\")\n",
                "    encoding = tiktoken.get_encoding(\"cl100k_base\")\n",
                "\n",
                "tokens = encoding.encode(entire_knowledge_base)\n",
                "token_count = len(tokens)\n",
                "print(f\"Total tokens for {MODEL}: {token_count:,}\")"
            ]
            
            cell['source'] = new_source
            print("Modified cell source.")
            break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
