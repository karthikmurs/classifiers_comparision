import json
import re

# Path to the notebook
notebook_path = r'c:\Users\karth\git_ws\classifiers_comparision\prompt_III-Ajit-Final.ipynb'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Track changes
changes_made = 0

# Iterate through cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        # Get the source (either string or list of strings)
        if isinstance(cell['source'], list):
            source = ''.join(cell['source'])
            is_list = True
        else:
            source = cell['source']
            is_list = False
        
        # Check if cell contains plot_roc_curve calls
        if 'plot_roc_curve(' in source:
            # Comment out lines that call plot_roc_curve
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if 'plot_roc_curve(' in line:
                    # Check if it's not already commented
                    stripped = line.lstrip()
                    if not stripped.startswith('#'):
                        # Add comment prefix while maintaining indentation
                        indent = len(line) - len(stripped)
                        new_lines.append(' ' * indent + '# ' + stripped)
                        changes_made += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            new_source = '\n'.join(new_lines)
            
            # Set the source back
            if is_list:
                cell['source'] = new_source.split('\n')
                # Add newlines back to each line except the last
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                                  for i, line in enumerate(cell['source'])]
            else:
                cell['source'] = new_source

# Save the fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Fixed {changes_made} plot_roc_curve() calls by commenting them out")
print("Notebook saved successfully")
