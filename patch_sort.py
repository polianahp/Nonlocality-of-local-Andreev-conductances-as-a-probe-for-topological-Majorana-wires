import re

file_path = '/home/pseudonym/Documents/Code/NonlocalProtocol/Scripts-Notebooks/transport_gap_analysis.py'

with open(file_path, 'r') as f:
    content = f.read()

new_sort_key = r"""# Remove duplicates 
def sort_key(x):
    import re
    m = re.search(r'V0_(\d+)_(\d+)', x)
    if m:
        try:
            return float(f"{m.group(1)}.{m.group(2)}")
        except:
            return 999.0
    try:
        return int(x.split('_')[2])
    except:
        return 999.0"""

content = content.replace("# Remove duplicates \ndef sort_key(x):\n    try:\n        return int(x.split('_')[2])\n    except:\n        return 999", new_sort_key)

with open(file_path, 'w') as f:
    f.write(content)

print("Sort key patched.")
