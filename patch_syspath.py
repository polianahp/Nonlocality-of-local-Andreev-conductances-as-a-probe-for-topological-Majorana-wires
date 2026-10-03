import re

file_path = '/home/pseudonym/Documents/Code/NonlocalProtocol/Scripts-Notebooks/transport_gap_analysis.py'
with open(file_path, 'r') as f:
    content = f.read()

import_block = """import sys
from pathlib import Path
sys.path.append("/home/pseudonym/Documents/Code/azure-quantum-tgp/notebooks")
from yield_analysis import analyze_2"""

content = content.replace("from yield_analysis import analyze_2", import_block)

# Also ensure it can import src (since it's run from Scripts-Notebooks)
# usually running `python Scripts-Notebooks/transport_gap_analysis.py` from root works for `src`,
# but if run from inside Scripts-Notebooks it needs the parent dir.
src_path_add = """sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.tgp_adapter import TGPAdapter"""

content = content.replace("from src.tgp_adapter import TGPAdapter", src_path_add)

with open(file_path, 'w') as f:
    f.write(content)

print("Imports patched.")
