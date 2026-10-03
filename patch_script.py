import re

file_path = '/home/pseudonym/Documents/Code/NonlocalProtocol/Scripts-Notebooks/transport_gap_analysis.py'

with open(file_path, 'r') as f:
    content = f.read()

# Add import for PathConfigs
if 'from src.config import PathConfigs' not in content:
    content = content.replace('from src.parameter_handler import ConfigManager', 
                              'from src.parameter_handler import ConfigManager\nfrom src.config import PathConfigs')

# Update DATA_DIRS
content = re.sub(r"DATA_DIRS = glob.glob\('Data/disorder_realization_\*_results'\)", 
                 "DATA_DIRS = [str(p) for p in PathConfigs.DATA.glob('Tdis_pfaff5_V0_0_*')]", content)

# Update OUTPUT_DIR
content = re.sub(r"OUTPUT_DIR = 'Data/Transport_Gap_Analysis'", 
                 "OUTPUT_DIR = str(PathConfigs.DATA / 'Transport_Gap_Analysis')", content)

with open(file_path, 'w') as f:
    f.write(content)

print("Script patched successfully.")
