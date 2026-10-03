import re

file_path = '/home/pseudonym/Documents/Code/NonlocalProtocol/cut_analysis.py'
with open(file_path, 'r') as f:
    content = f.read()

new_default = "DEFAULT_DATA_DIRS = [p.name for p in PathConfigs.DATA.glob('Tdis_pfaff5_V0_0_*')]"

content = re.sub(r'DEFAULT_DATA_DIRS = \["Tdis_pfaff5"\](?: #.*)?', new_default, content)

# Check if there's any absolute path requirement or other things.
# Looking at main():
# for d in data_dirs_to_process:
#     p_dir = Path(d)
#     if not p_dir.is_absolute():
#         p_dir = PathConfigs.DATA / d
# So passing the names is perfectly fine!

with open(file_path, 'w') as f:
    f.write(content)

print("cut_analysis patched.")
