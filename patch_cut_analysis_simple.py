import re

file_path = '/home/pseudonym/Documents/Code/NonlocalProtocol/cut_analysis.py'
with open(file_path, 'r') as f:
    content = f.read()

new_default = """DEFAULT_DATA_DIRS = [
    "Tdis_pfaff5_V0_0_0",
    "Tdis_pfaff5_V0_0_1",
    "Tdis_pfaff5_V0_0_378",
    "Tdis_pfaff5_V0_0_645",
    "Tdis_pfaff5_V0_0_872",
    "Tdis_pfaff5_V0_0_91"
]"""

content = re.sub(r"DEFAULT_DATA_DIRS = \[p.name for p in PathConfigs.DATA.glob\('Tdis_pfaff5_V0_0_\*'\)\]", new_default, content)

with open(file_path, 'w') as f:
    f.write(content)

print("cut_analysis simplified.")
