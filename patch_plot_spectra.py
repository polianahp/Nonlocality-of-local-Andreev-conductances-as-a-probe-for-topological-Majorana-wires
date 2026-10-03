file_path = '/home/pseudonym/Documents/Code/NonlocalProtocol/Scripts-Notebooks/plot_spectra.py'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace('default="Data/Tdis_pfaff3"', 'default=str(PathConfigs.DATA / "Tdis_pfaff5")')

with open(file_path, 'w') as f:
    f.write(content)

print("plot_spectra patched.")
