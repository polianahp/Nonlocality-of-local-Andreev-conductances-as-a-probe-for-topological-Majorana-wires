import os

amplitudes = ["0.1", "0.0"]

base_yaml_path = "/home/pseudonym/Documents/Code/NonlocalProtocol/Inputs/Parameters/Tdis.yaml"
base_slurm_path = "/home/pseudonym/Documents/Code/NonlocalProtocol/Slurm_Scripts/Tdis.slurm"

with open(base_yaml_path, 'r') as f:
    yaml_lines = f.readlines()
    
with open(base_slurm_path, 'r') as f:
    slurm_text = f.read()

for amp in amplitudes:
    # 1. Create new YAML
    new_yaml_path = f"/home/pseudonym/Documents/Code/NonlocalProtocol/Inputs/Parameters/Tdis_V0_{amp}.yaml"
    
    new_yaml_lines = []
    for line in yaml_lines:
        if line.startswith("dirname:"):
            new_yaml_lines.append(f'dirname: "Tdis_pfaff5_V0_{amp}"\n')
            new_yaml_lines.append(f'V0: {amp}\n')
        else:
            new_yaml_lines.append(line)
            
    with open(new_yaml_path, 'w') as f:
        f.writelines(new_yaml_lines)
        
    # 2. Create new Slurm script
    new_slurm_path = f"/home/pseudonym/Documents/Code/NonlocalProtocol/Slurm_Scripts/Tdis_V0_{amp}.slurm"
    
    # modify job name and config path
    new_slurm_text = slurm_text.replace(
        '#SBATCH -J  Tdis_sim #Job Name',
        f'#SBATCH -J  Tdis_V0_{amp} #Job Name'
    ).replace(
        '--config_path Parameters/Tdis.yaml',
        f'--config_path Parameters/Tdis_V0_{amp}.yaml'
    )
    
    with open(new_slurm_path, 'w') as f:
        f.write(new_slurm_text)

print("Additional files generated.")
