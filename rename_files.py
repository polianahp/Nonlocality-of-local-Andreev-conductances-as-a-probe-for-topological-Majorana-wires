import os

amplitudes = ["0.378", "0.645", "0.872", "0.91", "0.1", "0.0"]

for amp in amplitudes:
    amp_us = amp.replace('.', '_')
    
    # 1. Update YAML files
    old_yaml = f"/home/pseudonym/Documents/Code/NonlocalProtocol/Inputs/Parameters/Tdis_V0_{amp}.yaml"
    new_yaml = f"/home/pseudonym/Documents/Code/NonlocalProtocol/Inputs/Parameters/Tdis_V0_{amp_us}.yaml"
    
    if os.path.exists(old_yaml):
        with open(old_yaml, 'r') as f:
            yaml_content = f.read()
        
        # update dirname
        yaml_content = yaml_content.replace(f'dirname: "Tdis_pfaff5_V0_{amp}"', f'dirname: "Tdis_pfaff5_V0_{amp_us}"')
        
        with open(new_yaml, 'w') as f:
            f.write(yaml_content)
        os.remove(old_yaml)
        
    # 2. Update Slurm scripts
    old_slurm = f"/home/pseudonym/Documents/Code/NonlocalProtocol/Slurm_Scripts/Tdis_V0_{amp}.slurm"
    new_slurm = f"/home/pseudonym/Documents/Code/NonlocalProtocol/Slurm_Scripts/Tdis_V0_{amp_us}.slurm"
    
    if os.path.exists(old_slurm):
        with open(old_slurm, 'r') as f:
            slurm_content = f.read()
        
        # update job name and config path
        slurm_content = slurm_content.replace(f'Tdis_V0_{amp}', f'Tdis_V0_{amp_us}')
        
        with open(new_slurm, 'w') as f:
            f.write(slurm_content)
        os.remove(old_slurm)

print("Files renamed and updated successfully.")
