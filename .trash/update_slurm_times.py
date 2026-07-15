import glob

def main():
    files = (
        glob.glob("Slurm_Scripts/Disorder_Realizations/**/*.slurm", recursive=True) +
        glob.glob("Slurm_Scripts/Batches_To_Run/**/*.slurm", recursive=True)
    )
    
    updated_count = 0
    for file_path in files:
        with open(file_path, "r") as f:
            content = f.read()
        
        if "0-03:00:00" in content:
            new_content = content.replace("0-03:00:00", "0-08:00:00")
            with open(file_path, "w") as f:
                f.write(new_content)
            updated_count += 1
            
    print(f"Successfully updated {updated_count} Slurm files to 8 hours limit.")

if __name__ == "__main__":
    main()
