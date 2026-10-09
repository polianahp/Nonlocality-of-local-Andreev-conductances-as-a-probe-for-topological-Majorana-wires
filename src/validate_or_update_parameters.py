#!/usr/bin/env python3
"""
validate_or_update_parameters.py

Audit, modernize, and refactor YAML parameter files for disorder realizations.
1. Safely archives legacy flat YAML files into Legacy_Flat/
2. Generates modernized, schema-compliant YAML files organized by benchmark disorder strength:
   Inputs/Parameters/Disorder_Realizations/V0_<strength>/disorder_realization_<idx>.yaml
3. Validates all generated configurations against Pydantic v2 SimulationConfig.
"""

import os
import sys
import shutil
import glob
from pathlib import Path
import yaml
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.parameter_handler import SimulationConfig

DISORDER_PARAM_DIR = ROOT_DIR / "Inputs" / "Parameters" / "Disorder_Realizations"
LEGACY_DIR = DISORDER_PARAM_DIR / "Legacy_Flat"

# 5 Active benchmark disorder amplitudes with tailored mu ranges
BENCHMARK_DISORDERS = [
    ("V0_0_1", 0.1, -2.0, 2.0),
    ("V0_0_378", 0.378, -2.0, 2.0),
    ("V0_0_645", 0.645, -1.0, 3.5),
    ("V0_0_872", 0.872, 0.0, 4.5),
    ("V0_1_2", 1.2, 0.0, 4.5),
]

NUM_REALIZATIONS = 150

def archive_legacy_flat_files():
    """Moves legacy flat disorder_realization_*.yaml files to Legacy_Flat/."""
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    flat_files = glob.glob(str(DISORDER_PARAM_DIR / "disorder_realization_*.yaml"))
    if flat_files:
        print(f"Archiving {len(flat_files)} flat YAML files to {LEGACY_DIR}...")
        for f in flat_files:
            dst = LEGACY_DIR / Path(f).name
            shutil.move(f, dst)
        print("Legacy flat files archived safely.")
    else:
        print("No flat YAML files found at top level to archive.")

def generate_disorder_strength_subfolders():
    """Generates modernized YAML parameter files separated by disorder strength with Upoints=15."""
    print("Generating modernized parameter files across 5 active disorder strengths (Upoints=15)...")
    total_created = 0

    for dir_name, v0_val, mu_min, mu_max in BENCHMARK_DISORDERS:
        sub_dir = DISORDER_PARAM_DIR / dir_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(NUM_REALIZATIONS):
            yaml_path = sub_dir / f"disorder_realization_{idx}.yaml"

            config_dict = {
                # Disorder Management
                "dirname": f"Disorder_Realizations/{dir_name}/disorder_realization_{idx}",
                "realization_index": idx,
                "lambda_dis": 4.5,
                "V0": float(v0_val),

                # Geometry & Physics Scale Keys (strictly enforced)
                "Ls": 300,
                "a0": 100.0,
                "barrier0": 2.0,

                # Sweep Ranges (Tailored per disorder regime)
                "mu_max": float(mu_max),
                "mu_min": float(mu_min),
                "mu_dist": 0.02,

                "Vz_max": 1.4,
                "Vz_min": 0.0,
                "Vz_dist": 0.02,

                # Simulation Resolution (Optimized Upoints=15 for positive barrier asymmetry)
                "Upoints": 15,
                "num_engs": 51,
                "acceleration_type": "parallel",
                "weight_threshold": 0.8,
                "separation_threshold": 0.65,

                # Ground Truth Topological Invariant
                "calc_pfaffian": True,
            }

            with open(yaml_path, "w") as fp:
                yaml.dump(config_dict, fp, default_flow_style=False, sort_keys=False)

            total_created += 1

    print(f"Successfully generated {total_created} parameter files across {len(BENCHMARK_DISORDERS)} directories.")

def validate_all_generated_parameters():
    """Validates all generated parameter YAML files against SimulationConfig."""
    print("Validating all parameter files against Pydantic v2 SimulationConfig...")
    errors = []
    validated_count = 0

    for dir_name, v0_val, mu_min, mu_max in BENCHMARK_DISORDERS:
        sub_dir = DISORDER_PARAM_DIR / dir_name
        yaml_files = sorted(sub_dir.glob("disorder_realization_*.yaml"))
        if len(yaml_files) != NUM_REALIZATIONS:
            errors.append(f"Expected {NUM_REALIZATIONS} files in {sub_dir}, found {len(yaml_files)}")

        for yf in yaml_files:
            try:
                raw_dict = OmegaConf.to_container(OmegaConf.load(str(yf)), resolve=True)
                cfg = SimulationConfig(**raw_dict)
                assert cfg.Ls == 300, f"Ls mismatch in {yf}"
                assert cfg.a0 == 100.0, f"a0 mismatch in {yf}"
                assert cfg.barrier0 == 2.0, f"barrier0 mismatch in {yf}"
                assert cfg.Upoints == 15, f"Upoints mismatch in {yf}: expected 15, got {cfg.Upoints}"
                assert cfg.calc_pfaffian is True, f"calc_pfaffian is False in {yf}"
                assert abs(cfg.V0 - v0_val) < 1e-6, f"V0 mismatch in {yf}: expected {v0_val}, got {cfg.V0}"
                assert abs(cfg.mu_min - mu_min) < 1e-6, f"mu_min mismatch in {yf}: expected {mu_min}, got {cfg.mu_min}"
                assert abs(cfg.mu_max - mu_max) < 1e-6, f"mu_max mismatch in {yf}: expected {mu_max}, got {cfg.mu_max}"
                validated_count += 1
            except Exception as e:
                errors.append(f"Error in {yf}: {str(e)}")

    if errors:
        print(f"FAILED validation! {len(errors)} errors found:")
        for err in errors[:10]:
            print(f"  {err}")
        raise RuntimeError("Parameter validation failed.")
    else:
        print(f"SUCCESS: All {validated_count} parameter files validated cleanly against SimulationConfig!")

if __name__ == "__main__":
    archive_legacy_flat_files()
    generate_disorder_strength_subfolders()
    validate_all_generated_parameters()
