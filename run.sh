#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "Starting Local Parameter Sweeps (8 Jobs Total)"
echo "=================================================="

# Using the absolute path to your Python executable to guarantee no environment errors
PYTHON_EXEC="/home/pseudonym/miniconda3/envs/kwant_final/bin/python"

# --- Vdis1 ---
echo "--- Running Job 1/8: vdis1_b_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis1_b_strdis" --fname "Vdis1.npz" --Lb_pdi 3

echo "--- Running Job 2/8: vdis1_nb_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis1_nb_strdis" --fname "Vdis1.npz" --Lb_pdi 0

# --- Vdis2 ---
echo "--- Running Job 3/8: vdis2_b_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis2_b_strdis" --fname "Vdis2.npz" --Lb_pdi 3

echo "--- Running Job 4/8: vdis2_nb_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis2_nb_strdis" --fname "Vdis2.npz" --Lb_pdi 0

# --- Vdis3 ---
echo "--- Running Job 5/8: vdis3_b_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis3_b_strdis" --fname "Vdis3.npz" --Lb_pdi 3

echo "--- Running Job 6/8: vdis3_nb_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis3_nb_strdis" --fname "Vdis3.npz" --Lb_pdi 0

# --- Vdis4 ---
echo "--- Running Job 7/8: vdis4_b_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis4_b_strdis" --fname "Vdis4.npz" --Lb_pdi 3

echo "--- Running Job 8/8: vdis4_nb_strdis ---"
$PYTHON_EXEC -u main_parallel.py --dirname "vdis4_nb_strdis" --fname "Vdis4.npz" --Lb_pdi 0

echo "=================================================="
echo "All 8 jobs completed successfully!"
echo "=================================================="