#!/bin/bash
#BSUB -J wall_kernprof
#BSUB -o logs/kernprof_%J.out
#BSUB -e logs/kernprof_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -q hpc
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p logs

echo "=== Task 4 - kernprof line-by-line profiling ==="
kernprof -l -v jacobi_profile.py > logs/kernprof_output.txt 2>&1
cat logs/kernprof_output.txt
