#!/bin/bash
#BSUB -J wall_heat_p3
#BSUB -o logs/part3_%J.out
#BSUB -e logs/part3_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 01:00
#BSUB -q hpc
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p logs

echo "=== Part 3 - Numba JIT ==="
python part3_numba.py 10
echo "=== Done $(date) ==="
