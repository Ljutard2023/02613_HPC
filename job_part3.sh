#!/bin/bash
#BSUB -J wall_heat_p3
#BSUB -o logs/part3_%J.out
#BSUB -e logs/part3_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 01:00
#BSUB -q hpc

module load python3/3.9.23
source /zhome/3f/9/223204/projects/.venv/bin/activate
mkdir -p logs

echo "=== Part 3 - Numba JIT ==="
export NUMBA_NUM_THREADS=$LSB_DJOB_NUMPROC
python part3_numba.py 10
echo "=== Done $(date) ==="
