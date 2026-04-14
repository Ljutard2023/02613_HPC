#!/bin/bash
#BSUB -J wall_heat_p2
#BSUB -o logs/part2_%J.out
#BSUB -e logs/part2_%J.err
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 02:00
#BSUB -q hpc

mkdir -p logs

echo "=== Part 2 - CPU parallelisation ==="
echo "Available CPUs: $LSB_DJOB_NUMPROC  |  $(date)"

# Tasks 5 + 6: static and dynamic speed-up on 100 buildings
python part2_parallel.py 100 all

echo "=== Done $(date) ==="
