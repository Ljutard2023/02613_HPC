#!/bin/bash
#BSUB -J wall_heat_p1
#BSUB -o logs/part1_%J.out
#BSUB -e logs/part1_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 01:00
#BSUB -q hpc

mkdir -p logs

echo "=== Part 1 - Data exploration ==="
echo "Job ID : $LSB_JOBID  |  Node : $LSB_HOSTS  |  $(date)"

python part1_explore.py

echo ""
echo "=== Timing reference script on 10 buildings ==="
time python simulate.py 10

echo "=== Done $(date) ==="
