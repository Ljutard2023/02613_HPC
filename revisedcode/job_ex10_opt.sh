#!/bin/bash
#BSUB -J wall_heat_task8
#BSUB -o logs/task10_opt_%J.out
#BSUB -e logs/task10_opt_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 00:30
#BSUB -q c02613
#BSUB -gpu "num=1:mode=exclusive_process"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p logs

python exercise_10_opt.py

