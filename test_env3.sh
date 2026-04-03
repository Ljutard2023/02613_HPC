#!/bin/bash
#BSUB -J test_env3
#BSUB -o logs/test_env3_%J.out
#BSUB -e logs/test_env3_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 00:05
#BSUB -q hpc

source /zhome/3f/9/223204/projects/.venv/bin/activate
echo "which python3: $(which python3)"
python3 -c "import numpy; print('numpy OK:', numpy.__version__)"
