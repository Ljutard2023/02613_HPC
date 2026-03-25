#!/bin/bash
# =============================================================================
# JOB SCRIPTS SLURM pour le cluster DTU HPC
# =============================================================================
# Ces scripts servent à soumettre les calculs sur le cluster.
# Sur un cluster HPC, on ne lance PAS les calculs directement (python script.py)
# car ça consomme les ressources des nœuds de connexion partagés.
# On SOUMET un job au gestionnaire SLURM qui l'exécute sur un nœud dédié.
#
# Commandes SLURM utiles :
#   sbatch job_xxx.sh     → soumettre un job
#   squeue -u $USER       → voir ses jobs en attente/en cours
#   scancel <JOBID>       → annuler un job
#   seff <JOBID>          → efficacité du job (après fin)
# =============================================================================

cat > job_part1.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_heat_p1
#SBATCH --output=logs/part1_%j.out     # %j = numéro du job
#SBATCH --error=logs/part1_%j.err
#SBATCH --ntasks=1                     # 1 seul processus
#SBATCH --cpus-per-task=1              # 1 cœur CPU
#SBATCH --mem=4G                       # 4 Go de RAM
#SBATCH --time=01:00:00                # 1 heure max
#SBATCH --partition=hpc                # partition à utiliser sur DTU

# Charger les modules nécessaires
module load python3/3.11.3
source /path/to/venv/bin/activate   # adapter selon votre environnement

mkdir -p logs

echo "=== PARTIE 1 – Exploration des données ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Nœud   : $SLURMD_NODENAME"
echo "Date   : $(date)"
echo ""

# Tâche 1, 3 : visualisation (besoin de peu de ressources)
python part1_explore.py

# Tâche 2 : timing sur 10 bâtiments
echo ""
echo "=== Timing sur 10 bâtiments ==="
time python simulate.py 10

echo ""
echo "=== Terminé $(date) ==="
EOF

cat > job_part1_kernprof.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_kernprof
#SBATCH --output=logs/kernprof_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=hpc

module load python3/3.11.3
source /path/to/venv/bin/activate

mkdir -p logs

echo "=== TÂCHE 4 – Profiling kernprof ==="
# -l = line profiling, -v = verbose output
kernprof -l -v jacobi_profile.py > logs/kernprof_output.txt 2>&1
cat logs/kernprof_output.txt
EOF

cat > job_part2.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_heat_p2
#SBATCH --output=logs/part2_%j.out
#SBATCH --error=logs/part2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32             # IMPORTANT : demander assez de cœurs !
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=hpc

module load python3/3.11.3
source /path/to/venv/bin/activate

mkdir -p logs

echo "=== PARTIE 2 – Parallélisation CPU ==="
echo "CPUs disponibles : $SLURM_CPUS_PER_TASK"
echo ""

# Tâche 5 + 6 : speed-up statique ET dynamique sur 100 bâtiments
# 'all' = génère les graphes + CSV
python part2_parallel.py 100 all

echo "=== Terminé $(date) ==="
EOF

cat > job_part3.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_heat_p3
#SBATCH --output=logs/part3_%j.out
#SBATCH --error=logs/part3_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # pour Numba parallèle
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=hpc

module load python3/3.11.3
source /path/to/venv/bin/activate

mkdir -p logs

echo "=== PARTIE 3 – Numba JIT ==="

# Variable d'environnement Numba pour utiliser tous les cœurs
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

python part3_numba.py 10

echo "=== Terminé $(date) ==="
EOF

cat > job_part4_gpu.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_heat_p4
#SBATCH --output=logs/part4_%j.out
#SBATCH --error=logs/part4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1                   # IMPORTANT : demander 1 GPU !
#SBATCH --time=01:00:00
#SBATCH --partition=gpuv100            # partition GPU sur DTU

module load python3/3.11.3
module load cuda/12.1                  # charger CUDA
source /path/to/venv/bin/activate

mkdir -p logs

echo "=== PARTIE 4 – GPU CUDA + CuPy ==="
nvidia-smi                             # affiche les infos du GPU
echo ""

# Tâches 8 et 9 : CUDA kernel + CuPy
python part4_gpu.py all 10

echo "=== Terminé $(date) ==="
EOF

cat > job_part4_nsys.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_nsys
#SBATCH --output=logs/nsys_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpuv100

module load python3/3.11.3
module load cuda/12.1
source /path/to/venv/bin/activate

mkdir -p logs

echo "=== TÂCHE 10 – Profiling nsys ==="
# --trace=cuda = tracer les appels CUDA
# --trace=nvtx = tracer les annotations NVTX
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=logs/profile_cupy \
    python part4_gpu.py cupy 5

echo "Rapport généré : logs/profile_cupy.nsys-rep"
echo "Ouvrir avec : nsys-ui logs/profile_cupy.nsys-rep"
EOF

cat > job_part4_all_buildings.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=wall_all
#SBATCH --output=logs/all_buildings_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=hpc               # ou gpuv100 si CuPy disponible

module load python3/3.11.3
source /path/to/venv/bin/activate

mkdir -p logs

echo "=== TÂCHE 12 – Traitement de tous les bâtiments ==="
echo "Nombre de CPUs : $SLURM_CPUS_PER_TASK"

# Utiliser la parallélisation dynamique (la plus rapide sur CPU)
python part2_parallel.py 4571 dynamic > results_all.csv

# Puis analyse
python part4_gpu.py analyse 4571

echo "=== Terminé $(date) ==="
EOF

echo "=== Tous les scripts de job ont été créés ==="
echo ""
echo "Pour soumettre :"
echo "  sbatch job_part1.sh"
echo "  sbatch job_part2.sh"
echo "  sbatch job_part3.sh"
echo "  sbatch job_part4_gpu.sh"
