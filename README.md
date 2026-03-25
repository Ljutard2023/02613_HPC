# 🏠 Wall Heating – Mini-Projet HPC Python
## Cours 02613 – Python and High Performance Computing

---

## 📋 RÉSUMÉ DU SUJET

Le projet simule la distribution de chaleur dans 4 571 plans de bâtiments suisses.
On modélise les murs intérieurs chauffants (25°C) et les murs porteurs froids (5°C),
puis on calcule comment la chaleur se diffuse dans les pièces via **l'équation de Laplace**,
résolue par la **méthode itérative de Jacobi**.

Le script de référence fourni (`simulate.py`) est trop lent.
**L'objectif est de l'optimiser le plus possible** en utilisant différentes
techniques HPC (parallélisme CPU, Numba JIT, CUDA GPU, CuPy).

---

## 👥 RÉPARTITION DU TRAVAIL (4 PERSONNES)

```
┌─────────────┬──────────────────────────────────────────────────────┐
│  Personne   │  Travail                                             │
├─────────────┼──────────────────────────────────────────────────────┤
│  👩 ALICE   │  Partie 1 : Exploration, visualisation, timing,      │
│  (part1)    │  profiling (Tâches 1–4)                              │
├─────────────┼──────────────────────────────────────────────────────┤
│  👨 BOB     │  Partie 2 : Parallélisation CPU statique & dynamique, │
│  (part2)    │  loi d'Amdahl, graphes speed-up (Tâches 5–6)        │
├─────────────┼──────────────────────────────────────────────────────┤
│  👩 CARLA   │  Partie 3 : Numba JIT CPU, analyse cache, benchmark  │
│  (part3)    │  (Tâche 7)                                           │
├─────────────┼──────────────────────────────────────────────────────┤
│  👨 DAMIEN  │  Partie 4 : GPU CUDA kernel (Numba), CuPy,           │
│  (part4)    │  profiling nsys, analyse finale (Tâches 8–12)       │
└─────────────┴──────────────────────────────────────────────────────┘
```

---

## 📁 STRUCTURE DES FICHIERS

```
wall_heating/
│
├── simulate.py              ← Script de référence (fourni par le cours)
│
├── jacobi_profile.py        ← Script pour kernprof (Tâche 4)
│
├── part1_explore.py         ← PARTIE 1 : Exploration & visualisation
│                               → génère : part1_task1_input_data.png
│                               → génère : part1_task3_simulation_results.png
│
├── part2_parallel.py        ← PARTIE 2 : Parallélisation CPU
│                               → génère : part2_speedup.png
│
├── part3_numba.py           ← PARTIE 3 : Numba JIT CPU
│
├── part4_gpu.py             ← PARTIE 4 : GPU (CUDA + CuPy) + Analyse finale
│                               → génère : part4_task12_analysis.png
│                               → génère : results_all.csv
│
├── create_job_scripts.sh    ← Script qui crée tous les scripts SLURM
│
└── logs/                    ← Dossier pour les sorties des jobs SLURM
```

---

## 🚀 INSTALLATION

Sur le cluster DTU HPC, dans ton environnement Python :

```bash
pip install numpy matplotlib pandas numba cupy-cuda12x line_profiler
```

---

## ▶️ UTILISATION

### Sur ton ordinateur (test local sans GPU) :
```bash
# Partie 1 : exploration
python part1_explore.py

# Partie 2 : parallélisation (ajuster N selon ta machine)
python part2_parallel.py 10 all

# Partie 3 : Numba JIT
python part3_numba.py 5

# Partie 4 : analyse finale (sans GPU)
python part4_gpu.py analyse 10
```

### Sur le cluster DTU HPC :
```bash
# Créer les scripts de job SLURM
bash create_job_scripts.sh

# Soumettre les jobs
sbatch job_part1.sh
sbatch job_part1_kernprof.sh
sbatch job_part2.sh
sbatch job_part3.sh
sbatch job_part4_gpu.sh       ← nécessite un GPU
sbatch job_part4_nsys.sh      ← profiling nsys

# Pour tous les bâtiments (Tâche 12)
sbatch job_part4_all_buildings.sh
```

---

## 📐 EXPLICATIONS DES CONCEPTS CLÉS

### L'équation de Laplace et la méthode de Jacobi
```
Chaque point de la grille est mis à jour comme la MOYENNE de ses 4 voisins :
    u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4

On répète jusqu'à convergence (la grille ne change plus de plus de atol).
```

### Pourquoi c'est lent ?
```
514 × 514 = 264 196 points × 20 000 itérations = 5 milliards d'opérations
```

### Les 4 niveaux d'optimisation
```
1. Référence (NumPy)      : 1×        (base)
2. Parallèle CPU (32 CPUs): ~20×      (multiprocessing)
3. Numba JIT              : ~10-50×   (compilation native)
4. GPU CuPy/CUDA          : ~50-200×  (milliers de threads GPU)
```

---

## 📊 RÉSULTATS ATTENDUS (exemples)

| Méthode              | Temps (10 bâtiments) | Temps estimé (4571) |
|---------------------|---------------------|---------------------|
| Référence NumPy     | ~300s               | ~38h                |
| Parallèle statique  | ~40s                | ~5h                 |
| Parallèle dynamique | ~35s                | ~4.4h               |
| Numba JIT           | ~15s                | ~1.9h               |
| GPU CuPy            | ~5s                 | ~38min              |
| GPU CUDA kernel     | ~8s                 | ~61min              |

*(Ces chiffres sont des estimations – tes résultats varieront selon le matériel)*

---

## 📝 RAPPORT

Le rapport doit être rendu en PDF sur DTU Learn avant le **4 mai 2025**.
Il doit inclure :
- Réponses aux 12 tâches
- Graphes (speed-up, histogrammes)
- Snippets de code pertinents
- Analyse et conclusions

---

## ❓ FAQ DÉBUTANT

**Q : C'est quoi un "job SLURM" ?**
> Sur un cluster HPC partagé, on ne peut pas lancer des calculs lourds directement.
> On "soumet" un script à SLURM (le gestionnaire de ressources) qui l'exécute
> quand un nœud de calcul est disponible.

**Q : Pourquoi la 1ère exécution Numba est lente ?**
> Numba compile le code Python en code machine la première fois (JIT = Just-In-Time).
> C'est comme "installer" le programme → long une fois, mais rapide ensuite.
> C'est pourquoi on fait un "warm-up" avant le benchmark.

**Q : Pourquoi CuPy peut être plus lent que NumPy ?**
> Envoyer les données RAM → mémoire GPU prend du temps (~milliseconde par bâtiment).
> Si le calcul lui-même est court, ce coût fixe domine. Solution : envoyer
> toutes les données d'un coup, calculer tout, récupérer tout (Tâche 10 fix).

**Q : C'est quoi la différence entre statique et dynamique ?**
> Statique : on divise le travail EN AVANCE en parts égales. Simple, mais
>   inefficace si certains bâtiments prennent beaucoup plus de temps que d'autres.
> Dynamique : chaque worker prend un bâtiment à la fois dans une file.
>   Meilleur équilibrage de charge → généralement plus rapide.
